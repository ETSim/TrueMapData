#!/usr/bin/env python3
""".

TMD to 3D Model Converter

Advanced command-line tool to convert TMD height map files to various 3D model formats
with additional features like cropping and normal map generation.
"""

import os
import sys
from enum import Enum
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich import print as rprint
import numpy as np

from tmd.processor import TMDProcessor
from tmd.utils.processing import crop_height_map
from tmd.exporters.image.image_io import load_heightmap, save_heightmap  # Updated import
from tmd.exporters.model import (
    convert_heightmap_to_stl,
    convert_heightmap_to_obj,
    convert_heightmap_to_ply,
    convert_heightmap_to_gltf,
    convert_heightmap_to_glb,
    convert_heightmap_to_threejs,
    convert_heightmap_to_usdz
)
from tmd.exporters.image import convert_heightmap_to_normal_map

# For adaptive mesh support
from tmd.exporters.model.adaptive_mesh import convert_heightmap_to_adaptive_mesh

from tmd.exporters.model.backends import ModelBackend, _check_backend_available

# Initialize Typer app and Rich console
app = typer.Typer(help="Convert TMD height map files to 3D model formats and more.")
console = Console()

# Define format options as an enum for better validation
class Format(str, Enum):
    stl = "stl"
    obj = "obj"
    ply = "ply"
    gltf = "gltf"
    glb = "glb"
    threejs = "threejs"
    usdz = "usdz"
    normal_map = "normal_map"
    heightmap = "heightmap"

class ImportType(str, Enum):
    tmd = "tmd"
    image = "image"
    exr = "exr"
    numpy = "numpy"
    auto = "auto"

def process_tmd_file(input_file: str) -> Optional[dict]:
    """.

    Process a TMD file and return the data dictionary.
    Returns None if processing fails.
    """
    processor = TMDProcessor(input_file)
    data = processor.process()
    if not data:
        rprint(f"[bold red]Error:[/bold red] Could not process TMD file {input_file}")
        return None
    return data

def load_heightmap_from_file(input_file: str, import_type: ImportType = ImportType.auto) -> Optional[np.ndarray]:
    """.

    Load a heightmap from a file based on the specified import type.
    
    Args:
        input_file: Path to the input file
        import_type: Type of file to import (auto, tmd, image, exr, numpy)
        
    Returns:
        numpy.ndarray: 2D array of height values or None if loading failed
    """
    try:
        # Determine file type if auto
        if import_type == ImportType.auto:
            ext = os.path.splitext(input_file)[1].lower()
            if ext == '.tmd':
                import_type = ImportType.tmd
            elif ext == '.exr':
                import_type = ImportType.exr
            elif ext in ['.npy', '.npz']:
                import_type = ImportType.numpy
            else:
                import_type = ImportType.image
        
        # Load based on type
        if import_type == ImportType.tmd:
            data = process_tmd_file(input_file)
            if data and 'height_map' in data:
                return data['height_map']
            return None
        elif import_type in [ImportType.image, ImportType.exr, ImportType.numpy]:
            return load_heightmap(input_file, normalize=True)
        else:
            rprint(f"[bold red]Error:[/bold red] Unknown import type: {import_type}")
            return None
    except Exception as e:
        rprint(f"[bold red]Error:[/bold red] Failed to load heightmap: {e}")
        return None

def generate_common_params(height_map, output_file: str, z_scale: float, base_height: float) -> dict:
    """.

    Generate a common parameter dictionary for conversion functions.
    """
    return {
        "height_map": height_map,
        "filename": output_file,
        "z_scale": z_scale,
        "base_height": base_height,
    }

def is_large_heightmap(height_map: np.ndarray, threshold: int = 1000000) -> bool:
    """.

    Check if a heightmap is considered large (exceeds threshold of total pixels).
    
    Args:
        height_map: The heightmap to check
        threshold: Number of pixels to consider large (default: 1 million)
        
    Returns:
        bool: True if the heightmap is large
    """
    return height_map.size > threshold

# Define available backends
available_backends = [b for b in ModelBackend if _check_backend_available(b)]
backend_choices = [b.value for b in available_backends]

# Add a helper function to prepare STL export parameters
def fix_stl_coordinates(height_map: np.ndarray, z_scale: float, base_height: float, **kwargs) -> dict:
    """
    Prepare parameters for STL export with corrected coordinate system.
    
    This helper function creates an improved parameter set for STL export with proper
    coordinate handling. It ensures the x and y values in the STL file correspond correctly
    to the heightmap coordinates.
    
    Args:
        height_map: The heightmap data
        z_scale: Scaling factor for z values
        base_height: Height of the base 
        **kwargs: Additional parameters to pass to the exporter
        
    Returns:
        dict: Parameters for STL export with coordinate correction
    """
    params = {
        "height_map": height_map,
        "z_scale": z_scale,
        "base_height": base_height,
        **kwargs
    }
    
    # Set x and y scales to maintain proper aspect ratio
    height, width = height_map.shape
    aspect_ratio = width / height
    
    # Set scale factors with aspect ratio consideration
    params["x_scale"] = aspect_ratio  
    params["y_scale"] = 1.0
    
    # Always include coordinate system parameters
    params["coordinate_system"] = kwargs.get("coordinate_system", "right-handed")
    params["origin_at_zero"] = kwargs.get("origin_at_zero", True)
    
    # Preserve height map orientation by default
    params["preserve_orientation"] = kwargs.get("preserve_orientation", True)
    
    # Handle base inversion
    params["invert_base"] = kwargs.get("invert_base", False)
    
    return params

@app.command()
def convert(
    input_file: str = typer.Argument(..., help="Input file path (TMD, image, EXR, etc.)"),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output file path (default: based on input file)"
    ),
    format: Format = typer.Option(
        Format.stl, "--format", "-f", help="Output format"
    ),
    z_scale: float = typer.Option(
        1.0, "--z-scale", "-z", help="Z-axis scaling factor"
    ),
    base_height: float = typer.Option(
        0.0, "--base-height", "-b", help="Height of solid base below model"
    ),
    texture: bool = typer.Option(
        False, "--texture", "-t", help="Add texture to supported formats"
    ),
    adaptive: bool = typer.Option(
        True, "--adaptive/--standard", "-a/-s", help="Use adaptive triangulation for better memory efficiency"
    ),
    max_error: float = typer.Option(
        0.01, "--max-error", "-e", help="Maximum error for adaptive triangulation"
    ),
    max_triangles: Optional[int] = typer.Option(
        None, "--max-triangles", "-n", help="Maximum triangle count for adaptive triangulation"
    ),
    binary: bool = typer.Option(
        True, "--binary/--ascii", help="Use binary format (STL/PLY)"
    ),
    crop: Optional[Tuple[int, int, int, int]] = typer.Option(
        None, "--crop", help="Crop region as min_row,max_row,min_col,max_col"
    ),
    rotate: int = typer.Option(
        180, "--rotate", "-r", help="Rotate heightmap (0, 90, 180, or 270 degrees)"
    ),
    mirror_x: bool = typer.Option(
        True, "--mirror-x/--no-mirror-x", help="Mirror heightmap along the X-axis"
    ),
    downscale: Optional[int] = typer.Option(
        None, "--downscale", help="Downscale factor (e.g., 2 reduces dimensions by half)"
    ),
    normal_map_z_scale: float = typer.Option(
        10.0, "--normal-z-scale", help="Z-scale for normal map generation"
    ),
    max_subdivisions: int = typer.Option(
        8, "--max-subdivisions", "-m", help="Maximum quad tree subdivisions for adaptive algorithm"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
    heightmap_format: str = typer.Option(
        "png", "--heightmap-format", help="Format for heightmap export (png, jpg, tiff)"
    ),
    invert: bool = typer.Option(
        False, "--invert", help="Invert heightmap values (black becomes white)"
    ),
    normalize: bool = typer.Option(
        True, "--normalize/--no-normalize", help="Normalize heightmap values to 0-1 range"
    ),
    resolution: Optional[int] = typer.Option(
        None, "--resolution", help="Resolution for heightmap export (max dimension)"
    ),
    import_type: ImportType = typer.Option(
        ImportType.auto, "--import-type", "-i", help="Type of file to import"
    ),
    backend: Optional[str] = typer.Option(
        None, "--backend", "-be", help=f"Backend to use for mesh generation ({', '.join(backend_choices)})"
    ),
    coordinate_system: str = typer.Option(
        "left-handed", "--coordinate-system", "-cs", 
        help="Coordinate system for STL (right-handed or left-handed)"
    ),
    origin_at_zero: bool = typer.Option(
        True, "--origin-at-zero/--origin-at-corner", 
        help="Place origin at center (True) or at corner (False)"
    ),
    preserve_orientation: bool = typer.Option(
        True, "--preserve-orientation/--raw-coords", 
        help="Preserve heightmap orientation in 3D model (default: True)"
    ),
    invert_base: bool = typer.Option(
        False, "--invert-base/--normal-base", 
        help="Invert the base to create a mold/negative (default: False)"
    ),
):
    """.

    Convert height map files to 3D models with various options.
    
    Examples:
        tmd2model convert input.tmd -f stl
        tmd2model convert input.exr -f stl -z 10.0 -b 0.5
        tmd2model convert input.png -f obj -z 5.0
        tmd2model convert input.tmd -f gltf -t
        tmd2model convert input.tmd -f normal_map
        tmd2model convert input.tmd -f stl --backend openstl
    """
    with console.status("[bold green]Processing input file..."):
        height_map = load_heightmap_from_file(input_file, import_type)
        if height_map is None:
            sys.exit(1)
        original_shape = height_map.shape

        # Apply cropping if specified
        if crop:
            try:
                height_map = crop_height_map(height_map, crop)
                rprint(f"[bold green]Cropped[/bold green] height map to region {crop}")
            except ValueError as e:
                rprint(f"[bold red]Error:[/bold red] Invalid crop region: {e}")
                sys.exit(1)
                
        # Apply X-mirroring if specified
        if mirror_x:
            try:
                height_map = np.flip(height_map, axis=1)
                rprint(f"[bold green]Mirrored[/bold green] height map along X-axis")
            except Exception as e:
                rprint(f"[bold red]Error:[/bold red] Failed to mirror: {e}")
                sys.exit(1)
                
        # Apply rotation if specified
        if rotate:
            try:
                # Ensure rotation is one of the allowed values
                if rotate not in [0, 90, 180, 270]:
                    rprint(f"[bold yellow]Warning:[/bold yellow] Invalid rotation angle {rotate}. Using 0 degrees.")
                    rotate = 0
                
                if rotate > 0:
                    # Calculate number of 90-degree rotations (1, 2, or 3)
                    k = rotate // 90
                    # Apply rotation using numpy.rot90
                    height_map = np.rot90(height_map, k=k)
                    rprint(f"[bold green]Rotated[/bold green] height map by {rotate} degrees")
                    rprint(f"New dimensions: {height_map.shape}")
            except Exception as e:
                rprint(f"[bold red]Error:[/bold red] Failed to rotate: {e}")
                sys.exit(1)

        # Apply downscaling if specified
        if downscale and downscale > 1:
            try:
                from scipy.ndimage import zoom
                factor = 1.0 / downscale
                height_map = zoom(height_map, factor, order=1)
                rprint(f"[bold green]Downscaled[/bold green] height map by factor of {downscale}")
                rprint(f"New dimensions: {height_map.shape}")
            except ImportError:
                rprint("[bold yellow]Warning:[/bold yellow] scipy required for downscaling. Proceeding without downscaling.")
            except Exception as e:
                rprint(f"[bold red]Error:[/bold red] Failed to downscale: {e}")
                sys.exit(1)

    # Check if heightmap is large 
    large_heightmap = height_map.size > 1000000  # Over 1 million pixels
    if large_heightmap and not adaptive and format == Format.stl:
        rprint("[bold yellow]Warning:[/bold yellow] Large heightmap detected. Using adaptive mesh generation.")
        adaptive = True

    # Determine output filename if not specified
    if output:
        output_file = output
    else:
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        ext = ".png" if format in [Format.normal_map, Format.heightmap] and not heightmap_format else f".{format.value}"
        if format == Format.heightmap and heightmap_format:
            ext = f".{heightmap_format}"
        output_file = f"{base_name}{ext}"

    # Show input/output info
    info_table = Table.grid(padding=(0, 1))
    info_table.add_row("Input file:", input_file)
    info_table.add_row("Original dimensions:", f"{original_shape[0]}x{original_shape[1]}")
    info_table.add_row("Processing dimensions:", f"{height_map.shape[0]}x{height_map.shape[1]}")
    info_table.add_row("Output format:", format.value)
    info_table.add_row("Output file:", output_file)
    if mirror_x:
        info_table.add_row("X-axis mirroring:", "Applied")
    if rotate:
        info_table.add_row("Rotation applied:", f"{rotate} degrees")
    if adaptive and format == Format.stl:
        info_table.add_row("Using adaptive algorithm:", "Yes")
    if format == Format.heightmap:
        info_table.add_row("Heightmap format:", heightmap_format)
        info_table.add_row("Invert values:", "Yes" if invert else "No")
    info_table.add_row("Preserve orientation:", "Yes" if preserve_orientation else "No")
    if invert_base and base_height > 0:
        info_table.add_row("Base style:", "Inverted (mold)")
    elif base_height > 0:
        info_table.add_row("Base style:", "Standard")
    console.print(Panel(info_table, title="[bold blue]TMD2Model Conversion[/bold blue]", expand=False))

    common_params = generate_common_params(height_map, output_file, z_scale, base_height)

    # Validate backend if specified
    if backend and backend not in backend_choices:
        rprint(f"[bold yellow]Warning:[/bold yellow] Unknown backend '{backend}'. Available backends: {', '.join(backend_choices)}")
        rprint(f"[bold yellow]Warning:[/bold yellow] Falling back to default backend")
        backend = None

    # Handle normal map conversion separately
    if format == Format.normal_map:
        with console.status("[bold green]Generating normal map..."):
            result = convert_heightmap_to_normal_map(
                common_params["height_map"],
                output_file,
                z_scale=normal_map_z_scale,
                normalize=True
            )
        if result:
            rprint(f"[bold green]Success:[/bold green] Normal map saved to {output_file}")
            file_size_kb = os.path.getsize(result) / 1024
            rprint(f"File size: {file_size_kb:.1f} KB")
            sys.exit(0)
        else:
            rprint(f"[bold red]Error:[/bold red] Failed to generate normal map")
            sys.exit(1)
    
    # Handle heightmap export separately
    if format == Format.heightmap:
        with console.status("[bold green]Exporting heightmap image..."):
            try:
                # Apply normalization if needed
                if normalize:
                    height_min, height_max = np.min(height_map), np.max(height_map)
                    if height_max > height_min:
                        height_map = (height_map - height_min) / (height_max - height_min)
                
                # Apply inversion if needed
                if invert:
                    height_map = 1.0 - height_map
                
                # Save heightmap
                output_file = save_heightmap(height_map, output_file, normalize=False)
                
                rprint(f"[bold green]Success:[/bold green] Heightmap image saved to {output_file}")
                file_size_kb = os.path.getsize(output_file) / 1024
                rprint(f"File size: {file_size_kb:.1f} KB")
                rprint(f"Dimensions: {height_map.shape[1]}x{height_map.shape[0]} pixels")
                sys.exit(0)
            except Exception as e:
                rprint(f"[bold red]Error:[/bold red] Failed to export heightmap image: {e}")
                sys.exit(1)

    # Conversion with progress display
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold green]{task.description}"),
        BarColumn(),
        TextColumn("[bold]{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task(f"Converting to {format.value.upper()}...", total=100)
        progress.update(task, advance=10)
        result = None

        # Define progress updater function
        def update_progress(percent):
            progress.update(task, completed=int(10 + percent * 0.9))

        if format == Format.stl:
            # STL format with unified interface
            progress.update(task, description="[bold green]Generating 3D mesh...")
            
            # Prepare STL parameters with correct coordinate handling
            stl_params = fix_stl_coordinates(
                height_map=height_map,
                z_scale=z_scale,
                base_height=base_height,
                filename=output_file,
                ascii=not binary,
                adaptive=adaptive,
                max_subdivisions=max_subdivisions,
                error_threshold=max_error,
                max_triangles=max_triangles,
                progress_callback=update_progress,
                backend=backend,
                coordinate_system=coordinate_system,
                origin_at_zero=origin_at_zero,
                preserve_orientation=preserve_orientation,
                invert_base=invert_base
            )
            
            # Call the STL exporter with the parameters
            result = convert_heightmap_to_stl(**stl_params)
        elif format == Format.obj:
            progress.update(task, advance=40)
            result = convert_heightmap_to_obj(**common_params)
        elif format == Format.ply:
            progress.update(task, advance=40)
            result = convert_heightmap_to_ply(
                **common_params,
                binary=binary,
            )
        elif format == Format.gltf:
            progress.update(task, advance=40)
            result = convert_heightmap_to_gltf(
                **common_params,
                add_texture=texture,
            )
        elif format == Format.glb:
            progress.update(task, advance=40)
            result = convert_heightmap_to_glb(
                **common_params,
                add_texture=texture,
            )
        elif format == Format.threejs:
            progress.update(task, advance=40)
            result = convert_heightmap_to_threejs(
                **common_params,
                add_texture=texture,
            )
        elif format == Format.usdz:
            progress.update(task, advance=40)
            result = convert_heightmap_to_usdz(
                **common_params,
                add_texture=texture,
            )
        progress.update(task, completed=100)

    if result:
        file_size = os.path.getsize(result)
        size_str = (
            f"{file_size / 1024:.1f} KB"
            if file_size < 1024 * 1024
            else f"{file_size / (1024 * 1024):.2f} MB"
        )
        
        # Try to get triangle count for the generated model
        triangle_count = None
        if format == Format.stl and os.path.exists(result):
            try:
                # For binary STL, read triangle count from header
                if binary:
                    with open(result, 'rb') as f:
                        f.seek(80)  # Skip header
                        triangle_count = int.from_bytes(f.read(4), byteorder='little')
                # For ASCII STL, estimate triangle count from file size
                else:
                    file_size = os.path.getsize(result)
                    # Typical ASCII STL structure is about 240 bytes per triangle
                    triangle_count = int(file_size / 240)
            except Exception:
                pass
                
        # Update summary table
        summary_table = Table(title="Conversion Result")
        summary_table.add_column("Property", style="cyan")
        summary_table.add_column("Value", style="green")
        summary_table.add_row("Output Format", format.value.upper())
        summary_table.add_row("Output File", output_file)
        summary_table.add_row("File Size", size_str)
        if triangle_count:
            summary_table.add_row("Triangle Count", f"{triangle_count:,}")
        if backend:
            summary_table.add_row("Backend Used", backend)
        if texture:
            summary_table.add_row("Texture", "Included")
        if base_height > 0:
            summary_table.add_row("Base Height", f"{base_height} units")
        if adaptive and format == Format.stl:
            summary_table.add_row("Triangulation", "Adaptive")
            summary_table.add_row("Error Tolerance", str(max_error))
        console.print(summary_table)
        rprint("[bold green]Conversion successful![/bold green]")
        sys.exit(0)
    else:
        rprint(f"[bold red]Error:[/bold red] Conversion to {format.value.upper()} failed")
        sys.exit(1)

@app.command()
def info(
    input_file: str = typer.Argument(..., help="Input TMD file path"),
    visualize: bool = typer.Option(False, "--visualize", "-v", help="Generate visualization")
):
    """.

    Display information about a TMD file without converting it.
    
    Examples:
        tmd2model info input.tmd
        tmd2model info input.tmd -v
    """
    with console.status("[bold green]Analyzing TMD file..."):
        data = process_tmd_file(input_file)
        if not data:
            sys.exit(1)
    height_map = data["height_map"]
    file_table = Table(title=f"TMD File Information: {os.path.basename(input_file)}")
    file_table.add_column("Property", style="cyan")
    file_table.add_column("Value", style="green")
    file_table.add_row("Dimensions", f"{height_map.shape[0]}x{height_map.shape[1]}")
    
    # Check if heightmap is large and provide a note
    if is_large_heightmap(height_map):
        file_table.add_row("Size Category", "[yellow]Large[/yellow] (consider using --adaptive for exports)")
    else:
        file_table.add_row("Size Category", "Standard")
    metadata_fields = [
        ("title", "Title"),
        ("description", "Description"),
        ("author", "Author"),
        ("date_created", "Date Created"),
        ("x_offset", "X Offset"),
        ("y_offset", "Y Offset"),
        ("x_length", "X Length"),
        ("y_length", "Y Length"),
        ("min_height", "Min Height"),
        ("max_height", "Max Height"),
        ("elevation_unit", "Elevation Unit"),
        ("crs", "Coordinate System"),
        ("version", "TMD Version"),
    ]
    for field, label in metadata_fields:
        if field in data:
            file_table.add_row(label, str(data[field]))
    file_table.add_row("Height Min", f"{np.min(height_map):.6f}")
    file_table.add_row("Height Max", f"{np.max(height_map):.6f}")
    file_table.add_row("Height Mean", f"{np.mean(height_map):.6f}")
    file_table.add_row("Height Std Dev", f"{np.std(height_map):.6f}")
    console.print(file_table)

    if visualize:
        try:
            from tmd.plotters.matplotlib import plot_height_map_3d
            output_dir = "tmd_info_output"
            os.makedirs(output_dir, exist_ok=True)

            # 2D visualization
            plt.figure(figsize=(8, 6))
            plt.imshow(height_map, cmap="terrain")
            plt.colorbar(label="Height")
            plt.title(f"Height Map: {os.path.basename(input_file)}")
            plt.savefig(os.path.join(output_dir, "height_map_2d.png"), dpi=150)
            plt.close()
            rprint(f"[bold green]2D visualization saved to {output_dir}/height_map_2d.png[/bold green]")

            # Only attempt 3D visualization for reasonably sized maps
            if not is_large_heightmap(height_map, threshold=250000):  # For 3D, use lower threshold
                try:
                    fig = plt.figure(figsize=(10, 8))
                    plot_height_map_3d(height_map, fig=fig, z_scale=1.0)
                    plt.savefig(os.path.join(output_dir, "height_map_3d.png"), dpi=150)
                    plt.close()
                    rprint(f"[bold green]3D visualization saved to {output_dir}/height_map_3d.png[/bold green]")
                except Exception as e:
                    if "projection='3d'" in str(e):
                        rprint("[bold yellow]Warning:[/bold yellow] 3D visualization not available")
                    else:
                        rprint(f"[bold yellow]Warning:[/bold yellow] 3D visualization failed: {e}")
            else:
                rprint("[bold yellow]Info:[/bold yellow] Skipping 3D visualization for large heightmap")
        except ImportError:
            rprint("[bold yellow]Warning:[/bold yellow] Matplotlib required for visualization")
        except Exception as e:
            rprint(f"[bold yellow]Warning:[/bold yellow] Visualization failed: {e}")
    sys.exit(0)

@app.command()
def batch(
    input_dir: str = typer.Argument(..., help="Directory containing TMD files"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", "-o", help="Output directory"),
    format: Format = typer.Option(Format.stl, "--format", "-f", help="Output format"),
    z_scale: float = typer.Option(1.0, "--z-scale", "-z", help="Z-axis scaling factor"),
    base_height: float = typer.Option(0.0, "--base-height", "-b", help="Base height"),
    texture: bool = typer.Option(False, "--texture", "-t", help="Add texture to supported formats"),
    binary: bool = typer.Option(True, "--binary/--ascii", help="Use binary format (STL/PLY)"),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Recursively search for TMD files"),
    pattern: str = typer.Option("*.tmd", "--pattern", "-p", help="File pattern to match"),
    adaptive: bool = typer.Option(False, "--adaptive", "-a", help="Use adaptive triangulation for better memory efficiency"),
    max_error: float = typer.Option(0.01, "--max-error", "-e", help="Maximum error for adaptive triangulation"),
    rotate: int = typer.Option(180, "--rotate", "--rot", help="Rotate heightmap (0, 90, 180, or 270 degrees)"),
    mirror_x: bool = typer.Option(True, "--mirror-x/--no-mirror-x", help="Mirror heightmap along the X-axis"),
):
    """.
    Batch convert multiple TMD files in a directory.
    
    Examples:
        tmd2model batch .
        tmd2model batch data/ -f glb -t
        tmd2model batch data/ -r
    """
    from pathlib import Path
    input_path = Path(input_dir)
    output_path = Path(output_dir) if output_dir else input_path / "tmd_batch_output"
    output_path.mkdir(parents=True, exist_ok=True)

    if recursive:
        matches = list(input_path.rglob(pattern))
    else:
        matches = list(input_path.glob(pattern))

    if not matches:
        rprint(f"[bold yellow]Warning:[/bold yellow] No TMD files found matching {pattern}")
        sys.exit(1)

    rprint(f"[bold blue]Found {len(matches)} TMD files to process[/bold blue]")
    successful = 0
    failed = 0

    with Progress() as progress:
        task = progress.add_task("[bold green]Processing files...", total=len(matches))
        for tmd_file in matches:
            progress.update(task, description=f"[bold green]Processing {tmd_file.name}...")
            base_name = tmd_file.stem
            ext = ".png" if format == Format.normal_map else f".{format.value}"
            output_file = output_path / f"{base_name}{ext}"

            with console.status(f"[bold green]Processing {tmd_file.name}..."):
                data = process_tmd_file(str(tmd_file))
                if not data:
                    rprint(f"[bold red]Error:[/bold red] Could not process {tmd_file}")
                    failed += 1
                    progress.update(task, advance=1)
                    continue
                height_map = data["height_map"]

                # Apply X-mirroring if specified
                if mirror_x:
                    try:
                        height_map = np.flip(height_map, axis=1)
                        rprint(f"[bold green]Mirrored[/bold green] height map along X-axis")
                    except Exception as e:
                        rprint(f"[bold red]Error:[/bold red] Failed to mirror: {e}")
                        failed += 1
                        progress.update(task, advance=1)
                        continue
                
                # Apply rotation if specified
                if rotate:
                    if rotate in [0, 90, 180, 270]:
                        if rotate > 0:
                            # Calculate number of 90-degree rotations
                            k = rotate // 90
                            # Apply rotation
                            height_map = np.rot90(height_map, k=k)
                    else:
                        rprint(f"[bold yellow]Warning:[/bold yellow] Invalid rotation angle {rotate}. Using 0 degrees.")

            common_params = {
                "height_map": height_map,
                "filename": str(output_file),
                "z_scale": z_scale,
                "base_height": base_height,
            }
            
            # Check if heightmap is large and we're exporting STL
            large_map = is_large_heightmap(height_map)
            if large_map and format == Format.stl and not adaptive:
                rprint(f"[bold yellow]Warning:[/bold yellow] Large heightmap in {tmd_file.name}. Using adaptive algorithm.")
                use_adaptive = True
            else:
                use_adaptive = adaptive

            if format == Format.normal_map:
                result = convert_heightmap_to_normal_map(
                    height_map,
                    str(output_file),
                    z_scale=10.0,
                    normalize=True,
                )
            else:
                if format == Format.stl:
                    if use_adaptive:
                        # Try to use adaptive_mesh for large maps
                        try:
                            result = convert_heightmap_to_adaptive_mesh(
                                height_map=height_map, 
                                output_file=str(output_file),
                                z_scale=z_scale,
                                base_height=base_height,
                                error_threshold=max_error,
                            )
                            if isinstance(result, tuple) and len(result) >= 3:
                                result = result[2]
                        except Exception as e:
                            rprint(f"[bold yellow]Warning:[/bold yellow] Adaptive export failed: {e}")
                            result = convert_heightmap_to_stl(
                                **common_params,
                                ascii=not binary,
                            )
                    else:
                        result = convert_heightmap_to_stl(
                            **common_params,
                            ascii=not binary,
                        )
                elif format == Format.obj:
                    result = convert_heightmap_to_obj(**common_params)
                elif format == Format.ply:
                    result = convert_heightmap_to_ply(
                        **common_params,
                        binary=binary,
                    )
                elif format == Format.gltf:
                    result = convert_heightmap_to_gltf(
                        **common_params,
                        add_texture=texture,
                    )
                elif format == Format.glb:
                    result = convert_heightmap_to_glb(
                        **common_params,
                        add_texture=texture,
                    )
                elif format == Format.threejs:
                    result = convert_heightmap_to_threejs(
                        **common_params,
                        add_texture=texture,
                    )
                elif format == Format.usdz:
                    result = convert_heightmap_to_usdz(
                        **common_params,
                        add_texture=texture,
                    )
            if result:
                successful += 1
            else:
                failed += 1
            progress.update(task, advance=1)

    summary_table = Table(title="Batch Conversion Summary")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Count", style="green")
    summary_table.add_row("Total Files", str(len(matches)))
    summary_table.add_row("Successful", str(successful))
    summary_table.add_row("Failed", f"[red]{failed}[/red]" if failed > 0 else "0")
    console.print(summary_table)

    if successful > 0:
        rprint(f"[bold green]Output files saved to: {output_path}[/bold green]")
    sys.exit(0 if failed == 0 else 1)

@app.command()
def heightmap(
    input_file: str = typer.Argument(..., help="Input TMD file path"),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output image file path (default: based on input file)"
    ),
    format: str = typer.Option(
        "png", "--format", "-f", help="Output image format (png, jpg, tiff)"
    ),
    invert: bool = typer.Option(
        False, "--invert", "-i", help="Invert heightmap values (black becomes white)"
    ),
    normalize: bool = typer.Option(
        True, "--normalize/--no-normalize", help="Normalize heightmap values to 0-1 range"
    ),
    resolution: Optional[int] = typer.Option(
        None, "--resolution", "-r", help="Resolution for output image (max dimension)"
    ),
    color_map: Optional[str] = typer.Option(
        None, "--color-map", "-c", help="Apply a colormap (terrain, jet, viridis, etc.)"
    ),
    add_colorbar: bool = typer.Option(
        False, "--colorbar", help="Add a colorbar to the color map image"
    ),
    crop: Optional[Tuple[int, int, int, int]] = typer.Option(
        None, "--crop", help="Crop region as min_row,max_row,min_col,max_col"
    ),
    rotate: int = typer.Option(
        180, "--rotate", "--rot", help="Rotate heightmap (0, 90, 180, or 270 degrees)"
    ),
    mirror_x: bool = typer.Option(
        True, "--mirror-x/--no-mirror-x", help="Mirror heightmap along the X-axis"
    ),
):
    """.
    Extract a heightmap image from a TMD file.
    
    Examples:
        tmd2model heightmap input.tmd -f png
        tmd2model heightmap input.tmd -o output.tiff -f tiff --invert
        tmd2model heightmap input.tmd --color-map terrain --colorbar
    """
    with console.status("[bold green]Processing TMD file..."):
        data = process_tmd_file(input_file)
        if not data:
            sys.exit(1)
        height_map = data["height_map"]
        original_shape = height_map.shape

        # Apply cropping if specified
        if crop:
            try:
                height_map = crop_height_map(height_map, crop)
                rprint(f"[bold green]Cropped[/bold green] height map to region {crop}")
            except ValueError as e:
                rprint(f"[bold red]Error:[/bold red] Invalid crop region: {e}")
                sys.exit(1)
                
        # Apply X-mirroring if specified
        if mirror_x:
            try:
                height_map = np.flip(height_map, axis=1)
                rprint(f"[bold green]Mirrored[/bold green] height map along X-axis")
            except Exception as e:
                rprint(f"[bold red]Error:[/bold red] Failed to mirror: {e}")
                sys.exit(1)
                
        # Apply rotation if specified
        if rotate:
            try:
                # Ensure rotation is one of the allowed values
                if rotate not in [0, 90, 180, 270]:
                    rprint(f"[bold yellow]Warning:[/bold yellow] Invalid rotation angle {rotate}. Using 0 degrees.")
                    rotate = 0
                
                if rotate > 0:
                    # Calculate number of 90-degree rotations (1, 2, or 3)
                    k = rotate // 90
                    # Apply rotation using numpy.rot90
                    height_map = np.rot90(height_map, k=k)
                    rprint(f"[bold green]Rotated[/bold green] height map by {rotate} degrees")
                    rprint(f"New dimensions: {height_map.shape}")
            except Exception as e:
                rprint(f"[bold red]Error:[/bold red] Failed to rotate: {e}")
                sys.exit(1)

    # Determine output filename if not specified
    if output:
        output_file = output
    else:
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        ext = f".{format.lower()}"
        output_file = f"{base_name}_heightmap{ext}"
    
    try:
        from PIL import Image
        import numpy as np
        import matplotlib.pyplot as plt

        # Normalize if requested
        if normalize:
            height_min, height_max = np.min(height_map), np.max(height_map)
            if height_max > height_min:
                height_map = (height_map - height_min) / (height_max - height_min)
        
        # Apply colormap if specified
        if color_map:
            plt.figure(figsize=(10, 8))
            img_plot = plt.imshow(height_map, cmap=color_map)
            if add_colorbar:
                plt.colorbar(img_plot, label="Height")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(output_file, bbox_inches='tight', pad_inches=0, dpi=300)
            plt.close()
            
            rprint(f"[bold green]Success:[/bold green] Colored heightmap saved to {output_file}")
            file_size_kb = os.path.getsize(output_file) / 1024
            rprint(f"File size: {file_size_kb:.1f} KB")
            sys.exit(0)
        
        # For grayscale heightmap
        # Invert if requested
        if invert:
            height_map = 1.0 - height_map
        
        # Convert to 8-bit grayscale image
        height_data = (height_map * 255).astype(np.uint8)
        img = Image.fromarray(height_data)
        
        # Resize if resolution is specified
        if resolution:
            current_width, current_height = height_data.shape[1], height_data.shape[0]
            aspect_ratio = current_width / current_height
            if current_width > current_height:
                new_width = resolution
                new_height = int(resolution / aspect_ratio)
            else:
                new_height = resolution
                new_width = int(resolution * aspect_ratio)
            img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Save the image
        img.save(output_file)
        rprint(f"[bold green]Success:[/bold green] Heightmap image saved to {output_file}")
        file_size_kb = os.path.getsize(output_file) / 1024
        rprint(f"File size: {file_size_kb:.1f} KB")
        rprint(f"Dimensions: {img.width}x{img.height} pixels")
        sys.exit(0)
    except ImportError:
        rprint("[bold yellow]Warning:[/bold yellow] PIL and Matplotlib required for heightmap export")
    except Exception as e:
        rprint(f"[bold red]Error:[/bold red] Failed to export heightmap image: {e}")
        sys.exit(1)

if __name__ == "__main__":
    app()
