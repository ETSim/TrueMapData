#!/usr/bin/env python3
"""
TMD to 3D Model Converter

Advanced command-line tool to convert TMD height map files to various 3D model formats
with additional features like cropping and normal map generation.
"""

# Standard library imports
import os
import sys
import time
import json
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Callable, Union, List

# Third-party imports
import numpy as np
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich import print as rprint

# Local imports
from tmd.processor import TMDProcessor
from tmd.utils.processing import crop_height_map
from tmd.exporters.image.image_io import load_image, ImageType
from tmd.utils.mesh_converter import (
    convert_heightmap,
    get_file_extension,
    print_conversion_stats,
    is_large_heightmap,
    get_heightmap_stats,
    prepare_conversion_info,
    display_conversion_stats
)

# Initialize Typer app and Rich console
app = typer.Typer(help="Convert TMD height map files to 3D model formats.")
console = Console()

# Define format options as an enum for better validation
class Format(str, Enum):
    stl = "stl"
    obj = "obj"
    ply = "ply"
    gltf = "gltf"
    glb = "glb"
    usd = "usd"
    usdz = "usdz"
    normal_map = "normal_map"
    bump_map = "bump_map"
    ao_map = "ao_map"  # Added ao_map format
    displacement_map = "displacement_map"  # Added displacement_map format
    heightmap = "heightmap"
    hillshade = "hillshade"
    material_set = "material_set"

class ImportType(str, Enum):
    tmd = "tmd"
    image = "image"

class CoordinateSystem(str, Enum):
    left_handed = "left-handed"
    right_handed = "right-handed"


def load_heightmap_from_file(input_file: str, import_type: ImportType = ImportType.tmd) -> Optional[np.ndarray]:
    """
    Load a heightmap from a file based on the specified import type.
    
    Args:
        input_file: Path to the input file
        import_type: Type of file to import (tmd or image)
        
    Returns:
        numpy.ndarray: 2D array of height values or None if loading failed
    """
    try:
        # Determine file type if not specified
        if import_type == ImportType.tmd:
            processor = TMDProcessor(input_file)
            data = processor.process()
            if data and 'height_map' in data:
                return data['height_map']
            return None
        else:  # ImportType.image
            return load_image(input_file, image_type=ImageType.HEIGHTMAP, normalize=True)
    except Exception as e:
        rprint(f"[bold red]Error:[/bold red] Failed to load heightmap: {e}")
        return None


def apply_transformations(
    height_map: np.ndarray,
    crop: Optional[Tuple[int, int, int, int]] = None,
    mirror_x: bool = False,
    rotate: int = 0,
    downscale: Optional[int] = None
) -> np.ndarray:
    """
    Apply various transformations to a heightmap.
    
    Args:
        height_map: The heightmap to transform
        crop: Crop region (min_row, max_row, min_col, max_col)
        mirror_x: Whether to mirror along X axis
        rotate: Rotation angle in degrees (0, 90, 180, 270)
        downscale: Downscale factor
        
    Returns:
        The transformed heightmap
    """
    result = height_map.copy()
    
    # Apply cropping if specified
    if crop:
        try:
            result = crop_height_map(result, crop)
            rprint(f"[bold green]Cropped[/bold green] height map to region {crop}")
        except ValueError as e:
            rprint(f"[bold red]Error:[/bold red] Invalid crop region: {e}")
            raise
                
    # Apply X-mirroring if specified
    if mirror_x:
        try:
            result = np.flip(result, axis=1)
            rprint(f"[bold green]Mirrored[/bold green] height map along X-axis")
        except Exception as e:
            rprint(f"[bold red]Error:[/bold red] Failed to mirror: {e}")
            raise
                
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
                result = np.rot90(result, k=k)
                rprint(f"[bold green]Rotated[/bold green] height map by {rotate} degrees")
        except Exception as e:
            rprint(f"[bold red]Error:[/bold red] Failed to rotate: {e}")
            raise

    # Apply downscaling if specified
    if downscale and downscale > 1:
        try:
            from scipy.ndimage import zoom
            factor = 1.0 / downscale
            result = zoom(result, factor, order=1)
            rprint(f"[bold green]Downscaled[/bold green] height map by factor of {downscale}")
        except ImportError:
            rprint("[bold yellow]Warning:[/bold yellow] scipy required for downscaling. Proceeding without downscaling.")
        except Exception as e:
            rprint(f"[bold red]Error:[/bold red] Failed to downscale: {e}")
            raise
            
    return result


@app.command()
def convert(
    input_file: str = typer.Argument(..., help="Input file path (TMD or image)"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path"),
    format: Format = typer.Option(Format.stl, "--format", "-f", help="Output format"),
    z_scale: float = typer.Option(1.0, "--z-scale", "-z", help="Z-axis scaling factor"),
    base_height: float = typer.Option(0.0, "--base-height", "-b", help="Height of solid base below model"),
    adaptive: bool = typer.Option(True, "--adaptive/--standard", "-a/-s", help="Use adaptive triangulation"),
    max_error: float = typer.Option(0.01, "--max-error", "-e", help="Maximum error for adaptive triangulation"),
    max_triangles: Optional[int] = typer.Option(None, "--max-triangles", "-n", help="Maximum triangle count"),
    binary: bool = typer.Option(True, "--binary/--ascii", help="Use binary format (STL/PLY)"),
    crop: Optional[Tuple[int, int, int, int]] = typer.Option(None, "--crop", help="Crop region (min_row,max_row,min_col,max_col)"),
    rotate: int = typer.Option(0, "--rotate", "-r", help="Rotate heightmap (0, 90, 180, 270 degrees)"),
    mirror_x: bool = typer.Option(False, "--mirror-x/--no-mirror-x", help="Mirror heightmap along X-axis"),
    downscale: Optional[int] = typer.Option(None, "--downscale", help="Downscale factor"),
    normal_map_z_scale: float = typer.Option(1.0, "--normal-z-scale", help="Z-scale for normal map generation"),
    bump_map_strength: float = typer.Option(1.0, "--bump-strength", help="Strength for bump map generation"),
    bump_map_blur: float = typer.Option(1.0, "--bump-blur", help="Blur radius for bump map generation"),
    max_subdivisions: int = typer.Option(8, "--max-subdivisions", "-m", help="Maximum quad tree subdivisions"),
    coordinate_system: CoordinateSystem = typer.Option(
        CoordinateSystem.right_handed, "--coordinate-system", "-cs", help="Coordinate system"
    ),
    origin_at_zero: bool = typer.Option(True, "--origin-at-zero/--origin-at-corner", help="Place origin at center"),
    invert_base: bool = typer.Option(False, "--invert-base/--normal-base", help="Invert the base (mold/negative)"),
    bit_depth: int = typer.Option(16, "--bit-depth", help="Bit depth for image export (8 or 16)"),
    add_texture: bool = typer.Option(True, "--add-texture/--no-texture", help="Add texture to 3D models"),
    preserve_aspect: bool = typer.Option(True, "--preserve-aspect/--no-preserve-aspect", help="Preserve heightmap aspect ratio"),
    hillshade_azimuth: float = typer.Option(315.0, "--hillshade-azimuth", help="Azimuth angle for hillshade (0-360)"),
    hillshade_altitude: float = typer.Option(45.0, "--hillshade-altitude", help="Altitude angle for hillshade (0-90)"),
    hillshade_z_factor: float = typer.Option(1.0, "--hillshade-z", help="Z factor for hillshade exaggeration"),
    material_base_name: str = typer.Option("material", "--material-name", help="Base name for material set files")
):
    """
    Convert height map files to 3D models or texture maps.
    
    Examples:
        # Convert TMD to STL
        tmd2model convert input.tmd -f stl
        
        # Convert TMD to STL with higher Z scale and a base
        tmd2model convert input.tmd -f stl -z 10.0 -b 0.5
        
        # Convert TMD to OBJ format
        tmd2model convert input.tmd -f obj -z 5.0
        
        # Convert to normal map
        tmd2model convert input.tmd -f normal_map
        
        # Convert to bump map with custom strength
        tmd2model convert input.tmd -f bump_map --bump-strength 2.0
        
        # Convert to hillshade visualization
        tmd2model convert input.tmd -f hillshade --hillshade-azimuth 315 --hillshade-altitude 45
        
        # Generate a complete material set
        tmd2model convert input.tmd -f material_set --material-name terrain_material
        
        # Convert image to STL
        tmd2model convert heightmap.png -f stl --adaptive
        
        # Convert to glTF with texture
        tmd2model convert input.tmd -f gltf --add-texture
        
        # Export a heightmap image
        tmd2model convert input.tmd -f heightmap --bit-depth 16
    """
    # Determine file type from extension
    ext = os.path.splitext(input_file)[1].lower()
    
    with console.status("[bold green]Processing input file..."):
        # Load heightmap based on file extension
        if ext == '.tmd':
            # Load TMD file
            processor = TMDProcessor(input_file)
            data = processor.process()
            if data and 'height_map' in data:
                height_map = data['height_map']
            else:
                rprint(f"[bold red]Error:[/bold red] Failed to load TMD file {input_file}")
                sys.exit(1)
        else:
            # Try to load as image
            height_map = load_image(input_file, normalize=True)
            if height_map is None:
                rprint(f"[bold red]Error:[/bold red] Failed to load image file {input_file}")
                sys.exit(1)
                
        original_shape = height_map.shape

        # Apply transformations
        try:
            height_map = apply_transformations(
                height_map,
                crop=crop,
                mirror_x=mirror_x,
                rotate=rotate,
                downscale=downscale
            )
        except Exception:
            sys.exit(1)

    # Check if heightmap is large 
    large_heightmap = is_large_heightmap(height_map)
    if large_heightmap and not adaptive and format == Format.stl:
        rprint("[bold yellow]Warning:[/bold yellow] Large heightmap detected. Using adaptive mesh generation.")
        adaptive = True

    # Determine output filename if not specified
    if output:
        output_file = output
    else:
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = f"{base_name}{get_file_extension(format.value)}"

    # Special handling for material_set format
    if format == Format.material_set:
        # Create output directory if it doesn't exist
        if output:
            output_dir = output
        else:
            # Default directory based on input file name
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            output_dir = f"{base_name}_materials"
        
        # Make sure it's treated as a directory
        os.makedirs(output_dir, exist_ok=True)
        output_file = output_dir  # Use directory as output_file for material_set format

    # Get detailed conversion info for reporting
    info_dict = prepare_conversion_info(
        input_file=input_file,
        height_map=height_map,
        original_shape=original_shape,
        format_type=format.value,
        output_file=output_file,
        z_scale=z_scale,
        base_height=base_height,
        mirror_x=mirror_x,
        rotate=rotate,
        adaptive=adaptive,
        max_error=max_error,
        coordinate_system=coordinate_system,
        binary=binary,
        origin_at_zero=origin_at_zero
    )

    # Show information panel
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
    if format in [Format.bump_map, Format.normal_map]:
        info_table.add_row("Z-scale for map:", str(normal_map_z_scale if format == Format.normal_map else bump_map_strength))
    if invert_base and base_height > 0:
        info_table.add_row("Base style:", "Inverted (mold)")
    elif base_height > 0:
        info_table.add_row("Base style:", "Standard")
        
    console.print(Panel(info_table, title="[bold blue]TMD2Model Conversion[/bold blue]", expand=False))

    # Prepare model dimensions to preserve aspect ratio if requested
    model_params = {}
    if preserve_aspect:
        # Calculate aspect ratio from heightmap
        aspect_ratio = height_map.shape[1] / height_map.shape[0] if height_map.shape[0] > 0 else 1.0
        model_params["x_length"] = aspect_ratio
        model_params["y_length"] = 1.0

    # Perform conversion with progress display
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold green]{task.description}"),
        BarColumn(),
        TextColumn("[bold]{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task(f"Converting to {format.value.upper()}...", total=100)
        progress.update(task, advance=10)
        
        # Define progress updater function
        def update_progress(percent):
            progress.update(task, completed=int(10 + percent * 0.9))

        # Update task description based on format
        progress.update(
            task, 
            description=f"[bold green]{'Generating' if format in [Format.normal_map, Format.bump_map, Format.heightmap] else 'Creating'} {format.value}..."
        )

        # Convert with unified function
        start_time = time.time()
        result = convert_heightmap(
            height_map,
            output_file,
            format.value,
            # Common parameters
            z_scale=z_scale,
            base_height=base_height,
            binary=binary,
            # STL specific parameters
            adaptive=adaptive and format == Format.stl,
            max_error=max_error,
            max_subdivisions=max_subdivisions,
            max_triangles=max_triangles,
            coordinate_system=str(coordinate_system),
            origin_at_zero=origin_at_zero,
            invert_base=invert_base,
            # Image specific parameters
            normal_map_z_scale=normal_map_z_scale,
            bump_map_strength=bump_map_strength,
            bump_map_blur=bump_map_blur,
            bit_depth=bit_depth,
            # GLTF/USD specific
            add_texture=add_texture,
            # Progress callback
            progress_callback=update_progress,
            # Pass model dimensions
            hillshade_azimuth=hillshade_azimuth,
            hillshade_altitude=hillshade_altitude,
            hillshade_z_factor=hillshade_z_factor,
            material_base_name=material_base_name,
            **model_params
        )
        elapsed_time = time.time() - start_time
        progress.update(task, completed=100)

    # Show results
    if result:
        stats = print_conversion_stats(result, format.value)
        # Add elapsed time to stats
        stats["elapsed_time"] = f"{elapsed_time:.2f}s"
        display_conversion_stats(stats)
        rprint("[bold green]Conversion successful![/bold green]")
        sys.exit(0)
    else:
        rprint(f"[bold red]Error:[/bold red] Conversion to {format.value.upper()} failed")
        sys.exit(1)


@app.command()
def batch(
    input_dir: str = typer.Argument(..., help="Directory containing TMD files"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", "-o", help="Output directory"),
    format: Format = typer.Option(Format.stl, "--format", "-f", help="Output format"),
    z_scale: float = typer.Option(1.0, "--z-scale", "-z", help="Z-axis scaling factor"),
    base_height: float = typer.Option(0.0, "--base-height", "-b", help="Base height"),
    binary: bool = typer.Option(True, "--binary/--ascii", help="Use binary format (STL/PLY)"),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Recursively search for files"),
    pattern: str = typer.Option("*.tmd", "--pattern", "-p", help="File pattern to match"),
    adaptive: bool = typer.Option(True, "--adaptive", "-a", help="Use adaptive triangulation"),
    max_error: float = typer.Option(0.01, "--max-error", "-e", help="Maximum error for adaptive triangulation"),
    max_subdivisions: int = typer.Option(8, "--max-subdivisions", "-m", help="Maximum quad tree subdivisions"),
    max_triangles: Optional[int] = typer.Option(None, "--max-triangles", "-n", help="Maximum triangle count"),
    rotate: int = typer.Option(0, "--rotate", "--rot", help="Rotate heightmap (0, 90, 180, 270 degrees)"),
    mirror_x: bool = typer.Option(False, "--mirror-x/--no-mirror-x", help="Mirror heightmap along X-axis"),
    coordinate_system: CoordinateSystem = typer.Option(
        CoordinateSystem.right_handed, "--coordinate-system", "-cs", help="Coordinate system"
    ),
    origin_at_zero: bool = typer.Option(True, "--origin-at-zero/--origin-at-corner", help="Place origin at center"),
    bump_map_strength: float = typer.Option(1.0, "--bump-strength", help="Strength for bump map generation"),
    bump_map_blur: float = typer.Option(1.0, "--bump-blur", help="Blur radius for bump map"),
    normal_map_z_scale: float = typer.Option(1.0, "--normal-z-scale", help="Z-scale for normal maps"),
    ao_strength: float = typer.Option(1.0, "--ao-strength", help="Strength for ambient occlusion maps"),
    ao_samples: int = typer.Option(16, "--ao-samples", help="Sample count for ambient occlusion maps"),
    bit_depth: int = typer.Option(16, "--bit-depth", help="Bit depth for image export (8 or 16)"),
    import_type: ImportType = typer.Option(ImportType.tmd, "--import-type", "-i", help="Type of file to import"),
    add_texture: bool = typer.Option(True, "--add-texture/--no-texture", help="Add texture to 3D models"),
    all_maps: bool = typer.Option(False, "--all-maps", help="Generate all image formats (normal, bump, heightmap, ao)"),
    hillshade_azimuth: float = typer.Option(315.0, "--hillshade-azimuth", help="Azimuth angle for hillshade (0-360)"),
    hillshade_altitude: float = typer.Option(45.0, "--hillshade-altitude", help="Altitude angle for hillshade (0-90)"),
    hillshade_z_factor: float = typer.Option(1.0, "--hillshade-z", help="Z factor for hillshade exaggeration"),
    material_base_name: str = typer.Option("material", "--material-name", help="Base name for material set files"),
    precision: str = typer.Option("float32", "--precision", help="Precision for height data (float16, float32, float64)"),
    preserve_precision: bool = typer.Option(True, "--preserve-precision/--standard-precision", help="Preserve precision during processing"),
    optimization_level: int = typer.Option(1, "--optimization-level", "-ol", help="Level of optimization (0-3, higher is more aggressive)"),
    parallel_processing: bool = typer.Option(True, "--parallel/--sequential", help="Use parallel processing when possible"),
    chunk_size: Optional[int] = typer.Option(None, "--chunk-size", help="Size of chunks for processing large files"),
):
    """
    Batch convert multiple TMD files in a directory.
    
    Examples:
        # Convert all TMD files in current directory to STL
        tmd2model batch .
        
        # Convert all TMD files in data/ to OBJ with recursion
        tmd2model batch data/ -f obj -r
        
        # Convert all TMD files to bump maps
        tmd2model batch data/ -f bump_map --bump-strength 2.0
        
        # Process PNG files instead of TMD
        tmd2model batch images/ -p "*.png" -i image -f stl
        
        # Generate all image maps from TMD files
        tmd2model batch data/ --all-maps
        
        # Generate hillshades with custom lighting
        tmd2model batch data/ -f hillshade --hillshade-azimuth 315 --hillshade-altitude 45
        
        # Generate complete material sets for all files
        tmd2model batch data/ -f material_set --material-name terrain
        
        # High-precision conversion with maximum subdivision
        tmd2model batch data/ -f stl --max-subdivisions 12 --precision float64 --preserve-precision
        
        # High performance batch processing
        tmd2model batch large_data/ -f stl --parallel --chunk-size 1024 --optimization-level 3
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir) if output_dir else input_path / f"tmd_batch_{format.value}"
    output_path.mkdir(parents=True, exist_ok=True)

    if recursive:
        matches = list(input_path.rglob(pattern))
    else:
        matches = list(input_path.glob(pattern))

    if not matches:
        rprint(f"[bold yellow]Warning:[/bold yellow] No files found matching {pattern}")
        sys.exit(1)

    rprint(f"[bold blue]Found {len(matches)} files to process[/bold blue]")
    successful = 0
    failed = 0
    total_time = 0.0
    
    # If all_maps is specified, determine formats to generate
    formats_to_generate = []
    if all_maps:
        formats_to_generate = [
            Format.normal_map, 
            Format.bump_map, 
            Format.heightmap, 
            Format.ao_map, 
            Format.displacement_map,
            Format.hillshade
        ]
        rprint(f"[bold blue]Generating all image formats for each file[/bold blue]")
    else:
        formats_to_generate = [format]
    
    # Set up parallel processing if requested
    if parallel_processing and chunk_size:
        try:
            from concurrent.futures import ProcessPoolExecutor
            import multiprocessing
            
            num_workers = min(multiprocessing.cpu_count(), 8)  # Limit to reasonable number
            rprint(f"[bold blue]Using parallel processing with {num_workers} workers[/bold blue]")
            executor = ProcessPoolExecutor(max_workers=num_workers)
            futures = []
        except ImportError:
            rprint("[bold yellow]Warning:[/bold yellow] concurrent.futures not available. Using sequential processing.")
            parallel_processing = False

    with Progress() as progress:
        task = progress.add_task("[bold green]Processing files...", total=len(matches) * len(formats_to_generate))
        
        for file_path in matches:
            progress.update(task, description=f"[bold green]Processing {file_path.name}...")
            
            # Determine if this is a TMD or image file based on extension
            current_import_type = import_type
            if current_import_type == ImportType.tmd and not file_path.suffix.lower() == ".tmd":
                current_import_type = ImportType.image
            
            # Load the heightmap
            height_map = load_heightmap_from_file(str(file_path), current_import_type)
            if height_map is None:
                rprint(f"[bold red]Error:[/bold red] Could not process {file_path}")
                failed += 1
                progress.update(task, advance=len(formats_to_generate))
                continue

            # Apply transformations
            try:
                height_map = apply_transformations(
                    height_map,
                    mirror_x=mirror_x,
                    rotate=rotate
                )
                
                # Apply precision conversion if specified
                if precision == "float64":
                    height_map = height_map.astype(np.float64)
                elif precision == "float32":
                    height_map = height_map.astype(np.float32)
                elif precision == "float16":
                    height_map = height_map.astype(np.float16)
                
                # Prepare model dimensions to preserve aspect ratio
                model_params = {
                    "x_length": height_map.shape[1] / height_map.shape[0] if height_map.shape[0] > 0 else 1.0,
                    "y_length": 1.0
                }
                
                # Process each format
                for current_format in formats_to_generate:
                    # Special handling for material_set format
                    if current_format == Format.material_set:
                        # Create directory for each file's material set
                        base_name = file_path.stem
                        material_dir = output_path / f"{base_name}_materials"
                        os.makedirs(material_dir, exist_ok=True)
                        output_file = str(material_dir)
                    else:
                        # Generate output filename
                        base_name = file_path.stem
                        format_suffix = "_normal" if current_format == Format.normal_map else \
                                    "_bump" if current_format == Format.bump_map else \
                                    "_ao" if current_format == Format.ao_map else \
                                    "_disp" if current_format == Format.displacement_map else \
                                    "_height" if current_format == Format.heightmap else \
                                    "_hillshade" if current_format == Format.hillshade else ""
                        
                        # Only add format suffix when generating multiple formats
                        if all_maps:
                            output_file = output_path / f"{base_name}{format_suffix}{get_file_extension(current_format.value)}"
                        else:
                            output_file = output_path / f"{base_name}{get_file_extension(current_format.value)}"
                    
                    progress.update(task, description=f"[bold green]Processing {file_path.name} → {current_format.value}...")
                
                    # Measure conversion time
                    start_time = time.time()
                    
                    # Set up chunk processing for large heightmaps if specified
                    large_heightmap_processing = {}
                    if chunk_size and is_large_heightmap(height_map, threshold=chunk_size):
                        large_heightmap_processing = {
                            "chunk_size": chunk_size,
                            "optimize_memory": True
                        }
                    
                    # Create a configuration dictionary for conversion params
                    conversion_params = {
                        "z_scale": z_scale,
                        "base_height": base_height,
                        "binary": binary,
                        "adaptive": adaptive and current_format == Format.stl,
                        "max_error": max_error,
                        "max_subdivisions": max_subdivisions,
                        "max_triangles": max_triangles,
                        "coordinate_system": str(coordinate_system),
                        "origin_at_zero": origin_at_zero,
                        "normal_map_z_scale": normal_map_z_scale,
                        "bump_map_strength": bump_map_strength,
                        "bump_map_blur": bump_map_blur,
                        "ao_strength": ao_strength,
                        "ao_samples": ao_samples,
                        "bit_depth": bit_depth,
                        "add_texture": add_texture,
                        "hillshade_azimuth": hillshade_azimuth,
                        "hillshade_altitude": hillshade_altitude,
                        "hillshade_z_factor": hillshade_z_factor,
                        "material_base_name": material_base_name,
                        "preserve_precision": preserve_precision,
                        "optimization_level": optimization_level,
                        **large_heightmap_processing,
                        **model_params
                    }
                    
                    # Convert using the unified function
                    result = convert_heightmap(
                        height_map,
                        str(output_file),
                        current_format.value,
                        **conversion_params
                    )
                    
                    elapsed = time.time() - start_time
                    total_time += elapsed
                    
                    if result:
                        successful += 1
                        format_label = format_suffix.strip("_") if format_suffix else current_format.value
                        rprint(f"[green]✓[/green] {file_path.name} → {output_file.name} ({format_label}, {elapsed:.2f}s)")
                    else:
                        failed += 1
                        rprint(f"[red]✗[/red] {file_path.name} conversion to {current_format.value} failed")
                    
                    progress.update(task, advance=1)
                    
            except Exception as e:
                rprint(f"[bold red]Error:[/bold red] {e}")
                failed += 1
                progress.update(task, advance=len(formats_to_generate) - (formats_to_generate.index(current_format) if 'current_format' in locals() else 0))

    # Display summary
    summary_table = Table(title="Batch Conversion Summary")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Count", style="green")
    summary_table.add_row("Total Files", str(len(matches)))
    summary_table.add_row("Total Conversions", str(len(matches) * len(formats_to_generate)))
    summary_table.add_row("Successful", str(successful))
    summary_table.add_row("Failed", f"[red]{failed}[/red]" if failed > 0 else "0")
    summary_table.add_row("Total Time", f"{total_time:.2f}s")
    summary_table.add_row("Average Time", f"{total_time/max(1, successful):.2f}s per conversion")
    console.print(summary_table)

    if successful > 0:
        rprint(f"[bold green]Output files saved to: {output_path}[/bold green]")
    sys.exit(0 if failed == 0 else 1)


@app.command()
def info(
    input_file: str = typer.Argument(..., help="Input file path (TMD or image)"),
    import_type: ImportType = typer.Option(ImportType.tmd, "--import-type", "-i", help="Type of file to import"),
):
    """
    Show information about a heightmap file.
    
    Examples:
        # Show info about a TMD file
        tmd2model info input.tmd
        
        # Show info about an image file
        tmd2model info heightmap.png -i image
    """
    with console.status("[bold green]Analyzing file..."):
        if import_type == ImportType.tmd:
            processor = TMDProcessor(input_file)
            data = processor.process()  # Use process() instead of load()
            if not data:
                rprint(f"[bold red]Error:[/bold red] Could not load TMD file {input_file}")
                sys.exit(1)
                
            # Get height map
            height_map = data.get('height_map')
            if height_map is None:
                rprint(f"[bold red]Error:[/bold red] No height map found in {input_file}")
                sys.exit(1)
                
            # Extract metadata if available
            metadata = data.get('metadata', {})
        else:
            # Load as image
            height_map = load_image(input_file, image_type=ImageType.HEIGHTMAP)
            if height_map is None:
                rprint(f"[bold red]Error:[/bold red] Could not load image file {input_file}")
                sys.exit(1)
            metadata = {}

    # Use the utility function from mesh_converter
    stats = get_heightmap_stats(height_map)
    
    # Display file information
    file_info = Path(input_file)
    file_size = file_info.stat().st_size
    file_size_str = f"{file_size / 1024:.1f} KB" if file_size < 1024 * 1024 else f"{file_size / (1024 * 1024):.2f} MB"
    
    # Create info table
    info_table = Table(title=f"[bold]File Information: {file_info.name}[/bold]")
    info_table.add_column("Property", style="cyan")
    info_table.add_column("Value", style="green")
    
    # File properties
    info_table.add_row("File Type", import_type.value.upper())
    info_table.add_row("File Size", file_size_str)
    info_table.add_row("Last Modified", str(file_info.stat().st_mtime))
    
    # Height map properties
    info_table.add_row("Dimensions", f"{stats['width']} x {stats['height']} pixels")
    info_table.add_row("Total Pixels", f"{stats['total_pixels']:,}")
    info_table.add_row("Height Range", f"{stats['min_height']:.4f} to {stats['max_height']:.4f}")
    info_table.add_row("Mean Height", f"{stats['mean_height']:.4f}")
    info_table.add_row("Height Std Dev", f"{stats['std_dev']:.4f}")
    
    # Add metadata if available
    if metadata:
        metadata_table = Table(title="Metadata")
        metadata_table.add_column("Key", style="cyan")
        metadata_table.add_column("Value", style="green")
        
        for key, value in metadata.items():
            if key != "height_map":
                metadata_table.add_row(str(key), str(value))
    else:
        metadata_table = None
    
    # Recommended export formats
    rec_table = Table(title="Recommended Export Options")
    rec_table.add_column("Format", style="cyan")
    rec_table.add_column("Command", style="green")
    
    # Large heightmap recommendations
    if stats["is_large"]:
        rec_table.add_row(
            "STL (Adaptive)", 
            f"tmd2model convert {input_file} -f stl -i {import_type} --adaptive -z 10.0"
        )
    else:
        rec_table.add_row(
            "STL", 
            f"tmd2model convert {input_file} -f stl -i {import_type} -z 10.0"
        )
    
    rec_table.add_row(
        "OBJ", 
        f"tmd2model convert {input_file} -f obj -i {import_type} -z 10.0"
    )
    
    rec_table.add_row(
        "Normal Map", 
        f"tmd2model convert {input_file} -f normal_map -i {import_type}"
    )
    
    # Display information
    console.print(info_table)
    
    if metadata_table:
        console.print("\n")
        console.print(metadata_table)
    
    console.print("\n")
    console.print(rec_table)
    
    # Preview heightmap visualization if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        from io import BytesIO
        import base64
        from rich.markdown import Markdown
        
        plt.figure(figsize=(6, 4))
        plt.imshow(height_map, cmap='terrain')
        plt.colorbar(label='Height')
        plt.title(f"Heightmap Preview: {file_info.name}")
        
        # Save to BytesIO object
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=75)
        buf.seek(0)
        
        # Convert to base64
        img_base64 = base64.b64encode(buf.read()).decode('ascii')
        
        # Display as Markdown
        console.print("\n[bold]Heightmap Preview:[/bold]")
        
        # Only display if terminal supports it
        if console.color_system and console.is_terminal:
            console.print(Markdown(f"![Heightmap Preview](data:image/png;base64,{img_base64})"))
        else:
            console.print("[yellow]Preview not available in this terminal[/yellow]")
            
    except ImportError:
        console.print("\n[yellow]Install matplotlib to enable heightmap previews[/yellow]")


if __name__ == "__main__":
    app()
