#!/usr/bin/env python3
"""Model generation core functionality for TMD CLI."""

from pathlib import Path
from typing import Optional, Dict, Any, Callable
from enum import Enum
import logging
import psutil

# Import CLI utilities
from tmd.cli.core import (
    console,
    print_warning,
    print_error,
    print_success,
    load_config,
    save_config,
    load_tmd_file
)

# Set up logging
logger = logging.getLogger(__name__)

# Add enum definitions
class MeshMethod(str, Enum):
    """Mesh generation methods."""
    ADAPTIVE = "adaptive"
    QUADTREE = "quadtree"

class QualityPreset(str, Enum):
    """Quality presets for mesh generation."""
    DRAFT = "draft"
    NORMAL = "normal"
    HIGH = "high"
    ULTRA = "ultra"

def get_quality_params(preset: QualityPreset) -> dict:
    """Get parameters for a quality preset."""
    presets = {
        QualityPreset.DRAFT: {
            'error_threshold': 0.1,
            'max_triangles': 10000,
            'min_quad_size': 8,
            'max_quad_size': 64,
            'max_subdivisions': 6
        },
        QualityPreset.NORMAL: {
            'error_threshold': 0.05,
            'max_triangles': 50000,
            'min_quad_size': 4,
            'max_quad_size': 32,
            'max_subdivisions': 8
        },
        QualityPreset.HIGH: {
            'error_threshold': 0.01,
            'max_triangles': 200000,
            'min_quad_size': 2,
            'max_quad_size': 16,
            'max_subdivisions': 10
        },
        QualityPreset.ULTRA: {
            'error_threshold': 0.005,
            'max_triangles': 500000,
            'min_quad_size': 1,
            'max_quad_size': 8,
            'max_subdivisions': 12
        }
    }
    return presets[preset]

def export_model(
    input_file: Path,
    output_file: Path,
    format: str,
    **kwargs
) -> bool:
    """Export model in specified format."""
    try:
        from time import time
        from rich.panel import Panel
        from rich.table import Table
        from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
        
        start_time = time()
        
        # Load TMD file
        from tmd.core import TMD
        tmd_data = TMD.load(str(input_file))
        
        # TMD File Info Table
        info_table = Table(title="TMD File Information", show_header=True, expand=True)
        info_table.add_column("Property", style="cyan")
        info_table.add_column("Value", style="green")
        
        info_table.add_row("Size", f"{tmd_data.height_map.shape[0]} × {tmd_data.height_map.shape[1]}")
        info_table.add_row("Height Range", f"{tmd_data.height_map.min():.2f} to {tmd_data.height_map.max():.2f}")
        info_table.add_row("Memory Usage", f"{tmd_data.height_map.nbytes / 1024:.1f} KB")
        
        # Add metadata to the table
        for key, value in tmd_data.metadata.items():
            info_table.add_row(key, str(value))
        
        console.print(info_table)
        console.print()

        # Show detailed export parameters
        param_table = Table(title="Export Parameters", show_header=True, expand=True)
        param_table.add_column("Parameter", style="cyan")
        param_table.add_column("Value", style="green")
        param_table.add_column("Description", style="yellow")

        parameter_info = {
            'format': (format.upper(), "Output format"),
            'method': (kwargs.get('method', 'adaptive'), "Mesh generation method"),
            'scale': (kwargs.get('scale', 1.0), "Height scale factor"),
            'error_threshold': (kwargs.get('error_threshold', 0.05), "Max error for mesh simplification"),
            'max_triangles': (kwargs.get('max_triangles', 50000), "Maximum triangle count"),
            'binary': (kwargs.get('binary', True), "Use binary format if supported"),
            'min_quad_size': (kwargs.get('min_quad_size', 4), "Minimum quad size for subdivision"),
            'max_quad_size': (kwargs.get('max_quad_size', 64), "Maximum quad size for subdivision"),
            'max_subdivisions': (kwargs.get('max_subdivisions', 4), "Maximum subdivision depth"),
            'detail_boost': (kwargs.get('detail_boost', 1.0), "Detail enhancement factor"),
            'coordinate_system': (kwargs.get('coordinate_system', 'right-handed'), "Coordinate system orientation"),
            'uv_method': (kwargs.get('uv_method', 'planar'), "UV mapping method"),
            'optimize': (kwargs.get('optimize', True), "Optimize mesh after generation"),
            'calculate_normals': (kwargs.get('calculate_normals', True), "Generate vertex normals"),
            'texture': (kwargs.get('texture', False), "Generate texture from heightmap")
        }

        for param, (value, desc) in parameter_info.items():
            param_table.add_row(param, str(value), desc)
        
        console.print(param_table)
        console.print()
        
        # Map CLI parameters to ExportConfig parameters
        config_params = {
            'triangulation_method': str(kwargs.get('method', 'adaptive')).replace('MeshMethod.', '').lower(),
            'error_threshold': kwargs.get('error_threshold', 0.05),
            'min_quad_size': kwargs.get('min_quad_size', 4),
            'max_quad_size': kwargs.get('max_quad_size', 64),
            'max_triangles': 50000,  # Default value
            'simplify_ratio': kwargs.get('simplify_ratio', 0.25),
            'z_scale': kwargs.get('scale', 1.0),
            'max_subdivisions': kwargs.get('max_subdivisions', 4),
            'binary': kwargs.get('binary', True),
            'x_length': tmd_data.metadata.get('x_length', 1.0),
            'y_length': tmd_data.metadata.get('y_length', 1.0),
            'x_offset': tmd_data.metadata.get('x_offset', 0.0),
            'y_offset': tmd_data.metadata.get('y_offset', 0.0)
        }

        # Override max_triangles if specified in kwargs
        if 'max_triangles' in kwargs and kwargs['max_triangles'] is not None:
            config_params['max_triangles'] = kwargs['max_triangles']

        # Progress display
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(complete_style="green"),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=False
        ) as progress:
            # Create tasks for each stage
            main_task = progress.add_task("[cyan]Generating mesh...", total=100)
            
            def progress_callback(percent):
                progress.update(main_task, completed=percent)
            
            # Create config and export
            from tmd.model.config import ExportConfig
            from tmd.model.factory import ModelExporterFactory
            
            config = ExportConfig(**config_params)
            config.progress_callback = progress_callback
            
            factory = ModelExporterFactory()
            result = factory.export(
                input_file=str(input_file),  # Ensure string path
                output_file=str(output_file), # Ensure string path
                format_name=format.lower(),   # Ensure lowercase format
                config=config
            )

        # Show completion message with timing
        elapsed = time() - start_time
        if result:
            success_panel = Panel.fit(
                f"Successfully exported [cyan]{input_file.name}[/] to [green]{output_file}[/]\n"
                f"Format: [yellow]{format.upper()}[/]\n"
                f"Time: [cyan]{elapsed:.1f}[/] seconds",
                title="Export Complete",
                border_style="green"
            )
            console.print(success_panel)
            return True

        print_error("Export failed")
        return False
        
    except Exception as e:
        print_error(f"Failed to export model: {e}")
        return False

def batch_export_models(
    input_dir: Path,
    output_dir: Optional[Path] = None,
    format: str = "stl",
    pattern: str = "*.tmd",
    quality: QualityPreset = QualityPreset.NORMAL,
    scale: float = 5.0,
    max_workers: int = 1,
    recursive: bool = False,
    **kwargs
) -> bool:
    """Batch process multiple TMD files."""
    try:
        import glob
        from concurrent.futures import ProcessPoolExecutor, as_completed
        
        # Find all TMD files
        if recursive:
            pattern_path = input_dir / "**" / pattern
            tmd_files = list(Path().glob(str(pattern_path)))
        else:
            pattern_path = input_dir / pattern
            tmd_files = list(Path().glob(str(pattern_path)))
        
        if not tmd_files:
            print_error(f"No TMD files found matching pattern '{pattern}' in {input_dir}")
            return False
        
        # Set up output directory
        if output_dir is None:
            output_dir = input_dir / "models"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        console.print(f"[cyan]Found {len(tmd_files)} TMD files to process[/cyan]")
        console.print(f"[yellow]Output directory: {output_dir}[/yellow]")
        
        # Get quality parameters
        quality_params = get_quality_params(quality)
        quality_params.update(kwargs)  # Include any additional kwargs
        
        # Process files
        from rich.progress import Progress
        
        with Progress(console=console) as progress:
            main_task = progress.add_task("[cyan]Processing files...", total=len(tmd_files))
            
            if max_workers == 1:
                # Sequential processing
                for tmd_file in tmd_files:
                    output_file = output_dir / f"{tmd_file.stem}.{format.lower()}"
                    success = export_model(
                        input_file=tmd_file,
                        output_file=output_file,
                        format=format,
                        scale=scale,
                        **quality_params
                    )
                    progress.advance(main_task)
                    if not success:
                        print_warning(f"Failed to process: {tmd_file}")
            else:
                # Parallel processing
                def process_file(tmd_file):
                    output_file = output_dir / f"{tmd_file.stem}.{format.lower()}"
                    return export_model(
                        input_file=tmd_file,
                        output_file=output_file,
                        format=format,
                        scale=scale,
                        **quality_params
                    )
                
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(process_file, f): f for f in tmd_files}
                    for future in as_completed(futures):
                        tmd_file = futures[future]
                        try:
                            success = future.result()
                            if not success:
                                print_warning(f"Failed to process: {tmd_file}")
                        except Exception as e:
                            print_error(f"Error processing {tmd_file}: {e}")
                        progress.advance(main_task)
        
        print_success(f"Batch processing completed. Check {output_dir} for results.")
        return True
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}", exc_info=True)
        print_error(f"Batch processing failed: {e}")
        return False

def list_model_formats():
    """List all available model export formats."""
    try:
        from tmd.model.formats import get_available_formats
        formats = get_available_formats()
        
        from rich.table import Table
        table = Table(title="Available 3D Model Formats", show_header=True, expand=True)
        table.add_column("Format", style="cyan")
        table.add_column("Extension", style="green")
        table.add_column("Description", style="yellow")
        table.add_column("Features", style="blue")
        
        format_info = {
            'stl': ('.stl', 'Stereolithography format', 'Binary/ASCII, widely supported'),
            'obj': ('.obj', 'Wavefront OBJ format', 'Text-based, materials support'),
            'ply': ('.ply', 'Polygon File Format', 'Binary/ASCII, efficient'),
            'gltf': ('.gltf/.glb', 'GL Transmission Format', 'Modern, web-friendly'),
            'usd': ('.usd/.usda', 'Universal Scene Description', 'Advanced features, Pixar standard')
        }
        
        for format_name in formats:
            if format_name in format_info:
                ext, desc, features = format_info[format_name]
                table.add_row(format_name.upper(), ext, desc, features)
            else:
                table.add_row(format_name.upper(), f".{format_name}", "", "")
        
        console.print(table)
        
    except ImportError:
        # Fallback if get_available_formats is not available
        console.print("[cyan]Available 3D model formats:[/cyan]")
        formats = ["stl", "obj", "ply", "gltf", "usd"]
        for format_name in formats:
            console.print(f"  - {format_name}")

def generate_model_command(
    tmd_file: Path,
    output_file: Optional[Path] = None,
    z_scale: float = 1.0,
    base_height: float = 0.0,
    max_triangles: Optional[int] = None,
    error_threshold: float = 0.01,
    coordinate_system: str = "right-handed",
    origin_at_zero: bool = True,
    invert_base: bool = False,
    progress_callback: Optional[Callable[[float], None]] = None
) -> bool:
    """Generate a 3D model from a TMD file (for backwards compatibility)."""
    try:
        # Validate input file
        if not tmd_file.exists():
            print_error(f"Input file does not exist: {tmd_file}")
            return False

        # Check available memory
        try:
            file_size = tmd_file.stat().st_size
            available_memory = psutil.virtual_memory().available
            required_memory = file_size * 4  # Rough estimate
            
            if required_memory > available_memory:
                print_error(f"Not enough memory available. Need {required_memory/(1024**3):.1f}GB but only {available_memory/(1024**3):.1f}GB available")
                return False
        except Exception as e:
            logger.warning(f"Could not check memory requirements: {e}")

        # Load TMD file with progress reporting
        with console.status(f"Loading {tmd_file.name}..."):
            try:
                data = load_tmd_file(tmd_file)
                if not data or not hasattr(data, 'height_map') or data.height_map is None:
                    raise RuntimeError("Invalid or missing height map data")
                
                height_map = data.height_map
                shape = height_map.shape
                
                # Validate dimensions
                if len(shape) != 2:
                    raise ValueError(f"Expected 2D height map, got {len(shape)}D")
                if any(dim > 10000 for dim in shape):
                    raise ValueError(f"Height map too large: {shape}")
                    
                logger.info(f"Loaded heightmap with shape {shape}")
                
            except Exception as e:
                raise RuntimeError(f"Failed to load TMD file: {e}")

        # Create output filename if not specified
        if output_file is None:
            # Load config for default format
            config = load_config()
            default_format = config.get("model", {}).get("default_format", "stl")
            output_format = default_format.lower()
            
            # Create output filename
            try:
                from tmd.cli.core.io import create_output_dir
                output_dir = create_output_dir(subdir="models")
                output_file = output_dir / f"{tmd_file.stem}.{output_format}"
            except ImportError:
                # Fallback if create_output_dir not available
                output_file = tmd_file.with_suffix(f".{output_format}")

        # Import and run model generation
        try:
            from tmd.model.adaptive_mesh import convert_heightmap_to_adaptive_mesh
            
            with console.status(f"Generating 3D model..."):
                result = convert_heightmap_to_adaptive_mesh(
                    height_map=height_map,
                    output_file=str(output_file),
                    z_scale=z_scale,
                    base_height=base_height,
                    error_threshold=error_threshold,
                    max_triangles=max_triangles,
                    progress_callback=progress_callback,
                    coordinate_system=coordinate_system,
                    origin_at_zero=origin_at_zero,
                    invert_base=invert_base
                )

            if result is None:
                raise RuntimeError("Model generation failed")

            vertices, faces = result
            print_success(f"Generated mesh with {len(vertices)} vertices and {len(faces)} triangles")
            print_success(f"Model saved to {output_file}")
            return True
            
        except ImportError as e:
            # Fallback to using export_model if adaptive_mesh not available
            logger.warning(f"Could not import adaptive_mesh, using fallback: {e}")
            return export_model(
                input_file=tmd_file,
                output_file=output_file,
                format=output_file.suffix[1:] if output_file.suffix else "stl",
                scale=z_scale,
                base_height=base_height,
                max_triangles=max_triangles,
                error_threshold=error_threshold,
                coordinate_system=coordinate_system
            )

    except ImportError as e:
        print_error(f"Missing dependencies: {e}")
        print_warning("Make sure SciPy, NumPy and OpenCV are installed")
        return False
        
    except Exception as e:
        logger.error(f"Model generation failed: {e}", exc_info=True)
        print_error(f"Error generating 3D model: {e}")
        return False