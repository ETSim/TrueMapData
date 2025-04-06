#!/usr/bin/env python3
"""Model generation commands for TMD CLI."""

from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, Union
from enum import Enum
import typer
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
    """Generate a 3D model from a TMD file."""
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
            from tmd.cli.core.io import create_output_dir
            output_dir = create_output_dir(subdir="models")
            output_file = output_dir / f"{tmd_file.stem}.{output_format}"

        # Import and run model generation
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
        print_error(f"Missing dependencies: {e}")
        print_warning("Make sure SciPy, NumPy and OpenCV are installed")
        return False
        
    except Exception as e:
        logger.error(f"Model generation failed: {e}", exc_info=True)
        print_error(f"Error generating 3D model: {e}")
        return False

"""Model export commands for TMD CLI."""
from pathlib import Path
from typing import Optional
import typer
from enum import Enum

# Import dependencies
from rich.console import Console
from rich.panel import Panel
from rich.status import Status
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.tree import Tree

from ..core.ui import print_error, print_success, print_warning
from ...model.factory import ModelExporterFactory
from ...model.formats import get_available_formats
from ...model.config import ExportConfig
from ...core import TMD

from time import time
from tqdm.rich import tqdm

console = Console()

def export_model(
    input_file: Path,
    output_file: Path,
    format: str,
    **kwargs
) -> bool:
    """Export model in specified format."""
    try:
        start_time = time()
        
        # Load TMD file
        from ...core import TMD
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

def export_command(
    tmd_file: Path,
    format: str = typer.Option("stl", help="Output format (stl, obj, ply, gltf, usd)"),
    output_file: Optional[Path] = None,
    method: MeshMethod = typer.Option(MeshMethod.ADAPTIVE, help="Mesh generation method"),
    quality: QualityPreset = typer.Option(QualityPreset.HIGH, help="Quality preset"),  # Changed default to HIGH
    scale: float = typer.Option(5.0, help="Scale factor for height values"),  # Increased default z-scale
    save_heightmap: bool = typer.Option(True, help="Save heightmap visualization"),
    colormap: str = typer.Option("terrain", help="Colormap for heightmap visualization"),
    base_height: float = typer.Option(0.0, help="Height of solid base below model"),
    max_triangles: Optional[int] = typer.Option(None, help="Override maximum triangle count"),
    error_threshold: Optional[float] = typer.Option(None, help="Override error threshold"),
    max_subdivisions: int = typer.Option(12, help="Maximum quadtree subdivisions"),  # Increased subdivisions
    binary: Optional[bool] = typer.Option(None, help="Use binary format if supported"),
    coordinate_system: str = typer.Option("right-handed", help="Coordinate system orientation"),
    optimize: bool = typer.Option(True, help="Optimize mesh after generation")
) -> None:
    """Export a TMD file to a 3D model format."""
    try:
        # Get quality parameters
        quality_params = get_quality_params(quality)
        
        # Override with command line parameters if provided
        if error_threshold is not None:
            quality_params['error_threshold'] = error_threshold
        if max_triangles is not None:
            quality_params['max_triangles'] = max_triangles
    
        # Determine output filename
        if not output_file:
            output_file = tmd_file.with_suffix(f".{format.lower()}")
    
        # Export the model
        success = export_model(
            input_file=tmd_file,
            output_file=output_file,
            format=format,
            method=method,
            scale=scale,
            base_height=base_height,
            binary=binary,
            coordinate_system=coordinate_system,
            optimize=optimize,
            save_heightmap=save_heightmap,  # Add this parameter
            colormap=colormap,  # Add this parameter
            **quality_params
        )
    
        if not success:
            raise typer.Exit(code=1)

    except Exception as e:
        logger.error(f"Export failed: {e}", exc_info=True)
        print_error(f"Export failed: {e}")
        raise typer.Exit(code=1)

