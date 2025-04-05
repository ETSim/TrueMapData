#!/usr/bin/env python3
"""
Model generation commands for TMD CLI.

This module provides functions for creating 3D models from TMD files
using different triangulation methods and mesh generation algorithms.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Callable

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
    """
    Generate a 3D model from a TMD file.
    
    Args:
        tmd_file: Path to TMD file
        output_file: Path to output file (STL, OBJ, or PLY)
        z_scale: Scaling factor for height values
        base_height: Height of solid base below the model
        max_triangles: Maximum number of triangles
        error_threshold: Error threshold for adaptive subdivision
        coordinate_system: Coordinate system orientation
        origin_at_zero: Place origin at center of model
        invert_base: Invert base to create mold/negative
        progress_callback: Optional callback function for progress updates
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load TMD file
        data = load_tmd_file(tmd_file, with_console_status=True)
        if not data:
            print_error(f"Failed to load TMD file: {tmd_file}")
            return False
        
        # Get height map
        height_map = data.height_map()
        
        # Determine output filename if not specified
        if output_file is None:
            # Load config for default format
            config = load_config()
            default_format = config.get("model", {}).get("default_format", "stl")
            output_format = default_format.lower()
            
            # Create output filename
            from tmd.cli.core.io import create_output_dir
            output_dir = create_output_dir(subdir="models")
            output_file = output_dir / f"{tmd_file.stem}.{output_format}"
        
        # Import model generation functionality
        from tmd.model.adaptive_mesh import convert_heightmap_to_adaptive_mesh
        
        # Generate the mesh
        with console.status(f"Generating 3D model from {tmd_file.name}..."):
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
                print_error(f"Failed to generate 3D model")
                return False
            
            vertices, faces = result
            
            print_success(f"Generated 3D model with {len(vertices)} vertices and {len(faces)} triangles")
            print_success(f"Model saved to {output_file}")
            
            return True
            
    except ImportError:
        print_error("3D model generation functionality not available")
        print_warning("Make sure SciPy, NumPy, and OpenCV are installed")
        return False
        
    except Exception as e:
        print_error(f"Error generating 3D model: {e}")
        return False

"""Model export commands for TMD CLI."""
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.status import Status
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..core.ui import print_error, print_success
from ...model.factory import ModelExporterFactory
from ...model.formats import get_available_formats
from ...model.config import ExportConfig

console = Console()

def export_model(
    input_file: Path,
    output_file: Path,
    format: str,
    **kwargs
) -> bool:
    """Export model in specified format."""
    try:
        # Show export configuration
        config_table = Table(title="Export Configuration", show_header=True)
        config_table.add_column("Parameter", style="cyan")
        config_table.add_column("Value", style="green")
        
        # Show configuration parameters
        for key, value in kwargs.items():
            config_table.add_row(key, str(value))
        
        console.print(config_table)
        
        # Map CLI parameters directly to ExportConfig
        config_params = {
            'method': kwargs.get('method', 'adaptive'),
            'error_threshold': kwargs.get('error_threshold', 0.01),
            'min_quad_size': kwargs.get('min_quad_size', 2),
            'max_quad_size': kwargs.get('max_quad_size', 32),
            'curvature_threshold': kwargs.get('curvature_threshold', 0.1),
            'simplify_ratio': kwargs.get('simplify_ratio', None),
            'max_triangles': kwargs.get('max_triangles', None),
            'use_feature_edges': kwargs.get('use_feature_edges', True),
            'smoothing': kwargs.get('smoothing', 0.0),
            'scale': kwargs.get('scale', 1.0),
            'binary': kwargs.get('binary', True),
            'texture': kwargs.get('texture', False),
            'texture_resolution': kwargs.get('texture_resolution', None),
            'color_map': kwargs.get('color_map', 'terrain'),
            'optimize': kwargs.get('optimize', True),
            'calculate_normals': kwargs.get('normals', True),
            'coordinate_system': kwargs.get('coordinate_system', 'right-handed'),
            'origin_at_zero': kwargs.get('origin_at_zero', True),
            'base_height': kwargs.get('base_height', 0.0)
        }
        
        # Show progress during export
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task(f"Exporting {input_file.name} to {format}...", total=None)
            
            # Create export configuration
            config = ExportConfig(**config_params)
            
            # Create factory and export
            factory = ModelExporterFactory()
            result = factory.export(
                input_file=input_file,
                output_file=output_file,
                format_name=format,
                config=config
            )
            
            progress.update(task, completed=True)
        
        if result:
            # Show success message with details
            success_panel = Panel.fit(
                f"Successfully exported [cyan]{input_file.name}[/] to [green]{output_file}[/]\n"
                f"Format: [yellow]{format.upper()}[/]",
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
