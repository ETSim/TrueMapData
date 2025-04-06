"""Mesh export functionality for the CLI."""
import typer
from pathlib import Path
from typing import Optional
from enum import Enum
from time import time
import logging
import psutil

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.panel import Panel

from ..core.ui import print_error, print_success, print_warning
from ...model.factory import ModelExporterFactory
from ...model.formats import get_available_formats
from ...core import TMD

# Set up logging
logger = logging.getLogger(__name__)
console = Console()

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
            'error_threshold': 0.1,            # Higher error tolerance
            'max_triangles': 25000,            # Limited triangle count
            'min_quad_size': 8,               # Larger minimum quad size
            'max_quad_size': 64,              # Larger maximum quad size
            'max_subdivisions': 6,            # Limited subdivisions
            'detail_boost': 0.25,             # Minimal detail enhancement
            'base_height': 0.01,              # Thin base
            'optimize': True,                 # Basic optimization
            'calculate_normals': True,
            'simplify_ratio': 0.5,            # Aggressive simplification
            'adaptive': {
                'min_area_fraction': 0.005,    # Larger minimum triangle area
                'smoothing': 2.0,             # More smoothing
                'feature_angle': 30           # Less sensitive feature detection
            },
            'quadtree': {
                'merge_threshold': 0.2,       # Aggressive node merging
                'split_threshold': 0.8,       # Less splitting
                'min_vertex_spacing': 4.0     # Larger vertex spacing
            }
        },
        QualityPreset.NORMAL: {
            'error_threshold': 0.02,           # Balanced error tolerance
            'max_triangles': 100000,           # Moderate triangle count
            'min_quad_size': 4,               # Moderate quad size
            'max_quad_size': 32,              # Moderate maximum quad size
            'max_subdivisions': 8,            # Standard subdivisions
            'detail_boost': 0.75,             # Standard detail enhancement
            'base_height': 0.02,              # Standard base
            'optimize': True,                 # Standard optimization
            'calculate_normals': True,
            'simplify_ratio': 0.25,           # Moderate simplification
            'adaptive': {
                'min_area_fraction': 0.002,    # Standard minimum triangle area
                'smoothing': 1.0,             # Standard smoothing
                'feature_angle': 20           # Standard feature detection
            },
            'quadtree': {
                'merge_threshold': 0.15,      # Standard node merging
                'split_threshold': 0.6,       # Standard splitting
                'min_vertex_spacing': 2.0     # Standard vertex spacing
            }
        },
        QualityPreset.HIGH: {
            'error_threshold': 0.005,          # Low error tolerance
            'max_triangles': 500000,           # High triangle count
            'min_quad_size': 2,               # Small quad size
            'max_quad_size': 16,              # Smaller maximum quad size
            'max_subdivisions': 10,           # More subdivisions
            'detail_boost': 1.5,              # Enhanced detail enhancement
            'base_height': 0.03,              # Thicker base
            'optimize': True,                 # Full optimization
            'calculate_normals': True,
            'simplify_ratio': 0.1,            # Light simplification
            'adaptive': {
                'min_area_fraction': 0.0005,   # Smaller minimum triangle area
                'smoothing': 0.5,             # Less smoothing
                'feature_angle': 15           # More sensitive feature detection
            },
            'quadtree': {
                'merge_threshold': 0.1,       # Conservative node merging
                'split_threshold': 0.4,       # More splitting
                'min_vertex_spacing': 1.0     # Smaller vertex spacing
            }
        },
        QualityPreset.ULTRA: {
            'error_threshold': 0.0005,        # Even more precise error threshold
            'max_triangles': 2500000,         # Increased triangle count
            'min_quad_size': 1,              # Smallest quad size
            'max_quad_size': 8,              # Small max quad size
            'max_subdivisions': 12,          # Maximum subdivisions
            'detail_boost': 3.0,             # Increased detail boost
            'base_height': 0.05,             # Thicker base
            'optimize': False,               # No optimization for maximum quality
            'calculate_normals': True,
            'simplify_ratio': 0.01,          # Minimal simplification
            'adaptive': {
                'min_area_fraction': 0.00005,
                'smoothing': 0.1,            # Minimal smoothing
                'feature_angle': 5           # Most sensitive feature detection
            },
            'quadtree': {
                'merge_threshold': 0.02,     # Very conservative merging
                'split_threshold': 0.2,      # Aggressive splitting
                'min_vertex_spacing': 0.25   # Finest vertex spacing
            }
        }
    }
    return presets[preset]

def print_mesh_info(tmd_data, console):
    """Print mesh-specific TMD information."""
    from rich.table import Table
    
    table = Table(title="Mesh Generation Info")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    
    if hasattr(tmd_data, 'height_map'):
        table.add_row("Height Map Shape", str(tmd_data.height_map.shape))
        table.add_row("Data Type", str(tmd_data.height_map.dtype))
    
    if hasattr(tmd_data, 'metadata') and tmd_data.metadata:
        for key, value in tmd_data.metadata.items():
            if isinstance(value, (str, int, float)):
                table.add_row(str(key), str(value))
    
    console.print(table)

def export_model(
    input_file: Path,
    output_file: Path,
    format: str,
    **kwargs
) -> bool:
    """Export model in specified format."""
    try:
        start_time = time()
        
        # Validate input file
        if not input_file.exists():
            print_error(f"Input file not found: {input_file}")
            return False
            
        # Load and validate TMD file
        try:
            # Check available memory
            available_memory = psutil.virtual_memory().available
            
            # Load TMD file
            tmd_data = TMD.load(str(input_file))
            
            # Validate height map
            if tmd_data.height_map is None:
                print_error("Invalid height map in file")
                return False
                
            # Check dimensions
            shape = tmd_data.height_map.shape
            if len(shape) != 2:
                print_error(f"Invalid height map dimensions: expected 2D array, got {len(shape)}D")
                return False
                
            if any(dim > 10000 for dim in shape):
                print_warning("Height map dimensions very large: {shape}")
                print_warning("Consider downsampling the height map first")
                
            # Check memory requirements (4 bytes per float32)
            array_size = tmd_data.height_map.size * 4
            if array_size > available_memory:
                print_error(f"Not enough memory for processing ({array_size/(1024**3):.1f}GB)")
                return False
                
            # Display info
            print_mesh_info(tmd_data, console)
            
            # Set up progress tracking
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=console,
                transient=False
            ) as progress:
                main_task = progress.add_task("[cyan]Generating mesh...", total=100)
                
                def progress_callback(percent):
                    progress.update(main_task, completed=int(percent * 100))
                
                # Configure and export
                from ...model.base import ExportConfig
                config = ExportConfig(**kwargs)
                config.progress_callback = progress_callback
                
                result = ModelExporterFactory().export(
                    input_file=str(input_file),
                    output_file=str(output_file),
                    format_name=format,
                    config=config
                )

            # Show completion message if successful
            if result:
                elapsed = time() - start_time
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
            logger.exception("Export failed")
            return False
            
    except Exception as e:
        print_error(f"Export error: {e}")
        return False

def create_export_mesh_app() -> typer.Typer:
    """Create the mesh export app."""
    app = typer.Typer(
        help="Export TMD files to various 3D model formats",
        short_help="Export to 3D model formats"
    )

    # Define common parameters
    common_params = {
        'input_file': typer.Argument(..., help="Input TMD file", exists=True),
        'output_file': typer.Option(None, help="Output filename"),
        'scale': typer.Option(1.0, help="Mesh scale factor"),
        'method': typer.Option(MeshMethod.ADAPTIVE, help="Mesh generation method"),
        'quality': typer.Option(QualityPreset.NORMAL, help="Quality preset"),
        'max_triangles': typer.Option(None, help="Override maximum number of triangles"),
        'error_threshold': typer.Option(None, help="Override error threshold")
    }

    @app.command("list")
    def list_formats():
        """List all available mesh export formats."""
        for format in get_available_formats():
            typer.echo(f"  - {format}")

    def create_export_command(format_name: str, binary_option: bool = False, suffix: str = None):
        """Creates an export command for a specific format."""
        def command_func(
            input_file: Path = common_params['input_file'],
            output_file: Optional[Path] = common_params['output_file'],
            scale: float = common_params['scale'],
            method: MeshMethod = common_params['method'],
            quality: QualityPreset = common_params['quality'],
            max_triangles: Optional[int] = common_params['max_triangles'],
            error_threshold: Optional[float] = common_params['error_threshold'],
            min_quad_size: Optional[int] = typer.Option(None, help="Minimum quad size"),
            max_quad_size: Optional[int] = typer.Option(None, help="Maximum quad size"),
            max_subdivisions: Optional[int] = typer.Option(None, help="Maximum subdivisions"),
            binary: bool = typer.Option(True, help=f"Use binary {format_name.upper()} format") if binary_option else None
        ):
            """Export as {format_name.upper()} mesh."""
            try:
                # Determine output suffix
                if suffix:
                    actual_suffix = suffix if not binary_option else (suffix[0] + suffix[1:])
                else:
                    actual_suffix = f".{format_name}"
                
                # Set output file if not specified
                output_file = output_file or input_file.with_suffix(actual_suffix)
                
                # Get quality parameters
                quality_params = get_quality_params(quality)
                
                # Build export parameters
                params = {
                    'scale': scale,
                    'method': method,
                    'max_triangles': max_triangles or quality_params['max_triangles'],
                    'error_threshold': error_threshold or quality_params['error_threshold'],
                    'min_quad_size': min_quad_size or quality_params['min_quad_size'],
                    'max_quad_size': max_quad_size or quality_params['max_quad_size'],
                    'max_subdivisions': max_subdivisions or quality_params['max_subdivisions']
                }
                
                if binary_option:
                    params['binary'] = binary

                # Export model
                if not export_model(input_file, output_file, format_name, **params):
                    raise typer.Exit(code=1)
                    
            except Exception as e:
                print_error(f"Error exporting {format_name.upper()}: {e}")
                raise typer.Exit(code=1)

        return command_func

    # Register format-specific commands
    app.command("stl")(create_export_command("stl", binary_option=True))
    app.command("obj")(create_export_command("obj"))
    app.command("ply")(create_export_command("ply", binary_option=True))
    app.command("gltf")(create_export_command("gltf", binary_option=True, suffix=".gltf"))
    app.command("usd")(create_export_command("usd", binary_option=True, suffix=".usda"))

    return app
