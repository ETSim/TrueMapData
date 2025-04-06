"""Mesh export functionality for the CLI."""
import typer
from pathlib import Path
from typing import Optional, Tuple
from enum import Enum
from time import time
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

from ..commands.model import export_model
from ...model.formats import get_available_formats

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
        
        # Display TMD info table
        # ...existing code...

        # Map CLI parameters to ExportConfig parameters
        config_params = {
            'triangulation_method': str(kwargs.get('method', 'adaptive')).replace('MeshMethod.', '').lower(),
            'error_threshold': kwargs.get('error_threshold', 0.05),
            'min_quad_size': kwargs.get('min_quad_size', 4),
            'max_quad_size': kwargs.get('max_quad_size', 64),
            'max_triangles': 50000,  # Default value
            'z_scale': kwargs.get('scale', 1.0),
            'max_subdivisions': kwargs.get('max_subdivisions', 4),
            'binary': kwargs.get('binary', True),
            'x_length': tmd_data.metadata.get('x_length', 1.0),
            'y_length': tmd_data.metadata.get('y_length', 1.0),
            'x_offset': tmd_data.metadata.get('x_offset', 0.0),
            'y_offset': tmd_data.metadata.get('y_offset', 0.0),
            'coordinate_system': kwargs.get('coordinate_system', 'right-handed'),
            'optimize': True,  # Enable mesh optimization by default
            'calculate_normals': True,  # Always calculate normals
            'base_height': kwargs.get('base_height', 0.0)
        }

        # Create progress display
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
            main_task = progress.add_task("[cyan]Generating mesh...", total=100)
            
            def progress_callback(percent):
                progress.update(main_task, completed=int(percent * 100))
            
            # Create config and export
            from ...model.base import ExportConfig
            config = ExportConfig(**config_params)
            config.progress_callback = progress_callback
            
            from ...model.factory import ModelExporterFactory
            result = ModelExporterFactory().export(
                input_file=str(input_file),
                output_file=str(output_file),
                format_name=format.lower(),
                config=config
            )

        # Show completion message
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
        return False

def create_export_mesh_app() -> typer.Typer:
    """Create the mesh export app."""
    app = typer.Typer(help="Export TMD files to various 3D model formats",
                     short_help="Export to 3D model formats")

    # Common parameters for all export commands
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
    
    @app.command("stl")
    def stl(
        input_file: Path = common_params['input_file'],
        output_file: Optional[Path] = common_params['output_file'],
        scale: float = common_params['scale'],
        binary: bool = typer.Option(True, help="Use binary STL format"),
        method: MeshMethod = common_params['method'],
        quality: QualityPreset = common_params['quality'],
        max_triangles: Optional[int] = common_params['max_triangles'],
        error_threshold: Optional[float] = common_params['error_threshold'],
        max_subdivisions: int = typer.Option(4, help="Maximum quadtree subdivisions"),
        min_quad_size: Optional[int] = typer.Option(None, help="Minimum quad size for quadtree subdivision"),
        max_quad_size: Optional[int] = typer.Option(None, help="Maximum quad size for quadtree subdivision"),
    ):
        """Export as STL mesh with optimized defaults."""
        try:
            output_file = output_file or input_file.with_suffix('.stl')
            
            # Calculate parameters based on quality preset
            quality_params = get_quality_params(quality)
            if error_threshold is None:
                error_threshold = quality_params['error_threshold']
            if max_triangles is None:
                max_triangles = quality_params['max_triangles']
            if min_quad_size is None:
                min_quad_size = quality_params['min_quad_size']
            if max_quad_size is None:
                max_quad_size = quality_params['max_quad_size']
            
            success = export_model(
                input_file=input_file,
                output_file=output_file,
                format="stl",
                scale=scale,
                binary=binary,
                method=method,
                error_threshold=error_threshold,
                max_triangles=max_triangles,
                max_subdivisions=max_subdivisions,
                min_quad_size=min_quad_size,
                max_quad_size=max_quad_size,
            )
            
            if not success:
                raise typer.Exit(code=1)
                
        except Exception as e:
            typer.echo(f"Error exporting STL: {e}", err=True)
            raise typer.Exit(code=1)

    @app.command("obj")
    def obj(
        input_file: Path = common_params['input_file'],
        output_file: Optional[Path] = common_params['output_file'],
        scale: float = common_params['scale'],
        method: MeshMethod = common_params['method'],
        quality: QualityPreset = common_params['quality'],
        max_triangles: Optional[int] = common_params['max_triangles'],
        error_threshold: Optional[float] = common_params['error_threshold'],
        min_quad_size: Optional[int] = typer.Option(4, help="Minimum quad size for quadtree subdivision"),
        max_quad_size: Optional[int] = typer.Option(64, help="Maximum quad size for quadtree subdivision"),
        max_subdivisions: Optional[int] = typer.Option(4, help="Maximum quadtree subdivisions"),
    ):
        """Export as OBJ mesh."""
        output_file = output_file or input_file.with_suffix('.obj')
        quality_params = get_quality_params(quality)
        if error_threshold is None:
            error_threshold = quality_params['error_threshold']
        if max_triangles is None:
            max_triangles = quality_params['max_triangles']
        if min_quad_size is None:
            min_quad_size = quality_params['min_quad_size']
        if max_quad_size is None:
            max_quad_size = quality_params['max_quad_size']
        export_model(
            input_file,
            output_file,
            "obj",  # Changed from "wavefront obj" to "obj"
            scale=scale,
            method=method,
            max_triangles=max_triangles,
            error_threshold=error_threshold,
            min_quad_size=min_quad_size,
            max_quad_size=max_quad_size,
            max_subdivisions=max_subdivisions,
        )

    @app.command("ply")
    def ply(
        input_file: Path = common_params['input_file'],
        output_file: Optional[Path] = common_params['output_file'],
        scale: float = common_params['scale'],
        binary: bool = typer.Option(True, help="Use binary PLY format"),
        method: MeshMethod = common_params['method'],
        quality: QualityPreset = common_params['quality'],
        max_triangles: Optional[int] = common_params['max_triangles'],
        error_threshold: Optional[float] = common_params['error_threshold'],
        min_quad_size: Optional[int] = typer.Option(None, help="Minimum quad size for quadtree subdivision"),
        max_quad_size: Optional[int] = typer.Option(None, help="Maximum quad size for quadtree subdivision"),
        max_subdivisions: Optional[int] = typer.Option(None, help="Maximum quadtree subdivisions"),
    ):
        """Export as PLY mesh."""
        output_file = output_file or input_file.with_suffix('.ply')
        quality_params = get_quality_params(quality)
        if error_threshold is None:
            error_threshold = quality_params['error_threshold']
        if max_triangles is None:
            max_triangles = quality_params['max_triangles']
        if min_quad_size is None:
            min_quad_size = quality_params['min_quad_size']
        if max_quad_size is None:
            max_quad_size = quality_params['max_quad_size']
        export_model(
            input_file,
            output_file,
            "ply",
            scale=scale,
            binary=binary,
            method=method,
            max_triangles=max_triangles,
            error_threshold=error_threshold,
            min_quad_size=min_quad_size,
            max_quad_size=max_quad_size,
            max_subdivisions=max_subdivisions,
        )

    @app.command("gltf")
    def gltf(
        input_file: Path = common_params['input_file'],
        output_file: Optional[Path] = common_params['output_file'],
        scale: float = common_params['scale'],
        binary: bool = typer.Option(True, help="Use binary GLTF (GLB) format"),
        method: MeshMethod = common_params['method'],
        quality: QualityPreset = common_params['quality'],
        max_triangles: Optional[int] = common_params['max_triangles'],
        error_threshold: Optional[float] = common_params['error_threshold'],
        min_quad_size: Optional[int] = typer.Option(None, help="Minimum quad size for quadtree subdivision"),
        max_quad_size: Optional[int] = typer.Option(None, help="Maximum quad size for quadtree subdivision"),
        max_subdivisions: Optional[int] = typer.Option(None, help="Maximum quadtree subdivisions"),
    ):
        """Export as GLTF/GLB mesh."""
        suffix = '.glb' if binary else '.gltf'
        output_file = output_file or input_file.with_suffix(suffix)
        quality_params = get_quality_params(quality)
        if error_threshold is None:
            error_threshold = quality_params['error_threshold']
        if max_triangles is None:
            max_triangles = quality_params['max_triangles']
        if min_quad_size is None:
            min_quad_size = quality_params['min_quad_size']
        if max_quad_size is None:
            max_quad_size = quality_params['max_quad_size']
        export_model(
            input_file,
            output_file,
            'glb' if binary else 'gltf',
            scale=scale,
            binary=binary,
            method=method,
            max_triangles=max_triangles,
            error_threshold=error_threshold,
            min_quad_size=min_quad_size,
            max_quad_size=max_quad_size,
            max_subdivisions=max_subdivisions,
        )

    @app.command("usd")
    def usd(
        input_file: Path = common_params['input_file'],
        output_file: Optional[Path] = common_params['output_file'],
        scale: float = common_params['scale'],
        binary: bool = typer.Option(True, help="Use binary USD (USDC) format"),
        method: MeshMethod = common_params['method'],
        quality: QualityPreset = common_params['quality'],
        max_triangles: Optional[int] = common_params['max_triangles'],
        error_threshold: Optional[float] = common_params['error_threshold'],
        min_quad_size: Optional[int] = typer.Option(None, help="Minimum quad size for quadtree subdivision"),
        max_quad_size: Optional[int] = typer.Option(None, help="Maximum quad size for quadtree subdivision"),
        max_subdivisions: Optional[int] = typer.Option(None, help="Maximum quadtree subdivisions"),
    ):
        """Export as USD/USDA/USDC mesh."""
        suffix = '.usdc' if binary else '.usda'
        output_file = output_file or input_file.with_suffix(suffix)
        quality_params = get_quality_params(quality)
        if error_threshold is None:
            error_threshold = quality_params['error_threshold']
        if max_triangles is None:
            max_triangles = quality_params['max_triangles']
        if min_quad_size is None:
            min_quad_size = quality_params['min_quad_size']
        if max_quad_size is None:
            max_quad_size = quality_params['max_quad_size']
        export_model(
            input_file,
            output_file,
            'usdc' if binary else 'usda',
            scale=scale,
            binary=binary,
            method=method,
            max_triangles=max_triangles,
            error_threshold=error_threshold,
            min_quad_size=min_quad_size,
            max_quad_size=max_quad_size,
            max_subdivisions=max_subdivisions,
        )

    return app
