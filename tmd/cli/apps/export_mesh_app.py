"""Mesh export functionality for the CLI."""
import typer
from pathlib import Path
from typing import Optional, Tuple
from enum import Enum

from ..commands.model import export_model
from ...model.formats import get_available_formats

class MeshMethod(str, Enum):
    """Mesh generation methods."""
    ADAPTIVE = "adaptive"
    QUADTREE = "quadtree"

def create_export_mesh_app() -> typer.Typer:
    """Create the mesh export app."""
    app = typer.Typer(help="Export TMD files to various 3D model formats",
                     short_help="Export to 3D model formats")
    
    @app.command("list")
    def list_formats():
        """List all available mesh export formats."""
        for format in get_available_formats():
            typer.echo(f"  - {format}")
    
    @app.command("stl")
    def stl(
        input_file: Path = typer.Argument(..., help="Input TMD file", exists=True),
        output_file: Optional[Path] = typer.Option(None, help="Output filename"),
        scale: float = typer.Option(1.0, help="Mesh scale factor"),
        binary: bool = typer.Option(True, help="Use binary STL format"),
        method: MeshMethod = typer.Option(MeshMethod.ADAPTIVE, help="Mesh generation method"),
        error_threshold: float = typer.Option(0.01, help="Error threshold for mesh generation"),
        min_quad_size: int = typer.Option(2, help="Minimum quadtree cell size"),
        max_quad_size: int = typer.Option(32, help="Maximum quadtree cell size"),
        curvature_threshold: float = typer.Option(0.1, help="Curvature threshold for subdivision"),
        max_triangles: Optional[int] = typer.Option(None, help="Maximum number of triangles"),
        simplify_ratio: Optional[float] = typer.Option(None, help="Mesh simplification ratio"),
        use_feature_edges: bool = typer.Option(True, help="Preserve feature edges"),
        smoothing: float = typer.Option(0.0, help="Smoothing factor (0.0 to 1.0)"),
    ):
        """Export as STL mesh with advanced mesh generation options."""
        output_file = output_file or input_file.with_suffix('.stl')
        export_model(
            input_file, output_file, "stl",
            scale=scale,
            binary=binary,
            method=method,
            error_threshold=error_threshold,
            min_quad_size=min_quad_size,
            max_quad_size=max_quad_size,
            curvature_threshold=curvature_threshold,
            max_triangles=max_triangles,
            simplify_ratio=simplify_ratio,
            use_feature_edges=use_feature_edges,
            smoothing=smoothing,
        )

    @app.command("obj")
    def obj(
        input_file: Path = typer.Argument(..., help="Input TMD file", exists=True),
        output_file: Optional[Path] = typer.Option(None, help="Output filename"),
        scale: float = typer.Option(1.0, help="Mesh scale factor"),
    ):
        """Export as OBJ mesh."""
        output_file = output_file or input_file.with_suffix('.obj')
        export_model(input_file, output_file, "obj", scale=scale)

    @app.command("ply")
    def ply(
        input_file: Path = typer.Argument(..., help="Input TMD file", exists=True),
        output_file: Optional[Path] = typer.Option(None, help="Output filename"),
        scale: float = typer.Option(1.0, help="Mesh scale factor"),
        binary: bool = typer.Option(True, help="Use binary PLY format"),
    ):
        """Export as PLY mesh."""
        output_file = output_file or input_file.with_suffix('.ply')
        export_model(input_file, output_file, "ply", scale=scale, binary=binary)

    @app.command("gltf")
    def gltf(
        input_file: Path = typer.Argument(..., help="Input TMD file", exists=True),
        output_file: Optional[Path] = typer.Option(None, help="Output filename"),
        scale: float = typer.Option(1.0, help="Mesh scale factor"),
        binary: bool = typer.Option(True, help="Use binary GLTF (GLB) format"),
    ):
        """Export as GLTF/GLB mesh."""
        suffix = '.glb' if binary else '.gltf'
        output_file = output_file or input_file.with_suffix(suffix)
        export_model(input_file, output_file, "gltf", scale=scale, binary=binary)

    @app.command("usd")
    def usd(
        input_file: Path = typer.Argument(..., help="Input TMD file", exists=True),
        output_file: Optional[Path] = typer.Option(None, help="Output filename"),
        scale: float = typer.Option(1.0, help="Mesh scale factor"),
        binary: bool = typer.Option(True, help="Use binary USD (USDC) format"),
    ):
        """Export as USD/USDA/USDC mesh."""
        suffix = '.usdc' if binary else '.usda'
        output_file = output_file or input_file.with_suffix(suffix)
        export_model(input_file, output_file, "usd", scale=scale, binary=binary)

    return app
