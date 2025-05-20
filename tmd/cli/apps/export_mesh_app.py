"""Mesh export functionality for the CLI"""
import typer
from pathlib import Path
from typing import Optional
from enum import Enum
import logging

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
            'error_threshold': 0.1,
            'max_triangles': 25000,
            'min_quad_size': 8,
            'max_quad_size': 64,
            'max_subdivisions': 6
        },
        QualityPreset.NORMAL: {
            'error_threshold': 0.02,
            'max_triangles': 100000,
            'min_quad_size': 4,
            'max_quad_size': 32,
            'max_subdivisions': 8
        },
        QualityPreset.HIGH: {
            'error_threshold': 0.005,
            'max_triangles': 500000,
            'min_quad_size': 2,
            'max_quad_size': 16,
            'max_subdivisions': 10
        },
        QualityPreset.ULTRA: {
            'error_threshold': 0.0005,
            'max_triangles': 2500000,
            'min_quad_size': 1,
            'max_quad_size': 8,
            'max_subdivisions': 12
        }
    }
    return presets[preset]

def create_export_mesh_app() -> typer.Typer:
    """Create the mesh export app."""
    app = typer.Typer(
        help="Export TMD files to various 3D model formats (Legacy - use 'tmd model' instead)",
        short_help="Export to 3D model formats (Legacy)"
    )

    @app.command("list")
    def list_formats():
        """List all available mesh export formats."""
        console.print("[yellow]Note: This is the legacy mesh export command.[/yellow]")
        console.print("[cyan]Please use 'tmd model formats' for the updated command.[/cyan]")
        
        formats = ["stl", "obj", "ply", "gltf", "usd"]
        for format_name in formats:
            console.print(f"  - {format_name}")

    @app.command("stl")
    def export_stl(
        input_file: Path = typer.Argument(..., help="Input TMD file"),
        output_file: Optional[Path] = typer.Option(None, help="Output filename"),
        scale: float = typer.Option(1.0, help="Mesh scale factor"),
        method: MeshMethod = typer.Option(MeshMethod.ADAPTIVE, help="Mesh generation method"),
        quality: QualityPreset = typer.Option(QualityPreset.NORMAL, help="Quality preset"),
        binary: bool = typer.Option(True, help="Use binary format")
    ):
        """Export as STL mesh (Legacy - use 'tmd model generate' instead)."""
        console.print("[yellow]Warning: This is a legacy command.[/yellow]")
        console.print("[cyan]Please use: tmd model generate --format stl[/cyan]")
        
        # Set output file if not specified
        if output_file is None:
            output_file = input_file.with_suffix(".stl")
        
        console.print(f"Would export {input_file} to {output_file}")
        console.print("Use the new 'tmd model generate' command for actual export functionality.")

    @app.command("obj")
    def export_obj(
        input_file: Path = typer.Argument(..., help="Input TMD file"),
        output_file: Optional[Path] = typer.Option(None, help="Output filename"),
        scale: float = typer.Option(1.0, help="Mesh scale factor"),
        method: MeshMethod = typer.Option(MeshMethod.ADAPTIVE, help="Mesh generation method"),
        quality: QualityPreset = typer.Option(QualityPreset.NORMAL, help="Quality preset")
    ):
        """Export as OBJ mesh (Legacy - use 'tmd model generate' instead)."""
        console.print("[yellow]Warning: This is a legacy command.[/yellow]")
        console.print("[cyan]Please use: tmd model generate --format obj[/cyan]")
        
        # Set output file if not specified
        if output_file is None:
            output_file = input_file.with_suffix(".obj")
        
        console.print(f"Would export {input_file} to {output_file}")
        console.print("Use the new 'tmd model generate' command for actual export functionality.")

    @app.command("ply")
    def export_ply(
        input_file: Path = typer.Argument(..., help="Input TMD file"),
        output_file: Optional[Path] = typer.Option(None, help="Output filename"),
        scale: float = typer.Option(1.0, help="Mesh scale factor"),
        method: MeshMethod = typer.Option(MeshMethod.ADAPTIVE, help="Mesh generation method"),
        quality: QualityPreset = typer.Option(QualityPreset.NORMAL, help="Quality preset"),
        binary: bool = typer.Option(True, help="Use binary format")
    ):
        """Export as PLY mesh (Legacy - use 'tmd model generate' instead)."""
        console.print("[yellow]Warning: This is a legacy command.[/yellow]")
        console.print("[cyan]Please use: tmd model generate --format ply[/cyan]")
        
        # Set output file if not specified
        if output_file is None:
            output_file = input_file.with_suffix(".ply")
        
        console.print(f"Would export {input_file} to {output_file}")
        console.print("Use the new 'tmd model generate' command for actual export functionality.")

    @app.command("gltf")
    def export_gltf(
        input_file: Path = typer.Argument(..., help="Input TMD file"),
        output_file: Optional[Path] = typer.Option(None, help="Output filename"),
        scale: float = typer.Option(1.0, help="Mesh scale factor"),
        method: MeshMethod = typer.Option(MeshMethod.ADAPTIVE, help="Mesh generation method"),
        quality: QualityPreset = typer.Option(QualityPreset.NORMAL, help="Quality preset"),
        binary: bool = typer.Option(True, help="Use binary format")
    ):
        """Export as GLTF/GLB mesh (Legacy - use 'tmd model generate' instead)."""
        console.print("[yellow]Warning: This is a legacy command.[/yellow]")
        console.print("[cyan]Please use: tmd model generate --format gltf[/cyan]")
        
        # Set output file if not specified
        if output_file is None:
            suffix = ".glb" if binary else ".gltf"
            output_file = input_file.with_suffix(suffix)
        
        console.print(f"Would export {input_file} to {output_file}")
        console.print("Use the new 'tmd model generate' command for actual export functionality.")

    @app.command("usd")
    def export_usd(
        input_file: Path = typer.Argument(..., help="Input TMD file"),
        output_file: Optional[Path] = typer.Option(None, help="Output filename"),
        scale: float = typer.Option(1.0, help="Mesh scale factor"),
        method: MeshMethod = typer.Option(MeshMethod.ADAPTIVE, help="Mesh generation method"),
        quality: QualityPreset = typer.Option(QualityPreset.NORMAL, help="Quality preset"),
        binary: bool = typer.Option(True, help="Use binary format")
    ):
        """Export as USD mesh (Legacy - use 'tmd model generate' instead)."""
        console.print("[yellow]Warning: This is a legacy command.[/yellow]")
        console.print("[cyan]Please use: tmd model generate --format usd[/cyan]")
        
        # Set output file if not specified
        if output_file is None:
            suffix = ".usd" if binary else ".usda"
            output_file = input_file.with_suffix(suffix)
        
        console.print(f"Would export {input_file} to {output_file}")
        console.print("Use the new 'tmd model generate' command for actual export functionality.")

    return app