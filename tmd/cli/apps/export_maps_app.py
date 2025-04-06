"""Map export functionality for the CLI."""
import typer
from pathlib import Path
from typing import List, Optional

from ..commands.export import export_maps_command
from ...image import get_available_map_types

def create_export_maps_app() -> typer.Typer:
    """Create the maps export app."""
    app = typer.Typer(help="Export TMD files to various map types (normal maps, height maps, etc.)",
                      short_help="Export to map formats")
    
    @app.command("list")
    def list_maps():
        """List all available map types."""
        for map_type in get_available_map_types():
            typer.echo(f"  - {map_type}")
    
    @app.command("ao")
    def ao(
        input_file: Path = typer.Argument(..., help="Input TMD file", exists=True),
        output_file: Optional[Path] = typer.Option(None, help="Output filename"),
        samples: int = typer.Option(16, help="Number of AO samples"),
        strength: float = typer.Option(1.0, help="AO effect strength"),
        compress: int = typer.Option(0, help="Compression level (0-100)", min=0, max=100),
        format: str = typer.Option("png", help="Output format (png, jpg, webp)")
    ):
        """Export an ambient occlusion map."""
        output_dir = Path(output_file).parent if output_file else input_file.parent
        export_maps_command(input_file, output_dir, ["ao"], samples=samples, strength=strength, 
                          compress=compress, format=format)

    @app.command("normal")
    def normal(
        input_file: Path = typer.Argument(..., help="Input TMD file", exists=True),
        output_file: Optional[Path] = typer.Option(None, help="Output filename"),
        strength: float = typer.Option(1.0, help="Normal map strength"),
        compress: int = typer.Option(0, help="Compression level (0-100)", min=0, max=100),
        format: str = typer.Option("png", help="Output format (png, jpg, webp)")
    ):
        """Export a normal map."""
        output_dir = Path(output_file).parent if output_file else input_file.parent
        export_maps_command(input_file, output_dir, ["normal"], strength=strength, 
                          compress=compress, format=format)

    @app.command("bump")
    def bump(
        input_file: Path = typer.Argument(..., help="Input TMD file", exists=True),
        output_file: Optional[Path] = typer.Option(None, help="Output filename"),
        strength: float = typer.Option(1.0, help="Bump map strength"),
        compress: int = typer.Option(0, help="Compression level (0-100)", min=0, max=100),
        format: str = typer.Option("png", help="Output format (png, jpg, webp)")
    ):
        """Export a bump map."""
        output_dir = Path(output_file).parent if output_file else input_file.parent
        export_maps_command(input_file, output_dir, ["bump"], strength=strength,
                          compress=compress, format=format)

    @app.command("roughness")
    def roughness(
        input_file: Path = typer.Argument(..., help="Input TMD file", exists=True),
        output_file: Optional[Path] = typer.Option(None, help="Output filename"),
        strength: float = typer.Option(1.0, help="Roughness map strength"),
        compress: int = typer.Option(0, help="Compression level (0-100)", min=0, max=100),
        format: str = typer.Option("png", help="Output format (png, jpg, webp)")
    ):
        """Export a roughness map."""
        output_dir = Path(output_file).parent if output_file else input_file.parent
        export_maps_command(input_file, output_dir, ["roughness"], strength=strength,
                          compress=compress, format=format)

    @app.command("metallic")
    def metallic(
        input_file: Path = typer.Argument(..., help="Input TMD file", exists=True),
        output_file: Optional[Path] = typer.Option(None, help="Output filename"),
        strength: float = typer.Option(1.0, help="Metallic map strength"),
        compress: int = typer.Option(0, help="Compression level (0-100)", min=0, max=100),
        format: str = typer.Option("png", help="Output format (png, jpg, webp)")
    ):
        """Export a metallic map."""
        output_dir = Path(output_file).parent if output_file else input_file.parent
        export_maps_command(input_file, output_dir, ["metallic"], strength=strength,
                          compress=compress, format=format)

    @app.command("displacement")
    def displacement(
        input_file: Path = typer.Argument(..., help="Input TMD file", exists=True),
        output_file: Optional[Path] = typer.Option(None, help="Output filename"),
        intensity: float = typer.Option(1.0, help="Displacement intensity"),
        compress: int = typer.Option(0, help="Compression level (0-100)", min=0, max=100),
        format: str = typer.Option("png", help="Output format (png, jpg, webp)")
    ):
        """Export a displacement map."""
        output_dir = Path(output_file).parent if output_file else input_file.parent
        export_maps_command(input_file, output_dir, ["displacement"], intensity=intensity,
                          compress=compress, format=format)

    @app.command("height")
    def height(
        input_file: Path = typer.Argument(..., help="Input TMD file", exists=True),
        output_file: Optional[Path] = typer.Option(None, help="Output filename"),
        colormap: str = typer.Option("viridis", help="Colormap to use"),
        compress: int = typer.Option(0, help="Compression level (0-100)", min=0, max=100),
        format: str = typer.Option("png", help="Output format (png, jpg, webp)")
    ):
        """Export a height map."""
        output_dir = Path(output_file).parent if output_file else input_file.parent
        export_maps_command(input_file, output_dir, ["height"], colormap=colormap,
                          compress=compress, format=format)

    @app.command("hillshade")
    def hillshade(
        input_file: Path = typer.Argument(..., help="Input TMD file", exists=True),
        output_file: Optional[Path] = typer.Option(None, help="Output filename"),
        azimuth: float = typer.Option(315.0, help="Light source azimuth in degrees"),
        altitude: float = typer.Option(45.0, help="Light source altitude in degrees"),
        compress: int = typer.Option(0, help="Compression level (0-100)", min=0, max=100),
        format: str = typer.Option("png", help="Output format (png, jpg, webp)")
    ):
        """Export a hillshade map."""
        output_dir = Path(output_file).parent if output_file else input_file.parent
        export_maps_command(input_file, output_dir, ["hillshade"], azimuth=azimuth, altitude=altitude,
                          compress=compress, format=format)

    @app.command("all")
    def all_maps(
        input_file: Path = typer.Argument(..., help="Input TMD file", exists=True),
        output_dir: Optional[Path] = typer.Option(None, help="Output directory"),
        types: Optional[List[str]] = typer.Option(None, help="List of map types to generate"),
        compress: int = typer.Option(0, help="Compression level (0-100)", min=0, max=100),
        format: str = typer.Option("png", help="Output format (png, jpg, webp)")
    ):
        """Export all or specified map types."""
        output_dir = output_dir or input_file.parent
        export_maps_command(input_file, output_dir, types, compress=compress, format=format)
    
    return app
