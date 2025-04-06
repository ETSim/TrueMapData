"""Map export functionality for the CLI."""
import typer
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from ..commands.export import export_maps_command
from ..commands.terrain import generate_synthetic_terrain
from ...image import get_available_map_types

def create_export_maps_app() -> typer.Typer:
    """Create the maps export app."""
    app = typer.Typer(help="Export TMD files to various map types")

    @app.command("batch")
    def batch_export(
        input_dir: Path = typer.Argument(Path("data"), help="Input directory containing TMD files"),
        output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory (default: ./textures)"),
        types: Optional[List[str]] = typer.Option(None, "--types", "-t", help="Map types (default: all)"),
        recursive: bool = typer.Option(False, "--recursive", "-r", help="Search subdirectories"),
        compress: int = typer.Option(0, "--compress", "-c", help="Compression (0-100)"),
        format: str = typer.Option("png", "--format", "-f", help="Output format"),
        pattern: str = typer.Option("*.tmd", "--pattern", "-p", help="File pattern to match")
    ):
        """Export maps for multiple TMD files in a directory."""
        # Set default output directory if not specified
        if output_dir is None:
            output_dir = Path("textures")
            
        # Find all TMD files
        files = list(input_dir.rglob(pattern) if recursive else input_dir.glob(pattern))
        
        if not files:
            typer.echo(f"No TMD files found in {input_dir}")
            raise typer.Exit(1)
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
            
        # Process each file
        with typer.progressbar(files, label="Processing files") as progress:
            for file in progress:
                typer.echo(f"\nProcessing {file.name}...")
                file_output_dir = output_dir / file.stem
                file_output_dir.mkdir(parents=True, exist_ok=True)
                
                export_maps_command(
                    file, 
                    file_output_dir, 
                    types,
                    compress=compress,
                    format=format
                )

    @app.command("normal")
    def normal(
        input_file: Path = typer.Argument(..., help="Input TMD file", exists=True),
        output_file: Optional[Path] = typer.Option(None, "--output", "-o", help="Output filename"),
        strength: float = typer.Option(1.0, "--strength", "-s", help="Normal map strength"),
        compress: int = typer.Option(0, "--compress", "-c", help="Compression (0-100)"),
        format: str = typer.Option("png", "--format", "-f", help="Output format")
    ):
        """Export a normal map."""
        export_maps_command(input_file, output_file.parent if output_file else None, 
                          ["normal"], strength=strength, compress=compress, format=format)

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
        output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory (default: ./textures)"),
        types: Optional[List[str]] = typer.Option(None, "--types", "-t", help="Map types to generate"),
        compress: int = typer.Option(0, "--compress", "-c", help="Compression (0-100)"),
        format: str = typer.Option("png", "--format", "-f", help="Output format"),
        strength: float = typer.Option(1.0, "--strength", "-s", help="Map strength"),
    ):
        """Export all or specified map types."""
        # If no output dir specified, use textures subdirectory
        if output_dir is None:
            output_dir = Path("textures")
            
        export_maps_command(input_file, output_dir, types, 
                          compress=compress, format=format, strength=strength)

    @app.command("synthetic")
    def synthetic(
        pattern: str = typer.Argument(
            "waves", 
            help="Pattern type (waves, peak, dome, ramp, combined, flat, random, perlin, fbm, square, sawtooth)"
        ),
        width: int = typer.Option(1024, "--width", "-w", help="Width of the height map"),
        height: int = typer.Option(1024, "--height", "-h", help="Height of the height map"),
        output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory (default: ./textures)"),
        types: Optional[List[str]] = typer.Option(None, "--types", "-t", help="Map types to generate"),
        compress: int = typer.Option(0, "--compress", "-c", help="Compression (0-100)"),
        format: str = typer.Option("png", "--format", "-f", help="Output format")
    ):
        """Generate and export maps from synthetic TMD data."""
        if not generate_synthetic_terrain(pattern, width, height, output_dir, types, compress, format):
            raise typer.Exit(1)

    return app
