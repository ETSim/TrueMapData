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
    ):
        """Export an ambient occlusion map."""
        output_dir = Path(output_file).parent if output_file else input_file.parent
        export_maps_command(input_file, output_dir, ["ao"], samples=samples, strength=strength)

    @app.command("normal")
    def normal(
        input_file: Path = typer.Argument(..., help="Input TMD file", exists=True),
        output_file: Optional[Path] = typer.Option(None, help="Output filename"),
        strength: float = typer.Option(1.0, help="Normal map strength"),
    ):
        """Export a normal map."""
        output_dir = Path(output_file).parent if output_file else input_file.parent
        export_maps_command(input_file, output_dir, ["normal"], strength=strength)

    @app.command("all")
    def all_maps(
        input_file: Path = typer.Argument(..., help="Input TMD file", exists=True),
        output_dir: Optional[Path] = typer.Option(None, help="Output directory"),
        types: Optional[List[str]] = typer.Option(None, help="List of map types to generate"),
    ):
        """Export all or specified map types."""
        output_dir = output_dir or input_file.parent
        export_maps_command(input_file, output_dir, types)
    
    return app
