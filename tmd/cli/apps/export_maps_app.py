"""Map export functionality for the CLI."""
import typer
from pathlib import Path
from typing import List, Optional

from ..commands.maps import list_available_maps, export_map, export_all_maps

def create_export_maps_app() -> typer.Typer:
    """Create the maps export app."""
    app = typer.Typer()
    
    @app.command("list")
    def list_maps():
        """List all available map types."""
        list_available_maps()
    
    @app.command("ao")
    def ao(
        input_file: Path = typer.Argument(..., help="Input TMD file", exists=True),
        output_file: Optional[Path] = typer.Option(None, help="Output filename"),
        samples: int = typer.Option(16, help="Number of AO samples"),
        strength: float = typer.Option(1.0, help="AO effect strength"),
    ):
        """Export an ambient occlusion map."""
        output_file = output_file or input_file.with_suffix('.ao.png')
        export_map("ao", input_file, output_file, samples=samples, strength=strength)

    @app.command("normal")
    def normal(
        input_file: Path = typer.Argument(..., help="Input TMD file", exists=True),
        output_file: Optional[Path] = typer.Option(None, help="Output filename"),
        strength: float = typer.Option(1.0, help="Normal map strength"),
    ):
        """Export a normal map."""
        output_file = output_file or input_file.with_suffix('.normal.png')
        export_map("normal", input_file, output_file, strength=strength)

    @app.command("all")
    def all_maps(
        input_file: Path = typer.Argument(..., help="Input TMD file", exists=True),
        output_dir: Optional[Path] = typer.Option(None, help="Output directory"),
        types: Optional[List[str]] = typer.Option(None, help="List of map types to generate"),
    ):
        """Export all or specified map types."""
        output_dir = output_dir or input_file.parent
        export_all_maps(input_file, output_dir, types)
    
    return app
