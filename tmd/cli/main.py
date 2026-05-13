"""TMD Typer application and CLI entry points (``tmd-process``, ``tmd-wear``)."""

from __future__ import annotations

import sys

if sys.platform == "win32":
    for _stream in (getattr(sys, "stdout", None), getattr(sys, "stderr", None)):
        if _stream is not None and hasattr(_stream, "reconfigure"):
            try:
                _stream.reconfigure(encoding="utf-8", errors="replace")
            except OSError:
                pass

import typer
from rich.console import Console

from tmd.cli.core import check_dependencies
from tmd.cli.apps.compress_app import create_compress_app
from tmd.cli.apps.config_app import create_config_app
from tmd.cli.apps.visualize_app import create_visualize_app
from tmd.cli.apps.cache_app import create_cache_app
from tmd.cli.apps.export_maps_app import create_export_maps_app
from tmd.cli.apps.info_app import info_command, version_command, check_command
from tmd.cli.apps.export_mesh_app import create_export_mesh_app
from tmd.cli.apps.roughness_app import create_roughness_app
from tmd.cli.apps.sequence_app import create_sequence_app
from tmd.cli.apps.terrain_app import create_terrain_app
from tmd.cli.apps.defect_app import create_defect_app
from tmd.cli.apps.tribology_app import create_tribology_app
from tmd.cli.apps.wear_app import create_wear_app

console = Console()

app = typer.Typer(
    help="TMD Command Line Interface - Tools for working with Topographic Mesh Data files",
    add_completion=False,
)

app.command(name="info", help="Show TMD file information")(info_command)
app.command(name="version", help="Show TMD version")(version_command)
app.command(name="check", help="Check system dependencies")(check_command)

# Same Typer instance is mounted under ``tmd-process wear`` and used by ``tmd-wear``.
wear_app = create_wear_app()


def _add_subcommands() -> None:
    app.add_typer(create_config_app(), name="config", help="Configuration management")
    app.add_typer(create_cache_app(), name="cache", help="Cache management")
    app.add_typer(create_compress_app(), name="compress", help="Compress and decompress TMD files")
    app.add_typer(
        create_export_maps_app(),
        name="maps",
        help="Export TMD files to various map formats (normal, height, etc.)",
    )
    app.add_typer(create_export_mesh_app(), name="mesh", help="Export TMD files to 3D model formats")
    app.add_typer(
        create_sequence_app(),
        name="sequence",
        help="Align and crop sequences of TMD height maps (OpenCV)",
    )
    app.add_typer(
        create_roughness_app(),
        name="roughness",
        help="Areal roughness (ISO 25178) via optional Surfalize",
    )
    app.add_typer(
        create_defect_app(),
        name="defect",
        help="Detect pits, peaks, scratches, cracks and directionality anomalies",
    )
    app.add_typer(
        create_tribology_app(),
        name="tribology",
        help="Tribology metrics: texture axis, contact curve, lubrication ISO volumes",
    )
    app.add_typer(
        wear_app,
        name="wear",
        help="Wear-oriented surface metrics (Abbott curve, wear volume, hazard maps, …)",
    )
    app.add_typer(create_visualize_app(), name="visualize", help="Visualize TMD files")
    app.add_typer(create_terrain_app(), name="terrain", help="Generate synthetic terrain and textures")


_add_subcommands()


@app.callback()
def main_callback() -> None:
    """Load CLI and verify optional dependencies."""
    check_dependencies(auto_install=False, exit_on_failure=True)


def main() -> int:
    """Run the TMD CLI application."""
    try:
        app()
        return 0
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        return 1
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        return 1


def wear_main() -> int:
    """Run the wear toolkit CLI (``tmd-wear`` console script)."""
    try:
        wear_app()
        return 0
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        return 1
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        return 1

