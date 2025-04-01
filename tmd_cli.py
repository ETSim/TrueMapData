#!/usr/bin/env python3
"""TMD Command-Line Interface"""
import sys
import typer
from rich.console import Console

# Initialize console
console = Console()

# Import core functionality
from tmd.cli.core import check_dependencies

# Import command modules
from tmd.cli.apps.compress_app import create_compress_app
from tmd.cli.apps.config_app import create_config_app
from tmd.cli.apps.visualize_app import create_visualize_app
from tmd.cli.apps.cache_app import create_cache_app
from tmd.cli.apps.export_maps_app import create_export_maps_app
from tmd.cli.apps.info_app import info_command, version_command, check_command

# Create main app
app = typer.Typer(
    help="TMD Command Line Interface - Tools for working with Topographic Mesh Data files",
)

# Register main commands
app.command(name="info")(info_command)
app.command(name="version")(version_command)
app.command(name="check")(check_command)

# Add subcommands
app.add_typer(create_compress_app(), name="compress")
app.add_typer(create_config_app(), name="config")
app.add_typer(create_visualize_app(), name="visualize")
app.add_typer(create_cache_app(), name="cache")

# Add map exporter
maps_app = create_export_maps_app()
app.add_typer(maps_app, name="maps")

@app.callback()
def callback():
    """TMD Command-Line Tools - Work with Topographic Mesh Data files"""
    check_dependencies(auto_install=False, exit_on_failure=True)

def main():
    """Run the TMD CLI application."""
    app()

if __name__ == "__main__":
    sys.exit(main())