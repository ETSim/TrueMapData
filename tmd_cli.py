#!/usr/bin/env python3
"""TMD Command-Line Interface"""
import sys
import typer
from rich.console import Console

# Initialize console
console = Console()

# Import core functionality
try:
    from tmd.cli.core import check_dependencies
    
    # Import command modules
    from tmd.cli.apps.compress_app import create_compress_app
    from tmd.cli.apps.config_app import create_config_app
    from tmd.cli.apps.visualize_app import create_visualize_app
    from tmd.cli.apps.cache_app import create_cache_app
    from tmd.cli.apps.export_maps_app import create_export_maps_app
    from tmd.cli.apps.info_app import info_command, version_command, check_command
    from tmd.cli.apps.export_mesh_app import create_export_mesh_app
    from tmd.cli.apps.terrain_app import create_terrain_app
    
    # Import the fixed model generation app
    from tmd.cli.commands.model import create_model_app
    
except ImportError as e:
    console.print(f"[red]Error importing TMD modules: {e}[/red]")
    console.print("[yellow]Make sure TMD is properly installed[/yellow]")
    sys.exit(1)

# Create main app
app = typer.Typer(
    help="TMD Command Line Interface - Tools for working with Topographic Mesh Data files",
    add_completion=False
)

# Register main info commands
app.command(name="info", help="Show TMD file information")(info_command)
app.command(name="version", help="Show TMD version")(version_command)
app.command(name="check", help="Check system dependencies")(check_command)

# Add subcommands with proper organization
def add_subcommands():
    """Add all subcommand groups to the main app."""
    
    # Core functionality
    app.add_typer(
        create_config_app(), 
        name="config", 
        help="Configuration management"
    )
    
    app.add_typer(
        create_cache_app(), 
        name="cache", 
        help="Cache management"
    )
    
    # File operations
    app.add_typer(
        create_compress_app(), 
        name="compress", 
        help="Compress and decompress TMD files"
    )
    
    # Export functionality
    app.add_typer(
        create_export_maps_app(), 
        name="maps", 
        help="Export TMD files to various map formats (normal, height, etc.)"
    )
    
    app.add_typer(
        create_export_mesh_app(), 
        name="mesh", 
        help="Export TMD files to 3D model formats (legacy commands)"
    )
    
    # New model generation commands (replaces mesh commands)
    app.add_typer(
        create_model_app(), 
        name="model", 
        help="Generate 3D models from TMD files (single and batch)"
    )
    
    # Visualization and generation
    app.add_typer(
        create_visualize_app(), 
        name="visualize", 
        help="Visualize TMD files"
    )
    
    app.add_typer(
        create_terrain_app(), 
        name="terrain", 
        help="Generate synthetic terrain and textures"
    )

# Add all subcommands
add_subcommands()

@app.callback()
def main_callback():
    """
    TMD Command-Line Tools - Work with Topographic Mesh Data files
    
    This tool provides comprehensive functionality for working with TMD files including:
    - Model generation (single and batch processing)
    - Map exports (normal maps, heightmaps, etc.)
    - Visualization and analysis
    - Terrain generation
    - Configuration management
    """
    # Check dependencies when the app starts
    check_dependencies(auto_install=False, exit_on_failure=True)

def main():
    """Run the TMD CLI application."""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        sys.exit(1)

if __name__ == "__main__":
    sys.exit(main())