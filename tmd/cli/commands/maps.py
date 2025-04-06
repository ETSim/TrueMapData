"""Map export commands for TMD CLI."""
from pathlib import Path
from typing import List, Optional, Dict, Callable
import os
import logging
import json
import typer

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.panel import Panel

from ..core.ui import print_error, print_success, print_warning
from tmd import TMD
from ...image import (
    get_available_map_types,
    export_ao_map,
    export_normal_map,
    export_bump_map,
    export_roughness_map,
    export_metallic_map,
    export_displacement_map,
    export_height_map,
    export_hillshade_map,
)
from ..core.ui import (
    display_tmd_info, 
    display_map_export_info, 
    display_map_export_results,
    display_batch_progress
)

# Set up logging
logger = logging.getLogger(__name__)
console = Console()

def process_metadata(metadata_str: str) -> Dict:
    """Process metadata string into dictionary."""
    try:
        return json.loads(metadata_str)
    except:
        return {}

def create_map_command(map_type: str, **defaults):
    """Create a map export command."""
    def command(
        input_file: Path = typer.Argument(..., help="Input TMD file", exists=True),
        output_file: Optional[Path] = typer.Option(None, help="Output filename"),
        compress: int = typer.Option(0, help="Compression level (0-100)", min=0, max=100),
        format: str = typer.Option("png", help="Output format (png, jpg, webp)"),
        extra_info: Optional[str] = typer.Option(None, help="Additional parameters as JSON string"),
    ):
        """Export a specific map type."""
        params = {
            'compress': compress, 
            'format': format,
            **defaults
        }
        
        if extra_info:
            params.update(process_metadata(extra_info))
            
        # Use default output filename if none provided
        if output_file is None:
            output_file = input_file.parent / f"{input_file.stem}_{map_type}.{format}"
            
        # Export the map
        export_map(map_type, input_file, output_file, **params)
        
    return command

# Map of export functions
export_funcs: Dict[str, Callable] = {
    'normal': export_normal_map,
    'ao': export_ao_map,
    'bump': export_bump_map,
    'roughness': export_roughness_map,
    'metallic': export_metallic_map,
    'displacement': export_displacement_map,
    'height': export_height_map,
    'hillshade': export_hillshade_map,
}

def list_available_maps():
    """List all available map types."""
    console.print("Available map types:")
    for map_type in get_available_map_types():
        console.print(f"  - {map_type}")

def export_map(map_type: str, input_file: Path, output_file: Path, **kwargs):
    """Export a specific map type."""
    try:
        # Load and validate TMD file
        tmd_data = TMD.load(str(input_file))
        if not tmd_data or not tmd_data.height_map:
            print_error("Invalid TMD file or missing height map")
            return False

        # Display TMD info
        console.print(Panel("[bold cyan]TMD File Details[/]"))
        display_tmd_info(tmd_data)
        
        # Prepare parameters
        metadata = tmd_data.metadata or {}
        params = {**kwargs, 'metadata': metadata}
        
        # Create output directory
        os.makedirs(os.path.dirname(str(output_file)), exist_ok=True)

        # Export with progress tracking
        with Progress() as progress:
            task = progress.add_task(f"[cyan]Generating {map_type} map...", total=None)
            
            # Execute export
            export_funcs[map_type](tmd_data.height_map, str(output_file), **params)
            progress.update(task, completed=True)
        
        print_success(f"Exported {map_type} map to {output_file}")
        return True
        
    except Exception as e:
        print_error(f"Failed to export {map_type} map: {e}")
        return False

def export_all_maps(
    input_file: Path,
    output_dir: Optional[Path] = None,
    types: Optional[List[str]] = None,
    compress: int = 0,
    format: str = "png",
    extra_info: Optional[str] = None
):
    """Export all or specified map types for a single file."""
    try:
        # Validate input file exists
        if not os.path.exists(input_file):
            console.print(f"[red]Input file not found: {input_file}")
            return
            
        # Load the TMD file once for all exports
        tmd_data = TMD(str(input_file))
        height_map = tmd_data.height_map
        
        # Get available map types if none specified
        if types is None:
            types = get_available_map_types()
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export each map type
        for map_type in types:
            if map_type in get_available_map_types():  # Only process valid types
                output_file = output_dir / f"{input_file.stem}_{map_type}.{format}"
                console.print(f"Generating {map_type} map...")
                export_map(map_type, input_file, output_file, compress=compress, format=format, extra_info=extra_info)
    except Exception as e:
        print_error(f"Failed to export maps: {e}")

def batch_export_maps(
    input_dir: Path,
    output_dir: Optional[Path] = None,
    types: Optional[List[str]] = None,
    compress: int = 0,
    format: str = "png",
    extra_info: Optional[str] = None,
    recursive: bool = False,
    pattern: str = "*.tmd",
):
    """Export maps for multiple TMD files."""
    try:
        # Validate input directory exists
        if not os.path.exists(input_dir):
            console.print(f"[red]Input directory not found: {input_dir}")
            return
        
        # Set default output directory if not provided
        if output_dir is None:
            output_dir = input_dir / "exported_maps"
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all TMD files in the input directory
        tmd_files = list(input_dir.rglob(pattern) if recursive else input_dir.glob(pattern))
        
        if not tmd_files:
            console.print(f"[red]No TMD files found in directory: {input_dir}")
            return
        
        # Export maps for each TMD file
        for tmd_file in tmd_files:
            console.print(f"Processing file: {tmd_file}")
            export_all_maps(tmd_file, output_dir, types, compress, format, extra_info)
    except Exception as e:
        print_error(f"Failed to batch export maps: {e}")
