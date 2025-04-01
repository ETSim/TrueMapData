"""Map export commands for TMD CLI."""
from pathlib import Path
from typing import List, Optional
import os

from rich.console import Console

from ..core.ui import print_error, print_success
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

console = Console()

def list_available_maps():
    """List all available map types."""
    console.print("Available map types:")
    for map_type in get_available_map_types():
        console.print(f"  - {map_type}")

def export_map(map_type: str, input_file: Path, output_file: Path, **kwargs):
    """Export a specific map type."""
    try:
        # Validate input file exists
        if not os.path.exists(input_file):
            console.print(f"[red]Input file not found: {input_file}")
            return
            
        # Load the TMD file to get height map data
        tmd_data = TMD(str(input_file))
        height_map = tmd_data.height_map
        metadata = tmd_data.metadata
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Add scale info from metadata for normal maps
        if map_type == "normal" and metadata:
            # Make sure to pass both strength and metadata
            kwargs['strength'] = kwargs.get('strength', 1.0)
            kwargs['metadata'] = metadata

        # Pass metadata for all map types
        kwargs['metadata'] = metadata
        
        # Add default colormap for height maps
        if map_type == "height":
            kwargs.setdefault('colormap', 'viridis')

        export_funcs = {
            "ao": export_ao_map,
            "normal": export_normal_map,
            "bump": export_bump_map,
            "roughness": export_roughness_map,
            "metallic": export_metallic_map,
            "displacement": export_displacement_map,
            "height": export_height_map,
            "hillshade": export_hillshade_map,
        }
        
        if map_type not in export_funcs:
            console.print(f"[red]Unknown map type: {map_type}")
            return
            
        export_funcs[map_type](height_map, output_file, **kwargs)
        print_success(f"Exported {map_type} map to {output_file}")
        
    except Exception as e:
        print_error(f"Failed to export {map_type} map: {e}")

def export_all_maps(input_file: Path, output_dir: Path, types: Optional[List[str]] = None):
    """Export multiple map types."""
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
                output_file = output_dir / f"{input_file.stem}_{map_type}.png"
                console.print(f"Generating {map_type} map...")
                export_map(map_type, input_file, output_file)
    except Exception as e:
        print_error(f"Failed to export maps: {e}")
