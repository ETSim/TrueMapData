"""Export commands for TMD CLI."""
from pathlib import Path
from typing import Optional, List
import time
from datetime import datetime

import logging

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
from rich.console import Console

from tmd.model.base import ModelExporter
from tmd.model.registry import get_available_formats

from ..core.ui import console, print_error, print_success, display_tmd_info
from ...image import MapExporter, get_available_map_types
from ...core import TMD


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def export_command(
    tmd_file: Path,
    output: Optional[Path] = None,
    format: str = "npz",
    **kwargs
) -> bool:
    """Export a TMD file to another format."""
    try:
        if format in get_available_map_types():
            MapExporter.export(tmd_file, output, format, **kwargs)
        elif format in get_available_formats():
            ModelExporter.export(tmd_file, output, format, **kwargs)
        else:
            print_error(f"Unknown export format: {format}")
            return False
        return True
    except Exception as e:
        print_error(f"Export failed: {e}")
        return False

def display_config_info(input_file: Path, output_dir: Path, types: List[str], params: dict):
    """Display export configuration."""
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Input File", str(input_file))
    table.add_row("Output Directory", str(output_dir))
    table.add_row("Map Types", ", ".join(types))
    table.add_row("Parameters", str(params))
    
    console.print("\n[bold cyan]Export Configuration[/]")
    console.print(table)
    console.print()

def export_maps_command(
    tmd_file: Path,
    output_dir: Optional[Path] = None,
    types: Optional[List[str]] = None,
    **kwargs
) -> bool:
    """Export a TMD file to multiple map types."""
    try:
        start_time = time.time()
        
        # Ensure compression, format and normalize parameters
        compress = kwargs.pop('compress', 75)  # Use pop instead of get to remove it
        format = kwargs.pop('format', 'png')   # Use pop instead of get to remove it
        normalize = kwargs.pop('normalize', True)  # Add normalize parameter
        
        # Load TMD file first to validate
        console.print("\n[bold cyan]Loading TMD file...[/]")
        tmd_data = TMD.load(str(tmd_file))
        
        # Set default output directory to 'textures'
        if output_dir is None:
            output_dir = tmd_file.parent / "textures"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Display TMD info and config
        console.print("\n[bold cyan]TMD File Details[/]")
        if hasattr(tmd_data, 'metadata'):
            for key, value in tmd_data.metadata.items():
                console.print(f"{key}: {value}")
                
        # Get map types to export (all available if none specified)
        if types is None:
            types = get_available_map_types()
            
        # Show export configuration
        display_config_info(tmd_file, output_dir, types, kwargs)
        
        # Track results
        results = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            main_task = progress.add_task("[cyan]Exporting maps...", total=len(types))
            
            for map_type in types:
                map_start_time = time.time()
                progress.update(main_task, description=f"[cyan]Generating {map_type} map...")
                
                # Build output path
                output_file = f"{tmd_file.stem}_{map_type}.{format}"
                output_path = output_dir / output_file
                
                # Export the map with proper parameters
                success = MapExporter.export_map(
                    tmd_data.height_map, 
                    str(output_path), 
                    map_type,
                    compress=compress,
                    format=format,
                    normalize=normalize,  # Pass normalize parameter
                    **kwargs  # kwargs no longer contains compress or format
                )
                
                # Track results
                map_time = time.time() - map_start_time
                results[map_type] = {
                    "success": success,
                    "path": output_path if success else None,
                    "time": map_time
                }
                
                progress.advance(main_task)
        
        # Display results table
        console.print("\n[bold cyan]Export Results[/]")
        display_export_results(results, time.time() - start_time)
        
        return True
    except Exception as e:
        print_error(f"Map export failed: {e}")
        return False

def display_export_results(results: dict, total_time: float):
    """Display a table of export results."""
    table = Table(title="Export Results")
    table.add_column("Map Type", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Time", justify="right", style="green")
    table.add_column("Output File", style="blue")
    
    for map_type, result in results.items():
        status = "✅" if result["success"] else "❌"
        time_str = f"{result['time']:.2f}s"
        path_str = str(result["path"]) if result["path"] else "Failed"
        table.add_row(map_type, status, time_str, path_str)
    
    console.print(table)
    console.print(f"\nTotal processing time: [green]{total_time:.2f}s[/]")
