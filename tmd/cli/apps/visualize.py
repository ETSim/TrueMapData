#!/usr/bin/env python3
"""
TMD Visualization CLI

An improved command-line interface for visualizing TMD files using 
different visualization backends (Matplotlib, Plotly, Seaborn, Polyscope).

This script provides a user-friendly interface for exploring TMD files with
customizable visualization options.

Usage:
    python tmd_visualize.py --help
    python tmd_visualize.py basic Dime.tmd
    python tmd_visualize.py 3d Dime.tmd --plotter plotly --output my_3d_viz.html
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union

# Terminal interface libraries
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich import print as rprint

# Import NumPy
import numpy as np

# Import TMD core
from tmd import TMD

# Import caching utilities
from tmd.cli.utils.caching import get_cache_stats, clear_cache

# Import TMD utilities
from tmd.cli.common import (
    check_available_plotters,
    print_missing_plotter_instructions,
    display_metadata,
    format_height_map_summary,
    load_tmd_file,
    select_plotter,
    get_output_filename,
    create_output_dir,
    check_dependencies_and_install,
    VIZ_OPTIONS,
    auto_open_file,
    create_visualization,
    create_comparison_visualization,
    load_config,
    save_config
)

# Initialize globals
console = Console()
app = typer.Typer(help="TMD Visualization Tool")

# Get configuration
config = load_config()
DEFAULT_PLOTTER = config.get("default_plotter", "matplotlib")
DEFAULT_COLORMAP = config.get("default_colormap", "viridis")
OUTPUT_DIR = Path(config.get("output_dir", "tmd_viz_output"))

# Initialize visualization utilities
from tmd.plotters import (
    TMDPlotterFactory, 
    TMDSequencePlotterFactory,
    get_registered_plotters, 
    get_best_plotter
)

# Register all available plotters
available_plotters = get_registered_plotters()

# Extra visualization utilities
try:
    from tmd.plotters import (
        HeightMapAnalyzer,
        ColorMapRegistry,
        TMDVisualizationUtils
    )
except ImportError:
    pass

@app.callback()
def callback():
    """ 
    TMD Visualization Tool - Create beautiful visualizations from TMD files
    """
    pass

@app.command("info")
def display_info(
    tmd_file: Path = typer.Argument(..., help="Path to TMD file", exists=True),
):
    """Display information about a TMD file."""
    data = load_tmd_file(tmd_file, with_console_status=True)
    if not data:
        return
    
    # Fix: Access metadata and height_map as properties, not methods
    metadata = data.metadata
    height_map = data.height_map
    
    summary = format_height_map_summary(height_map)
    
    console.print(Panel.fit(
        f"[bold]TMD File:[/bold] {tmd_file}\n"
        f"{summary}"
    ))
    
    display_metadata(metadata)
    
    # Display a small sample of the height map
    sample_rows = min(10, height_map.shape[0])
    sample_cols = min(10, height_map.shape[1])
    sample = height_map[:sample_rows, :sample_cols]
    
    console.print("\n[bold]Height Map Sample[/bold] (first few rows and columns):")
    console.print(sample)

@app.command("plotters")
def check_plotters():
    """Check which visualization backends are available."""
    rprint("[bold blue]Checking available visualization backends...[/bold blue]")
    plotters = check_available_plotters()
    
    # Provide installation instructions if some plotters are missing
    missing_plotters = [name for name, available in plotters.items() if not available]
    if missing_plotters:
        print_missing_plotter_instructions(missing_plotters)

# Modified function to better handle plotter selection
def get_available_plotter(requested_plotter: str, viz_type: str) -> str:
    """Get an available plotter that supports the requested visualization type."""
    # First check which plotters are available
    available_plotters = check_available_plotters()
    
    # Filter to those that are actually available (value is True)
    truly_available = [p for p, status in available_plotters.items() if status]
    
    if not truly_available:
        console.print("[bold red]Error:[/bold red] No visualization backends available.")
        console.print("Please install at least one of: matplotlib, plotly, seaborn, or polyscope")
        return requested_plotter  # Return the original to let caller handle error
    
    # Check if the requested plotter is available
    if requested_plotter.lower() in truly_available:
        # Check if it supports the viz_type
        if viz_type in VIZ_OPTIONS and requested_plotter.lower() in VIZ_OPTIONS[viz_type].get("supported_plotters", []):
            return requested_plotter.lower()
        else:
            console.print(f"[bold yellow]Warning:[/bold yellow] {requested_plotter} may not fully support {viz_type} visualizations.")
            
            # Try to find an alternative that supports this viz_type
            for p in VIZ_OPTIONS.get(viz_type, {}).get("supported_plotters", []):
                if p in truly_available:
                    console.print(f"[bold yellow]Falling back to {p}.[/bold yellow]")
                    return p
            
            # If we get here, use the requested one anyway
            return requested_plotter.lower()
    
    # Requested plotter not available, find an alternative
    console.print(f"[bold yellow]Warning:[/bold yellow] {requested_plotter} not available. Using {truly_available[0]} instead.")
    return truly_available[0]

@app.command("basic")
def basic_visualization(
    tmd_file: Path = typer.Argument(..., help="Path to TMD file", exists=True),
    plotter: str = typer.Option(DEFAULT_PLOTTER, help="Visualization backend to use"),
    output: Optional[Path] = typer.Option(None, help="Output filename"),
    auto_open: bool = typer.Option(True, help="Automatically open the output file")
):
    """Create a basic 2D visualization of a TMD file."""
    data = load_tmd_file(tmd_file, with_console_status=True)
    if not data:
        return
    
    # Use improved plotter selection
    selected_plotter = get_available_plotter(plotter, "2d")
    output_file = get_output_filename(tmd_file, selected_plotter, "2d", output)
    
    # Try to create the visualization with better error handling
    try:
        # Create the visualization
        success = create_visualization(
            data, 
            "2d", 
            selected_plotter, 
            output_file, 
            title=f"{tmd_file.name} - 2D Visualization"
        )
        
        if success and auto_open:
            auto_open_file(output_file)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] Failed to create visualization: {str(e)}")
        console.print("[yellow]Try a different plotter or check your TMD file.[/yellow]")

@app.command("3d")
def three_d_visualization(
    tmd_file: Path = typer.Argument(..., help="Path to TMD file", exists=True),
    plotter: str = typer.Option(DEFAULT_PLOTTER, help="Visualization backend to use"),
    output: Optional[Path] = typer.Option(None, help="Output filename"),
    z_scale: float = typer.Option(1.0, help="Z-axis scaling factor"),
    auto_open: bool = typer.Option(True, help="Automatically open the output file")
):
    """Create a 3D visualization of a TMD file."""
    data = load_tmd_file(tmd_file, with_console_status=True)
    if not data:
        return
    
    # Use improved plotter selection
    selected_plotter = get_available_plotter(plotter, "3d")
    output_file = get_output_filename(tmd_file, selected_plotter, "3d", output)
    
    # Try to create the visualization with better error handling
    try:
        # Create the visualization
        success = create_visualization(
            data, 
            "3d", 
            selected_plotter, 
            output_file, 
            title=f"{tmd_file.name} - 3D Visualization",
            z_scale=z_scale
        )
        
        if success and auto_open:
            auto_open_file(output_file)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] Failed to create 3D visualization: {str(e)}")
        if "plotly" in selected_plotter.lower():
            console.print("[yellow]Ensure plotly is correctly installed: pip install plotly[/yellow]")
        elif "polyscope" in selected_plotter.lower():
            console.print("[yellow]Ensure polyscope is correctly installed: pip install polyscope[/yellow]")
        else:
            console.print("[yellow]Try a different plotter or check your TMD file.[/yellow]")

@app.command("profile")
def profile_visualization(
    tmd_file: Path = typer.Argument(..., help="Path to TMD file", exists=True),
    plotter: str = typer.Option(DEFAULT_PLOTTER, help="Visualization backend to use"),
    row: Optional[int] = typer.Option(None, help="Row index for profile (default: middle)"),
    output: Optional[Path] = typer.Option(None, help="Output filename"),
    auto_open: bool = typer.Option(True, help="Automatically open the output file")
):
    """Create a profile visualization of a specific row in the TMD file."""
    data = load_tmd_file(tmd_file, with_console_status=True)
    if not data:
        return
    
    # Use improved plotter selection
    selected_plotter = get_available_plotter(plotter, "profile")
    
    # Prepare the output filename
    row_suffix = f"_row{row}" if row is not None else ""
    output_file = get_output_filename(
        tmd_file, 
        selected_plotter, 
        f"profile{row_suffix}", 
        output
    )
    
    # Try to create the visualization with better error handling
    try:
        # Create the visualization
        success = create_visualization(
            data, 
            "profile", 
            selected_plotter, 
            output_file, 
            profile_row=row,
            title=f"{tmd_file.name} - Profile Visualization" + 
                (f" (Row {row})" if row is not None else "")
        )
        
        if success and auto_open:
            auto_open_file(output_file)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] Failed to create profile visualization: {str(e)}")
        console.print("[yellow]Try a different plotter or check your TMD file.[/yellow]")

if __name__ == "__main__":
    # Check if TMD core is available
    if not TMD:
        console.print("[bold red]TMD core package is required but not found.[/bold red]")
        console.print("Please make sure the TMD package is installed.")
        sys.exit(1)
    
    # Check dependencies
    check_dependencies_and_install()
    
    # Run the app
    app()