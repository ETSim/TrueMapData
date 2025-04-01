#!/usr/bin/env python3
"""
UI components for TMD CLI tools.

This module provides functions for creating consistent user interfaces
across different TMD command-line tools.
"""

import logging
import sys
from typing import Optional, List, Dict, Any, Tuple, Union

logger = logging.getLogger(__name__)

# Import rich components
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich import print as rprint

console = Console()

def print_rich_table(data: List[Dict[str, Any]], title: str, 
                    columns: Optional[List[Tuple[str, str]]] = None) -> None:
    """
    Print data as a rich table.
    
    Args:
        data: List of dictionaries with row data.
        title: Table title.
        columns: Optional list of (column_name, style) tuples.
    """
    # Create rich table
    table = Table(title=title)
    
    # Add columns
    if not columns:
        # Use first row to determine columns
        if data:
            columns = [(key, "cyan") for key in data[0].keys()]
    
    for name, style in columns:
        table.add_column(name, style=style)
    
    # Add rows
    for row in data:
        table.add_row(*[str(row.get(col[0], "")) for col in columns])
    
    console.print(table)

def display_metadata(metadata: Dict[str, Any]) -> None:
    """
    Display TMD file metadata in a nice table.
    
    Args:
        metadata: Dictionary containing metadata.
    """
    table = Table(title="TMD File Metadata")
    table.add_column("Property", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")
    
    # Sort keys for consistent display
    for key in sorted(metadata.keys()):
        value = metadata[key]
        # Format certain values nicely
        if isinstance(value, float):
            formatted_value = f"{value:.6f}"
        elif isinstance(value, (list, tuple)) and all(isinstance(x, (int, float)) for x in value):
            formatted_value = ", ".join(f"{x:.4f}" for x in value)
        else:
            formatted_value = str(value)
            
        table.add_row(str(key), formatted_value)
    
    console.print(table)

def format_height_map_summary(height_map) -> str:
    """
    Format a summary of the height map.
    
    Args:
        height_map: NumPy array with height data.
        
    Returns:
        Formatted string with height map summary.
    """
    if height_map is None:
        return "Height map not available"
        
    try:
        import numpy as np
        return (
            f"Dimensions: {height_map.shape[1]}×{height_map.shape[0]}\n"
            f"Height Range: {height_map.min():.6f} to {height_map.max():.6f}\n"
            f"Mean Height: {np.mean(height_map):.6f}"
        )
    except Exception as e:
        logger.error(f"Error formatting height map summary: {e}")
        return "Error computing height map statistics"

class ProgressContext:
    """Context manager for progress reporting."""
    
    def __init__(self, description: str = "Processing...", total: int = None):
        """
        Initialize the progress context.
        
        Args:
            description: Progress description
            total: Total number of steps (optional)
        """
        self.description = description
        self.total = total
        self.progress = None
        self.task = None
        
    def __enter__(self):
        """Enter the progress context."""
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            console=console
        )
        self.progress.start()
        if self.total:
            self.task = self.progress.add_task(self.description, total=self.total)
        else:
            self.task = self.progress.add_task(self.description)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the progress context."""
        if self.progress:
            self.progress.stop()
    
    def update(self, advance: int = 0, description: str = None):
        """
        Update the progress.
        
        Args:
            advance: Number of steps to advance
            description: New description (optional)
        """
        if description:
            self.progress.update(self.task, description=description)
        if advance:
            self.progress.advance(self.task, advance)

def print_warning(message: str) -> None:
    """
    Print a warning message.
    
    Args:
        message: Warning message text
    """
    rprint(f"[bold yellow]Warning:[/bold yellow] {message}")

def print_error(message: str) -> None:
    """
    Print an error message.
    
    Args:
        message: Error message text
    """
    rprint(f"[bold red]Error:[/bold red] {message}")

def print_success(message: str) -> None:
    """
    Print a success message.
    
    Args:
        message: Success message text
    """
    rprint(f"[bold green]Success:[/bold green] {message}")