#!/usr/bin/env python3
"""
Progress bar utilities for TMD CLI.

This module provides helpers for creating progress bars and spinners
using both rich and tqdm libraries, with seamless fallbacks.
"""

import sys
import time
from typing import Optional, Any, List, Dict, Union, Callable, Iterator, TypeVar, Generic
from pathlib import Path
import logging

# Set up logger
logger = logging.getLogger(__name__)

# Import Rich components - assumed to always be available 
from rich.progress import (
    Progress, 
    SpinnerColumn, 
    BarColumn, 
    TextColumn, 
    TimeElapsedColumn, 
    TimeRemainingColumn,
    ProgressColumn
)
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rich_print

# Initialize console
console = Console()

# Import tqdm (now always available)
from tqdm import tqdm

# Define a type variable for generic iterables
T = TypeVar('T')

def create_progress_bar(total: int, description: str, unit: str = "it",
                       use_rich: bool = True) -> Any:
    """
    Create a progress bar using rich or tqdm.
    
    Args:
        total: Total number of items
        description: Description for the progress bar
        unit: Unit to display (e.g., "it", "files", "MB")
        use_rich: Whether to use rich (if available) or tqdm
        
    Returns:
        Progress bar object
    """
    if use_rich:
        # Create a Rich progress bar
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}[/bold blue]"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )
        progress.start()
        task_id = progress.add_task(description, total=total)
        return {"progress": progress, "task_id": task_id, "type": "rich"}
    else:
        # Create a tqdm progress bar
        return {"progress": tqdm(total=total, desc=description, unit=unit), "type": "tqdm"}

def update_progress(progress_bar: Dict[str, Any], n: int = 1, 
                   description: Optional[str] = None) -> None:
    """
    Update a progress bar.
    
    Args:
        progress_bar: Progress bar object from create_progress_bar
        n: Amount to increment
        description: New description (optional)
    """
    if progress_bar["type"] == "rich":
        if description:
            progress_bar["progress"].update(progress_bar["task_id"], 
                                          description=description, 
                                          advance=n)
        else:
            progress_bar["progress"].update(progress_bar["task_id"], advance=n)
    elif progress_bar["type"] == "tqdm":
        progress_bar["progress"].update(n)
        if description:
            progress_bar["progress"].set_description(description)

def close_progress(progress_bar: Dict[str, Any]) -> None:
    """
    Close a progress bar.
    
    Args:
        progress_bar: Progress bar object from create_progress_bar
    """
    if progress_bar["type"] == "rich":
        progress_bar["progress"].stop()
    elif progress_bar["type"] == "tqdm":
        progress_bar["progress"].close()

def progress_iterator(iterable: List[T], description: str, unit: str = "it",
                     use_rich: bool = True) -> Iterator[T]:
    """
    Create an iterator with progress tracking.
    
    Args:
        iterable: List or other iterable to process
        description: Description for the progress bar
        unit: Unit to display (e.g., "it", "files", "MB")
        use_rich: Whether to use rich (if available) or tqdm
        
    Yields:
        Items from the iterable
    """
    total = len(iterable)
    
    if use_rich:
        # Use Rich progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}[/bold blue]"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task(description, total=total)
            for item in iterable:
                yield item
                progress.update(task, advance=1)
    else:
        # Use tqdm progress
        for item in tqdm(iterable, desc=description, unit=unit):
            yield item

def spinner_context(description: str, use_rich: bool = True):
    """
    Create a context manager for a spinner/status indicator.
    
    Args:
        description: Description to display
        use_rich: Whether to use rich (if available)
        
    Returns:
        Context manager for spinner
    """
    if use_rich:
        return console.status(description)
    else:
        # Create a simple spinner for terminal without rich
        class SimpleSpinner:
            def __init__(self, desc):
                self.desc = desc
                self.chars = "|/-\\"
                self.current = 0
                self.running = False
                self.last_update = 0
                
            def __enter__(self):
                self.running = True
                print(f"{self.desc}... ", end="", flush=True)
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                self.running = False
                print("Done")
                return False  # Don't suppress exceptions
                
            def update(self, new_desc=None):
                if new_desc:
                    self.desc = new_desc
                    print(f"\r{self.desc}... ", end="", flush=True)
                    
        return SimpleSpinner(description)

def process_with_progress(items: List[T], process_func: Callable[[T], Any],
                         description: str, 
                         error_handler: Optional[Callable[[T, Exception], None]] = None) -> Dict:
    """
    Process items with a progress bar and error handling.
    
    Args:
        items: Items to process
        process_func: Function to process each item
        description: Description for the progress bar
        error_handler: Function to handle errors (optional)
        
    Returns:
        Results dictionary
    """
    results = {
        "total": len(items),
        "success": 0,
        "failed": 0,
        "results": []
    }
    
    if not items:
        rich_print("[yellow]No items to process[/yellow]")
        return results
    
    progress = create_progress_bar(len(items), description)
    
    for item in items:
        try:
            # Generate a descriptive message if the item has a name attribute
            if hasattr(item, 'name'):
                update_progress(progress, description=f"Processing {item.name}")
            
            # Process the item
            result = process_func(item)
            
            # Record success
            results["success"] += 1
            results["results"].append({
                "item": item,
                "result": result,
                "success": True
            })
            
        except Exception as e:
            # Handle error
            results["failed"] += 1
            error_info = {
                "item": item,
                "error": str(e),
                "success": False
            }
            results["results"].append(error_info)
            
            # Call error handler if provided
            if error_handler:
                try:
                    error_handler(item, e)
                except Exception as handler_error:
                    logger.error(f"Error in error handler: {handler_error}")
        
        # Update progress
        update_progress(progress, n=1)
    
    # Close progress bar
    close_progress(progress)
    
    # Display summary
    rich_print(f"[bold]Processing complete:[/bold]")
    rich_print(f"  [green]Success:[/green] {results['success']}")
    rich_print(f"  [red]Failed:[/red] {results['failed']}")
    rich_print(f"  [blue]Total:[/blue] {results['total']}")
    
    return results

def file_progress_bar(file_size: int, description: str = "Downloading",
                    unit: str = "B", unit_scale: bool = True,
                    unit_divisor: int = 1024) -> Dict[str, Any]:
    """
    Create a progress bar for file operations.
    
    Args:
        file_size: Size of the file in bytes
        description: Description for the progress bar
        unit: Unit to display
        unit_scale: Whether to scale units (e.g., KB, MB)
        unit_divisor: Divisor for unit scaling
        
    Returns:
        Progress bar object
    """
    # Rich progress with file size formatting
    def _format_size(size: float) -> str:
        if size < 1024:
            return f"{size:.0f}B"
        elif size < 1024**2:
            return f"{size/1024:.1f}KB"
        elif size < 1024**3:
            return f"{size/1024**2:.1f}MB"
        else:
            return f"{size/1024**3:.1f}GB"
            
    class FileSizeColumn(ProgressColumn):
        def render(self, task):
            return f"{_format_size(task.completed)}/{_format_size(task.total)}"
            
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}[/bold blue]"),
        BarColumn(),
        FileSizeColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
    )
    progress.start()
    task_id = progress.add_task(description, total=file_size)
    return {"progress": progress, "task_id": task_id, "type": "rich"}
