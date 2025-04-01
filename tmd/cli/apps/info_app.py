#!/usr/bin/env python3
"""
Information commands for TMD CLI.
"""

from pathlib import Path
import typer
from rich.panel import Panel

from tmd import TMD
from tmd.cli.core.ui import console, print_error, display_metadata
from tmd.utils.utils import TMDUtils

def info_command(
    tmd_file: Path = typer.Argument(..., help="Path to TMD file", exists=True),
    show_sample: bool = typer.Option(False, help="Show a sample of height map values")
):
    """Display detailed information about a TMD file."""
    try:
        # Display detailed file info
        display_file_info(tmd_file, show_sample)
        return 0
    except Exception as e:
        print_error(f"Failed to display file info: {e}")
        return 1

def version_command():
    """Display TMD CLI version information."""
    try:
        from tmd.cli import __version__ as cli_version
        from tmd import __version__ as tmd_version
    except ImportError:
        cli_version = "Unknown"
        tmd_version = "Unknown"
    
    console.print(Panel.fit(
        f"[bold]TMD Command-Line Interface[/bold]\n\n"
        f"CLI Version: {cli_version}\n"
        f"TMD Core Version: {tmd_version}\n"
    ))

def check_command():
    """Perform system checks to confirm TMD CLI is working properly."""
    # Check TMD core
    try:
        import numpy as np
        console.print("[green]✓[/green] NumPy is available")
    except ImportError:
        console.print("[red]✗[/red] NumPy is not available")

    # Check TMD package
    try:
        from tmd import TMD
        console.print("[green]✓[/green] TMD package is available")
        # Try creating a TMD object
        empty_tmd = TMD(np.zeros((10, 10)), {"comment": "Test data"})
        console.print("[green]✓[/green] TMD object creation is working")
    except ImportError:
        console.print("[red]✗[/red] TMD package is not available")
    except Exception as e:
        console.print(f"[red]✗[/red] TMD object creation failed: {e}")
    
    # Check cache
    try:
        from tmd.cli.utils.caching import get_cache_stats
        cache_stats = get_cache_stats()
        console.print("[green]✓[/green] Cache system is working")
        console.print(f"   Cache location: {cache_stats['cache_dir']}")
        console.print(f"   Cache entries: {cache_stats['entry_count']}")
    except (NameError, ImportError):
        console.print("[yellow]![/yellow] Cache module is not available")
    except Exception as e:
        console.print(f"[red]✗[/red] Cache system check failed: {e}")
    
    # Check visualization capabilities
    try:
        from tmd.cli.utils.visualization import check_available_visualization_backends
        backends = check_available_visualization_backends()
        console.print("[green]✓[/green] Visualization module is available")
        console.print(f"   Available backends: {', '.join(backends)}")
    except (NameError, ImportError):
        console.print("[yellow]![/yellow] Visualization module is not available")
    except Exception as e:
        console.print(f"[red]✗[/red] Visualization check failed: {e}")
    
    return 0

def display_file_info(
    tmd_file: Path,
    show_sample: bool = False
) -> bool:
    """
    Display information about a TMD file including file size.
    
    Args:
        tmd_file: Path to TMD file
        show_sample: Whether to display a sample of height values
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load the TMD file
        with console.status(f"Loading {tmd_file.name}..."):
            try:
                # Ensure tmd_file is a Path object when getting file size
                if isinstance(tmd_file, str):
                    tmd_file = Path(tmd_file)
                
                # Use the path string for TMD constructor
                tmd_obj = TMD(str(tmd_file))
                
                # Access height_map and metadata properly (could be properties or methods)
                height_map = tmd_obj.height_map if not callable(tmd_obj.height_map) else tmd_obj.height_map()
                metadata = tmd_obj.metadata if not callable(tmd_obj.metadata) else tmd_obj.metadata()
            except (NameError, ImportError):
                metadata, height_map = TMDUtils.process_tmd_file(tmd_file)
            
        # Get file size
        file_size = tmd_file.stat().st_size
        
        # Display information
        console.print(Panel.fit(
            f"[bold]TMD File:[/bold] {tmd_file}\n"
            f"File size: {file_size / 1024:.1f} KB\n"
            f"Dimensions: {height_map.shape[1]}×{height_map.shape[0]}\n"
            f"Height Range: {height_map.min():.6f} to {height_map.max():.6f}\n"
            f"Memory usage: {height_map.nbytes / 1024:.1f} KB"
        ))
        
        # Display metadata
        console.print("\n[bold]Metadata:[/bold]")
        for key, value in sorted(metadata.items()):
            if key != "file_path":  # Skip redundant file path
                console.print(f"  {key}: {value}")
        
        if show_sample:
            # Show a small sample of the height map
            console.print("\n[bold]Height Map Sample[/bold] (first few rows and columns):")
            sample_rows = min(5, height_map.shape[0])
            sample_cols = min(8, height_map.shape[1])
            console.print(height_map[:sample_rows, :sample_cols])
                
        return True
    except Exception as e:
        print_error(f"Error displaying file info: {e}")
        return False
