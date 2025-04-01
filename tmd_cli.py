#!/usr/bin/env python3
"""
TMD Command-Line Interface

A comprehensive command-line interface for working with TMD (Topographic Mesh Data) files.
This tool provides functionality for visualization, compression, analysis, file conversions,
and cache management.

Usage:
    python tmd_cli.py --help
    python tmd_cli.py info Dime.tmd
    python tmd_cli.py compress downsample Dime.tmd --scale 0.5
    python tmd_cli.py compress quantize Dime.tmd --levels 256
    
Visualization examples:
    python tmd_cli.py visualize basic Dime.tmd --colormap viridis
    python tmd_cli.py visualize 3d Dime.tmd --z-scale 2.0 --plotter plotly
    
Cache management:
    python tmd_cli.py cache info
    python tmd_cli.py cache clear
"""

import sys
import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Tuple
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# Initialize console for rich output
console = Console()

# Import core functionality from TMD CLI
from tmd.cli.core.config import load_config, save_config, get_config_value
from tmd.cli.core.io import load_tmd_file, auto_open_file, create_output_dir, get_output_filename
from tmd.cli.core.ui import print_warning, print_error, print_success, display_metadata
from tmd.cli.core import check_dependencies

# Import the consolidated compression module
from tmd.cli.compression import (
    compress_tmd_file, 
    display_compression_summary, 
    compress_height_map
)

# Import TMD utilities
from tmd.utils.files import TMDFileUtilities
from tmd.utils.utils import TMDUtils

# Import visualization utilities
try:
    from tmd.cli.utils.visualization import create_visualization, check_available_visualization_backends
except ImportError:
    pass

# Import TMD dependencies
import numpy as np
try:
    from tmd import TMD
except ImportError:
    pass

# Import caching utilities
try:
    from tmd.cli.utils.caching import get_cache_stats, clear_cache
except ImportError:
    pass

# Create main app
app = typer.Typer(
    help="TMD Command Line Interface - Tools for working with Topographic Mesh Data files",
)

# Create subcommands
compress_app = typer.Typer(help="Compression tools for TMD files")
app.add_typer(compress_app, name="compress")

config_app = typer.Typer(help="Manage TMD configuration")
app.add_typer(config_app, name="config")

visualize_app = typer.Typer(help="Visualization tools for TMD files")
app.add_typer(visualize_app, name="visualize")

cache_app = typer.Typer(help="Manage TMD file cache")
app.add_typer(cache_app, name="cache")

@app.callback()
def callback():
    """
    TMD Command-Line Tools - Work with Topographic Mesh Data files
    
    A suite of tools for analyzing, visualizing, and processing TMD files.
    """
    # Check dependencies
    check_dependencies(auto_install=False, exit_on_failure=True)

@app.command("info")
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
                tmd_obj = TMD(str(tmd_file))
                height_map = tmd_obj.height_map()
                metadata = tmd_obj.metadata()
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

@app.command("version")
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

@app.command("check")
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
        backends = check_available_visualization_backends()
        console.print("[green]✓[/green] Visualization module is available")
        console.print(f"   Available backends: {', '.join(backends)}")
    except (NameError, ImportError):
        console.print("[yellow]![/yellow] Visualization module is not available")
    except Exception as e:
        console.print(f"[red]✗[/red] Visualization check failed: {e}")
    
    return 0

# Compression commands
@compress_app.command("downsample")
def compress_downsample(
    tmd_file: Path = typer.Argument(..., help="Path to TMD file", exists=True),
    scale: float = typer.Option(0.5, help="Scale factor (0-1)"),
    method: str = typer.Option("bilinear", help="Interpolation method: nearest, bilinear, bicubic"),
    output: Optional[Path] = typer.Option(None, help="Output filename"),
    version: int = typer.Option(2, help="TMD format version (1 or 2)"),
    auto_open: bool = typer.Option(False, help="Automatically open the output file")
):
    """Downsample a TMD file to reduce resolution."""
    # Use the consolidated compression utility
    summary = compress_tmd_file(
        tmd_file=tmd_file,
        output=output,
        mode="downsample",
        scale=scale,
        method=method,
        version=version
    )
    
    # Display results
    display_compression_summary(summary)
    
    # Open file if requested and successful
    if auto_open and summary["success"]:
        auto_open_file(summary["output_file"])

@compress_app.command("quantize")
def compress_quantize(
    tmd_file: Path = typer.Argument(..., help="Path to TMD file", exists=True),
    levels: int = typer.Option(256, help="Number of height levels"),
    output: Optional[Path] = typer.Option(None, help="Output filename"),
    version: int = typer.Option(2, help="TMD format version (1 or 2)"),
    auto_open: bool = typer.Option(False, help="Automatically open the output file")
):
    """Quantize height values in a TMD file to reduce file size."""
    # Use the consolidated compression utility
    summary = compress_tmd_file(
        tmd_file=tmd_file,
        output=output,
        mode="quantize",
        levels=levels,
        version=version
    )
    
    # Display results
    display_compression_summary(summary)
    
    # Open file if requested and successful
    if auto_open and summary["success"]:
        auto_open_file(summary["output_file"])

@compress_app.command("combined")
def compress_combined(
    tmd_file: Path = typer.Argument(..., help="Path to TMD file", exists=True),
    scale: float = typer.Option(0.5, help="Scale factor (0-1)"),
    levels: int = typer.Option(256, help="Number of height levels"),
    method: str = typer.Option("bilinear", help="Interpolation method"),
    output: Optional[Path] = typer.Option(None, help="Output filename"),
    version: int = typer.Option(2, help="TMD format version (1 or 2)"),
    auto_open: bool = typer.Option(False, help="Automatically open the output file")
):
    """Apply both downsampling and quantization to a TMD file."""
    # Use the consolidated compression utility
    summary = compress_tmd_file(
        tmd_file=tmd_file,
        output=output,
        mode="both",
        scale=scale,
        levels=levels,
        method=method,
        version=version
    )
    
    # Display results
    display_compression_summary(summary)
    
    # Open file if requested and successful
    if auto_open and summary["success"]:
        auto_open_file(summary["output_file"])

@compress_app.command("batch")
def compress_batch(
    input_dir: Path = typer.Argument(..., help="Directory containing TMD files", exists=True),
    mode: str = typer.Option("downsample", help="Compression mode: downsample, quantize, both"),
    scale: float = typer.Option(0.5, help="Scale factor for downsampling (0-1)"),
    levels: int = typer.Option(256, help="Number of height levels for quantization"),
    method: str = typer.Option("bilinear", help="Interpolation method"),
    recursive: bool = typer.Option(False, help="Recursively search for TMD files"),
    version: int = typer.Option(2, help="TMD format version (1 or 2)")
):
    """Batch compress multiple TMD files in a directory."""
    # Find TMD files
    pattern = "**/*.tmd" if recursive else "*.tmd"
    files = list(input_dir.glob(pattern))
    
    if not files:
        print_warning(f"No TMD files found in {input_dir}")
        return 1
    
    print_success(f"Found {len(files)} TMD files to process")
    
    # Create output directory
    output_dir = TMDFileUtilities.ensure_directory(Path("batch_compressed"))
    processed = 0
    failed = 0
    
    # Process files with progress indicator
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task(f"Processing {len(files)} files...", total=len(files))
        
        for file_path in files:
            progress.update(task, description=f"Processing {file_path.name}...")
            
            try:
                # Generate output filename
                if mode == "downsample":
                    suffix = f"_ds{int(scale*100)}"
                elif mode == "quantize":
                    suffix = f"_q{levels}"
                else:
                    suffix = f"_comp"
                    
                output_path = output_dir / f"{file_path.stem}{suffix}.tmd"
                
                # Process the file
                result = compress_tmd_file(
                    file_path,
                    output=output_path,
                    mode=mode,
                    scale=scale,
                    levels=levels,
                    method=method,
                    version=version
                )
                
                if result["success"]:
                    processed += 1
                else:
                    failed += 1
                    
            except Exception as e:
                print_warning(f"Error processing {file_path.name}: {str(e)}")
                failed += 1
                
            progress.advance(task)
    
    # Display summary
    if processed > 0:
        console.print(f"[bold green]Batch processing summary:[/bold green]")
        console.print(f"Files processed: {processed} of {len(files)}")
        if failed > 0:
            console.print(f"Failed: {failed}")
        console.print(f"Output directory: {output_dir}")
    else:
        print_error("Batch processing failed: no files were processed successfully")
        return 1
        
    return 0

# Configuration commands
@config_app.command("show")
def config_show():
    """Display current configuration settings."""
    config = load_config()
    
    console.print(Panel.fit("[bold]TMD Configuration[/bold]"))
    for key, value in sorted(config.items()):
        console.print(f"{key}: {value}")

@config_app.command("set")
def config_set(
    key: str = typer.Argument(..., help="Configuration key"),
    value: str = typer.Argument(..., help="Configuration value")
):
    """Set a configuration value."""
    # Auto-convert value types
    if value.lower() == "true":
        typed_value = True
    elif value.lower() == "false":
        typed_value = False
    elif value.isdigit():
        typed_value = int(value)
    elif "." in value and all(part.isdigit() for part in value.split(".", 1)):
        typed_value = float(value)
    else:
        typed_value = value
    
    # Load config, update and save
    config = load_config()
    config[key] = typed_value
    save_config(config)
    print_success(f"Configuration updated: {key} = {typed_value}")

@config_app.command("reset")
def config_reset():
    """Reset configuration to default values."""
    default_config = {
        "default_colormap": "viridis",
        "auto_cache": True,
        "cache_ttl_days": 7,
        "default_plotter": "matplotlib",
        "default_compression_level": 9,
        "use_rich_formatting": True
    }
    
    save_config(default_config)
    print_success("Configuration reset to default values")

# Visualization commands
@visualize_app.command("basic")
def visualize_basic(
    tmd_file: Path = typer.Argument(..., help="Path to TMD file", exists=True),
    plotter: str = typer.Option("matplotlib", help="Visualization backend (matplotlib, plotly)"),
    colormap: str = typer.Option("viridis", help="Colormap name"),
    output: Optional[Path] = typer.Option(None, help="Output filename"),
    auto_open: bool = typer.Option(False, help="Automatically open the output file"),
    cache: bool = typer.Option(True, help="Use cache if available")
):
    """Create a basic 2D visualization of a TMD file."""
    try:
        success = create_visualization(
            tmd_file_or_data=tmd_file,
            mode="2d",
            plotter=plotter,
            colormap=colormap,
            output=output,
            use_cache=cache
        )
        
        if success and auto_open and output:
            auto_open_file(output)
            
        return 0 if success else 1
    except (NameError, ImportError):
        print_error("Visualization functionality is not available")
        return 1

@visualize_app.command("3d")
def visualize_3d(
    tmd_file: Path = typer.Argument(..., help="Path to TMD file", exists=True),
    plotter: str = typer.Option("matplotlib", help="Visualization backend (matplotlib, plotly)"),
    colormap: str = typer.Option("viridis", help="Colormap name"),
    output: Optional[Path] = typer.Option(None, help="Output filename"),
    z_scale: float = typer.Option(1.0, help="Z-axis scaling factor"),
    auto_open: bool = typer.Option(False, help="Automatically open the output file"),
    cache: bool = typer.Option(True, help="Use cache if available")
):
    """Create a 3D surface visualization of a TMD file."""
    try:
        success = create_visualization(
            tmd_file_or_data=tmd_file,
            mode="3d",
            plotter=plotter,
            colormap=colormap,
            output=output,
            z_scale=z_scale,
            use_cache=cache
        )
        
        if success and auto_open and output:
            auto_open_file(output)
            
        return 0 if success else 1
    except (NameError, ImportError):
        print_error("Visualization functionality is not available")
        return 1

# Cache commands
@cache_app.command("info")
def cache_info_command():
    """Display information about the TMD cache."""
    try:
        stats = get_cache_stats()
        
        console.print(Panel.fit(
            f"[bold]TMD Cache Information[/bold]\n\n"
            f"Location: {stats['cache_dir']}\n"
            f"Total entries: {stats['entry_count']}\n"
            f"Expired entries: {stats['expired_count']}\n"
            f"Total size: {stats['total_size_mb']:.2f} MB\n"
        ))
    except (NameError, ImportError):
        print_error("Cache functionality is not available")
        return 1

@cache_app.command("clear")
def cache_clear_command(
    expired_only: bool = typer.Option(True, help="Clear only expired entries")
):
    """Clear the TMD cache."""
    try:
        with console.status("Clearing cache..."):
            count = clear_cache(expired_only=expired_only)
        
        if expired_only:
            print_success(f"Cleared {count} expired entries from cache")
        else:
            print_success(f"Cleared entire cache ({count} entries)")
    except (NameError, ImportError):
        print_error("Cache functionality is not available")
        return 1

@cache_app.command("clear-all")
def cache_clear_all_command():
    """Clear the entire TMD cache."""
    try:
        with console.status("Clearing entire cache..."):
            count = clear_cache(expired_only=False)
        
        print_success(f"Cleared entire cache ({count} entries)")
    except (NameError, ImportError):
        print_error("Cache functionality is not available")
        return 1

@app.command("export")
def export_command(
    tmd_file: Path = typer.Argument(..., help="Path to TMD file", exists=True),
    format: str = typer.Option("npz", help="Export format (npz, zip, npy, pickle)"),
    output: Optional[Path] = typer.Option(None, help="Output filename"),
    compression: int = typer.Option(9, help="Compression level (0-9, for ZIP format)")
):
    """Export a TMD file to another format."""
    try:
        from tmd.compression.factory import TMDDataIOFactory
        
        # Load the TMD data
        with console.status(f"Loading {tmd_file.name}..."):
            try:
                tmd_obj = TMD(str(tmd_file))
                height_map = tmd_obj.height_map()
                metadata = tmd_obj.metadata()
            except (NameError, ImportError):
                metadata, height_map = TMDUtils.process_tmd_file(tmd_file)
        
        # Prepare data for export
        data = {
            "height_map": height_map,
            "metadata": metadata,
            "version": metadata.get("version", 2)
        }
        
        # Generate output filename if not provided
        if output is None:
            output = Path(f"{tmd_file.stem}.{format}")
        
        # Get exporter and export the data
        with console.status(f"Exporting to {format} format..."):
            exporter = TMDDataIOFactory.get_exporter(
                format, 
                compression_level=compression
            )
            output_path = exporter.export(data, str(output))
        
        # Show success message
        print_success(f"File exported successfully to {output_path}")
        print(f"Size: {Path(output_path).stat().st_size / 1024:.1f} KB")
        
        return 0
    except Exception as e:
        print_error(f"Export failed: {e}")
        return 1

def main():
    """Run the TMD CLI application."""
    app()

if __name__ == "__main__":
    sys.exit(main())
