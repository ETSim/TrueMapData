#!/usr/bin/env python3
"""
TMD Compression CLI

A command-line interface for compressing TMD files by reducing resolution
or quantizing height values.

This script provides utilities for creating smaller TMD files while
maintaining the essential topographic information.

Usage:
    python tmd_compress.py --help
    python tmd_compress.py downsample Dime.tmd --scale 0.5
    python tmd_compress.py quantize Dime.tmd --levels 256
"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Union, Dict, Any, Tuple

# Terminal interface libraries
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress

# Import NumPy and TMD
import numpy as np
from tmd import TMD
from tmd.utils.utils import TMDUtils

# Import caching utilities
from tmd.cli.utils.caching import get_cache_stats, clear_cache

from tmd.cli.common import (
    load_config,
    load_tmd_file,
    create_output_dir,
    auto_open_file,
    console,
    display_metadata,
    check_dependencies_and_install,
    HAS_RICH
)

# Initialize Typer app
app = typer.Typer(help="TMD Compression Tool")

@app.callback()
def callback():
    """ 
    TMD Compression Tool - Reduce file size while preserving topographic data
    """
    pass

@app.command("info")
def display_file_info(
    tmd_file: Path = typer.Argument(..., help="Path to TMD file", exists=True),
):
    """Display information about a TMD file including file size."""
    data = load_tmd_file(tmd_file, with_console_status=True)
    if not data:
        return
    
    metadata = data.metadata()
    height_map = data.height_map()
    file_size = tmd_file.stat().st_size
    
    if HAS_RICH:
        console.print(Panel.fit(
            f"[bold]TMD File:[/bold] {tmd_file}\n"
            f"File size: {file_size / 1024:.1f} KB\n"
            f"Dimensions: {height_map.shape[1]}×{height_map.shape[0]}\n"
            f"Height Range: {height_map.min():.6f} to {height_map.max():.6f}\n"
            f"Memory usage: {height_map.nbytes / 1024:.1f} KB"
        ))
        
        display_metadata(metadata)
    else:
        print(f"TMD File: {tmd_file}")
        print(f"File size: {file_size / 1024:.1f} KB")
        print(f"Dimensions: {height_map.shape[1]}×{height_map.shape[0]}")
        print(f"Height Range: {height_map.min():.6f} to {height_map.max():.6f}")
        print(f"Memory usage: {height_map.nbytes / 1024:.1f} KB")

@app.command("downsample")
def downsample_tmd(
    tmd_file: Path = typer.Argument(..., help="Path to TMD file", exists=True),
    scale: float = typer.Option(0.5, help="Scale factor (0-1)"),
    method: str = typer.Option("bilinear", help="Interpolation method: nearest, bilinear, bicubic"),
    output: Optional[Path] = typer.Option(None, help="Output filename"),
    version: int = typer.Option(2, help="TMD format version (1 or 2)"),
    auto_open: bool = typer.Option(False, help="Automatically open the output file")
):
    """Downsample a TMD file to reduce resolution."""
    if not 0 < scale < 1:
        console.print("[bold red]Error:[/bold red] Scale factor must be between 0 and 1")
        return
        
    data = load_tmd_file(tmd_file, with_console_status=True)
    if not data:
        return
    
    # Get the height map and original metadata
    height_map = data.height_map()
    metadata = data.metadata()
    
    original_size = tmd_file.stat().st_size
    original_dims = height_map.shape
    
    # Determine new dimensions
    new_height = int(height_map.shape[0] * scale)
    new_width = int(height_map.shape[1] * scale)
    
    with console.status(f"Downsampling to {new_width}×{new_height}..."):
        # Perform downsampling
        try:
            # Initialize scipy if not available
            import scipy.ndimage
            
            if method == "nearest":
                order = 0
            elif method == "bilinear":
                order = 1
            elif method == "bicubic":
                order = 3
            else:
                order = 1
                
            # Resize the height map
            downsampled = scipy.ndimage.zoom(
                height_map, 
                (new_height / height_map.shape[0], new_width / height_map.shape[1]), 
                order=order
            )
            
        except ImportError:
            # Fallback to numpy for nearest neighbor
            console.print("[yellow]Warning:[/yellow] scipy not found, using simple numpy downsampling")
            
            y_indices = np.linspace(0, height_map.shape[0] - 1, new_height).astype(np.int32)
            x_indices = np.linspace(0, height_map.shape[1] - 1, new_width).astype(np.int32)
            downsampled = height_map[y_indices[:, np.newaxis], x_indices]
    
    # Prepare output filename
    if output is None:
        output_dir = create_output_dir("compressed")
        output = output_dir / f"{tmd_file.stem}_ds{int(scale*100)}.tmd"
    
    # Adjust metadata for new dimensions
    x_length = metadata.get("x_length", 10.0)
    y_length = metadata.get("y_length", 10.0)
    x_offset = metadata.get("x_offset", 0.0)
    y_offset = metadata.get("y_offset", 0.0)
    
    # Create a comment about compression
    comment = f"Downsampled from {original_dims[1]}×{original_dims[0]} using {method}"
    
    # Write the new TMD file
    with console.status("Writing compressed TMD file..."):
        try:
            TMDUtils.write_tmd_file(
                downsampled,
                output,
                comment=comment,
                x_length=x_length,
                y_length=y_length,
                x_offset=x_offset,
                y_offset=y_offset,
                version=version
            )
            
            new_size = output.stat().st_size
            reduction = (1 - new_size / original_size) * 100
            
            if HAS_RICH:
                console.print(Panel.fit(
                    f"[bold green]Compression successful![/bold green]\n\n"
                    f"Original: {original_dims[1]}×{original_dims[0]} ({original_size / 1024:.1f} KB)\n"
                    f"New: {new_width}×{new_height} ({new_size / 1024:.1f} KB)\n"
                    f"Reduction: {reduction:.1f}%\n\n"
                    f"Output: {output}"
                ))
            else:
                print(f"\nCompression successful!")
                print(f"Original: {original_dims[1]}×{original_dims[0]} ({original_size / 1024:.1f} KB)")
                print(f"New: {new_width}×{new_height} ({new_size / 1024:.1f} KB)")
                print(f"Reduction: {reduction:.1f}%")
                print(f"Output: {output}")
                
            if auto_open:
                auto_open_file(output)
                
        except Exception as e:
            console.print(f"[bold red]Error creating compressed file:[/bold red] {str(e)}")

@app.command("quantize")
def quantize_tmd(
    tmd_file: Path = typer.Argument(..., help="Path to TMD file", exists=True),
    levels: int = typer.Option(256, help="Number of height levels"),
    output: Optional[Path] = typer.Option(None, help="Output filename"),
    version: int = typer.Option(2, help="TMD format version (1 or 2)"),
    auto_open: bool = typer.Option(False, help="Automatically open the output file")
):
    """Quantize height values in a TMD file to reduce file size."""
    if levels < 2:
        console.print("[bold red]Error:[/bold red] Levels must be at least 2")
        return
        
    data = load_tmd_file(tmd_file, with_console_status=True)
    if not data:
        return
    
    # Get the height map and original metadata
    height_map = data.height_map()
    metadata = data.metadata()
    
    original_size = tmd_file.stat().st_size
    
    with console.status(f"Quantizing to {levels} height levels..."):
        # Perform quantization
        height_min = height_map.min()
        height_max = height_map.max()
        
        # Normalize to 0-1 range
        normalized = (height_map - height_min) / (height_max - height_min) if height_max > height_min else height_map
        
        # Quantize to specified levels
        quantized_normalized = np.round(normalized * (levels - 1)) / (levels - 1)
        
        # Convert back to original range
        quantized = quantized_normalized * (height_max - height_min) + height_min
    
    # Prepare output filename
    if output is None:
        output_dir = create_output_dir("compressed")
        output = output_dir / f"{tmd_file.stem}_q{levels}.tmd"
    
    # Create a comment about quantization
    comment = f"Quantized from float32 to {levels} levels"
    
    # Write the new TMD file
    with console.status("Writing quantized TMD file..."):
        try:
            TMDUtils.write_tmd_file(
                quantized,
                output,
                comment=comment,
                x_length=metadata.get("x_length", 10.0),
                y_length=metadata.get("y_length", 10.0),
                x_offset=metadata.get("x_offset", 0.0),
                y_offset=metadata.get("y_offset", 0.0),
                version=version
            )
            
            new_size = output.stat().st_size
            reduction = (1 - new_size / original_size) * 100
            
            if HAS_RICH:
                console.print(Panel.fit(
                    f"[bold green]Quantization successful![/bold green]\n\n"
                    f"Original range: {height_min:.6f} to {height_max:.6f} (float32)\n"
                    f"Quantized to {levels} levels\n"
                    f"Original size: {original_size / 1024:.1f} KB\n"
                    f"New size: {new_size / 1024:.1f} KB\n"
                    f"Reduction: {reduction:.1f}%\n\n"
                    f"Output: {output}"
                ))
            else:
                print(f"\nQuantization successful!")
                print(f"Original range: {height_min:.6f} to {height_max:.6f} (float32)")
                print(f"Quantized to {levels} levels")
                print(f"Original size: {original_size / 1024:.1f} KB")
                print(f"New size: {new_size / 1024:.1f} KB")
                print(f"Reduction: {reduction:.1f}%")
                print(f"Output: {output}")
                
            if auto_open:
                auto_open_file(output)
                
        except Exception as e:
            console.print(f"[bold red]Error creating quantized file:[/bold red] {str(e)}")

@app.command("batch")
def batch_compress(
    input_dir: Path = typer.Argument(..., help="Directory containing TMD files", exists=True),
    mode: str = typer.Option("downsample", help="Compression mode: downsample or quantize"),
    scale: float = typer.Option(0.5, help="Scale factor for downsampling (0-1)"),
    levels: int = typer.Option(256, help="Number of height levels for quantization"),
    recursive: bool = typer.Option(False, help="Recursively search for TMD files"),
    version: int = typer.Option(2, help="TMD format version (1 or 2)")
):
    """Batch compress multiple TMD files in a directory."""
    # Find all TMD files
    if recursive:
        tmd_files = list(input_dir.glob("**/*.tmd"))
    else:
        tmd_files = list(input_dir.glob("*.tmd"))
    
    if not tmd_files:
        console.print(f"[bold yellow]Warning:[/bold yellow] No TMD files found in {input_dir}")
        return
    
    console.print(f"[bold blue]Found {len(tmd_files)} TMD files in {input_dir}[/bold blue]")
    
    # Create output directory
    output_dir = create_output_dir("batch_compressed")
    
    # Process each file
    with Progress() as progress:
        task = progress.add_task("Processing TMD files...", total=len(tmd_files))
        
        success_count = 0
        total_original_size = 0
        total_new_size = 0
        
        for tmd_file in tmd_files:
            progress.update(task, description=f"Processing {tmd_file.name}...")
            
            try:
                # Load the TMD file
                data = TMD(str(tmd_file))
                height_map = data.height_map()
                metadata = data.metadata()
                
                original_size = tmd_file.stat().st_size
                total_original_size += original_size
                
                if mode == "downsample":
                    # Determine new dimensions
                    new_height = int(height_map.shape[0] * scale)
                    new_width = int(height_map.shape[1] * scale)
                    
                    # Perform downsampling with bilinear interpolation
                    try:
                        import scipy.ndimage
                        downsampled = scipy.ndimage.zoom(
                            height_map, 
                            (new_height / height_map.shape[0], new_width / height_map.shape[1]), 
                            order=1
                        )
                        processed = downsampled
                        comment = f"Downsampled from {height_map.shape[1]}×{height_map.shape[0]} using bilinear"
                        suffix = f"_ds{int(scale*100)}"
                    except ImportError:
                        # Fallback to numpy for nearest neighbor
                        y_indices = np.linspace(0, height_map.shape[0] - 1, new_height).astype(np.int32)
                        x_indices = np.linspace(0, height_map.shape[1] - 1, new_width).astype(np.int32)
                        processed = height_map[y_indices[:, np.newaxis], x_indices]
                        comment = f"Downsampled from {height_map.shape[1]}×{height_map.shape[0]} using nearest"
                        suffix = f"_ds{int(scale*100)}"
                        
                elif mode == "quantize":
                    # Perform quantization
                    height_min = height_map.min()
                    height_max = height_map.max()
                    
                    # Normalize to 0-1 range
                    normalized = (height_map - height_min) / (height_max - height_min) if height_max > height_min else height_map
                    
                    # Quantize to specified levels
                    quantized_normalized = np.round(normalized * (levels - 1)) / (levels - 1)
                    
                    # Convert back to original range
                    processed = quantized_normalized * (height_max - height_min) + height_min
                    comment = f"Quantized from float32 to {levels} levels"
                    suffix = f"_q{levels}"
                    
                else:
                    progress.update(task, description=f"Unknown mode: {mode}")
                    progress.advance(task)
                    continue
                
                # Create the output filename
                output_file = output_dir / f"{tmd_file.stem}{suffix}.tmd"
                
                # Write the new TMD file
                TMDUtils.write_tmd_file(
                    processed,
                    output_file,
                    comment=comment,
                    x_length=metadata.get("x_length", 10.0),
                    y_length=metadata.get("y_length", 10.0),
                    x_offset=metadata.get("x_offset", 0.0),
                    y_offset=metadata.get("y_offset", 0.0),
                    version=version
                )
                
                new_size = output_file.stat().st_size
                total_new_size += new_size
                success_count += 1
                
            except Exception as e:
                progress.update(task, description=f"Error processing {tmd_file.name}: {str(e)}")
            
            progress.advance(task)
    
    # Display a summary
    if success_count > 0:
        reduction = (1 - total_new_size / total_original_size) * 100
        
        if HAS_RICH:
            console.print(Panel.fit(
                f"[bold green]Batch compression summary[/bold green]\n\n"
                f"Files processed: {success_count} of {len(tmd_files)}\n"
                f"Original size: {total_original_size / 1024 / 1024:.2f} MB\n"
                f"New size: {total_new_size / 1024 / 1024:.2f} MB\n"
                f"Overall reduction: {reduction:.1f}%\n\n"
                f"Output directory: {output_dir}"
            ))
        else:
            print(f"\nBatch compression summary")
            print(f"Files processed: {success_count} of {len(tmd_files)}")
            print(f"Original size: {total_original_size / 1024 / 1024:.2f} MB")
            print(f"New size: {total_new_size / 1024 / 1024:.2f} MB")
            print(f"Overall reduction: {reduction:.1f}%")
            print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    # Check dependencies
    check_dependencies_and_install()
    
    # Run the app
    app()
