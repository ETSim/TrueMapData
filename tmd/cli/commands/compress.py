#!/usr/bin/env python3
"""
Compression command implementations for TMD CLI.

This module provides commands for compressing TMD files through downsampling,
quantization, or a combination of both methods.
"""

import os
import time
from pathlib import Path
from typing import Optional, Dict, Any, Union

# Import NumPy
import numpy as np

# Import TMD core
from tmd import TMD

from tmd.cli.core import (
    load_tmd_file, create_output_dir, 
    print_warning, print_error, print_success,
    auto_open_file, console
)

from tmd.utils.utils import TMDUtils
from tmd.cli.exceptions import CommandError, InputError

# Import caching utilities
from tmd.cli.utils.caching import get_cache_stats, clear_cache

def display_file_info_command(
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
        data = load_tmd_file(tmd_file, with_console_status=True)
        if not data:
            return False
        
        metadata = data.metadata()
        height_map = data.height_map()
        file_size = tmd_file.stat().st_size
        
        from rich.panel import Panel
        from tmd.cli.core import display_metadata
        
        console.print(Panel.fit(
            f"[bold]TMD File:[/bold] {tmd_file}\n"
            f"File size: {file_size / 1024:.1f} KB\n"
            f"Dimensions: {height_map.shape[1]}×{height_map.shape[0]}\n"
            f"Height Range: {height_map.min():.6f} to {height_map.max():.6f}\n"
            f"Memory usage: {height_map.nbytes / 1024:.1f} KB"
        ))
        
        display_metadata(metadata)
        
        if show_sample:
            # Show a small sample of the height map
            console.print("\n[bold]Height Map Sample[/bold] (first few rows and columns):")
            sample_rows = min(5, height_map.shape[0])
            sample_cols = min(8, height_map.shape[1])
            console.print(height_map[:sample_rows, :sample_cols])
                
        return True
    except Exception as e:
        print_error(f"Error displaying file info: {e}")
        # Use custom CLI exception
        raise CommandError(f"Failed to display file info: {e}") from e

def compress_tmd_command(
    tmd_file: Path,
    output: Optional[Path] = None,
    mode: str = "downsample",
    scale: float = 0.5,
    method: str = "bilinear",
    levels: int = 256,
    version: int = 2,
    auto_open: bool = False
) -> bool:
    """
    Compress a TMD file using downsampling or quantization.
    
    Args:
        tmd_file: Path to TMD file
        output: Output file path (if None, auto-generated)
        mode: Compression mode (downsample, quantize, both)
        scale: Scale factor for downsampling (0-1)
        method: Interpolation method (nearest, bilinear, bicubic)
        levels: Number of quantization levels
        version: TMD format version (1 or 2)
        auto_open: Whether to automatically open the output file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Validate parameters using custom exceptions
        if mode == "downsample" or mode == "both":
            if not 0 < scale < 1:
                raise InputError("Scale factor must be between 0 and 1")
                
        if mode == "quantize" or mode == "both":
            if levels < 2:
                raise InputError("Quantization levels must be at least 2")
        
        # Load the TMD file
        data = load_tmd_file(tmd_file, with_console_status=True)
        if not data:
            return False
            
        # Get height map and metadata
        height_map = data.height_map()
        metadata = data.metadata()
        original_size = tmd_file.stat().st_size
        original_dims = height_map.shape
        
        # Process based on mode
        if mode == "downsample" or mode == "both":
            # Determine new dimensions
            new_height = int(height_map.shape[0] * scale)
            new_width = int(height_map.shape[1] * scale)
            
            with console.status(f"Downsampling to {new_width}×{new_height}..."):
                # Use the enhanced TMDUtils method for downsampling
                processed = TMDUtils.downsample_array(height_map, new_width, new_height, method)
        else:
            processed = height_map
            
        if mode == "quantize" or mode == "both":
            with console.status(f"Quantizing to {levels} height levels..."):
                # Use the enhanced TMDUtils method for quantization
                processed = TMDUtils.quantize_array(processed, levels)
        
        # Prepare output filename
        if output is None:
            output_dir = create_output_dir("compressed")
            if mode == "downsample":
                suffix = f"_ds{int(scale*100)}"
            elif mode == "quantize":
                suffix = f"_q{levels}"
            else:
                suffix = f"_comp"
                
            output = output_dir / f"{tmd_file.stem}{suffix}.tmd"
        
        # Create a comment
        if mode == "downsample":
            comment = f"Downsampled from {original_dims[1]}×{original_dims[0]} using {method}"
        elif mode == "quantize":
            comment = f"Quantized from float32 to {levels} levels"
        else:
            comment = f"Compressed: downsampled to {int(scale*100)}% and quantized to {levels} levels"
            
        # Write the new TMD file
        with console.status("Writing compressed TMD file..."):
            TMDUtils.write_tmd_file(
                processed,
                output,
                comment=comment,
                x_length=metadata.get("x_length", 10.0),
                y_length=metadata.get("y_length", 10.0),
                x_offset=metadata.get("x_offset", 0.0),
                y_offset=metadata.get("y_offset", 0.0),
                version=version
            )
            
        # Report results
        new_size = output.stat().st_size
        reduction = (1 - new_size / original_size) * 100
        
        from rich.panel import Panel
        console.print(Panel.fit(
            f"[bold green]Compression successful![/bold green]\n\n"
            f"Original: {original_dims[1]}×{original_dims[0]} ({original_size / 1024:.1f} KB)\n"
            f"New: {processed.shape[1]}×{processed.shape[0]} ({new_size / 1024:.1f} KB)\n"
            f"Reduction: {reduction:.1f}%\n\n"
            f"Output: {output}"
        ))
            
        # Auto-open if requested
        if auto_open:
            auto_open_file(output)
            
        return True
    except InputError as e:
        # Handle input validation errors specifically
        print_error(str(e))
        return False
    except Exception as e:
        # Handle other errors
        print_error(f"Error during compression: {e}")
        return False
