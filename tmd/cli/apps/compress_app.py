#!/usr/bin/env python3
"""
Compression app for TMD CLI.
"""

from pathlib import Path
from typing import Optional
import typer

from tmd.cli.compression import compress_tmd_file, display_compression_summary
from tmd.cli.core.io import auto_open_file

def create_compress_app():
    """Create the compression app with all commands."""
    compress_app = typer.Typer(help="Compression tools for TMD files")
    
    compress_app.command(name="downsample")(compress_downsample)
    compress_app.command(name="quantize")(compress_quantize)
    compress_app.command(name="combined")(compress_combined)
    compress_app.command(name="batch")(compress_batch)
    
    return compress_app

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
    # Implementation from tmd.cli.commands.batch
    from tmd.cli.commands.compress import compress_batch_command
    return compress_batch_command(
        input_dir=input_dir,
        mode=mode,
        scale=scale,
        levels=levels,
        method=method,
        recursive=recursive,
        version=version
    )
