"""
TMD Compression Utilities

This module provides functions for compressing TMD files through various methods
including downsampling, quantization, and combination of both.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
from rich import print as rprint

logger = logging.getLogger(__name__)

# Try importing TMD, but provide fallbacks if not available
try:
    from tmd import TMD
except ImportError:
    logger.warning("TMD module not available, using limited functionality")
    TMD = None

def compress_height_map(
    height_map: np.ndarray,
    mode: str = "downsample",
    scale: float = 0.5,
    levels: int = 256,
    method: str = "bilinear"
) -> np.ndarray:
    """
    Compress a height map using specified method.
    
    Args:
        height_map: Original height map as NumPy array
        mode: Compression mode - 'downsample', 'quantize', or 'both'
        scale: Scale factor for downsampling (0-1)
        levels: Number of quantization levels
        method: Interpolation method for downsampling
        
    Returns:
        Compressed height map
    """
    result = height_map.copy()
    
    # Apply downsampling if requested
    if mode in ["downsample", "both"]:
        try:
            from scipy.ndimage import zoom
            
            # Calculate new dimensions
            new_height = max(2, int(height_map.shape[0] * scale))
            new_width = max(2, int(height_map.shape[1] * scale))
            
            # Map method names to order parameter
            order_map = {"nearest": 0, "bilinear": 1, "bicubic": 3}
            order = order_map.get(method, 1)  # Default to bilinear
            
            # Downsample using scipy.ndimage.zoom
            result = zoom(height_map, (new_height/height_map.shape[0], new_width/height_map.shape[1]), order=order)
            
            logger.info(f"Downsampled from {height_map.shape} to {result.shape}")
        except ImportError:
            logger.error("SciPy is required for downsampling but not available")
            return height_map
    
    # Apply quantization if requested
    if mode in ["quantize", "both"]:
        min_val = np.min(result)
        max_val = np.max(result)
        
        if min_val == max_val:
            logger.warning("Height map is flat, quantization not applied")
        else:
            # Quantize to specified number of levels
            scaled = (result - min_val) / (max_val - min_val)  # Scale to 0-1 range
            quantized = np.round(scaled * (levels - 1)) / (levels - 1)  # Quantize
            result = quantized * (max_val - min_val) + min_val  # Scale back
            
            logger.info(f"Quantized height map to {levels} levels")
    
    return result

def compress_tmd_file(
    tmd_file: Path,
    output: Optional[Path] = None,
    mode: str = "downsample",
    scale: float = 0.5,
    levels: int = 256,
    method: str = "bilinear",
    version: int = 2,
    **kwargs
) -> Dict[str, Any]:
    """
    Compress a TMD file using the specified method and save it.
    
    Args:
        tmd_file: Path to input TMD file
        output: Path for output file (or None to autogenerate)
        mode: Compression mode - 'downsample', 'quantize', or 'both'
        scale: Scale factor for downsampling (0-1)
        levels: Number of quantization levels
        method: Interpolation method for downsampling
        version: TMD file format version to save as
        
    Returns:
        Dictionary with compression summary
    """
    # Validate inputs
    if not tmd_file.exists() or not tmd_file.is_file():
        return {"success": False, "error": f"Input file not found: {tmd_file}"}
    
    if scale <= 0 or scale > 1:
        return {"success": False, "error": f"Scale factor must be between 0 and 1, got {scale}"}
    
    if levels < 2:
        return {"success": False, "error": f"Levels must be at least 2, got {levels}"}
    
    # Generate output path if not provided
    if output is None:
        suffix = ""
        if mode == "downsample":
            suffix = f"_ds{int(scale*100)}"
        elif mode == "quantize":
            suffix = f"_q{levels}"
        else:  # both
            suffix = f"_ds{int(scale*100)}_q{levels}"
            
        output = tmd_file.with_stem(f"{tmd_file.stem}{suffix}")
    
    try:
        # Load the TMD file
        if TMD is not None:
            tmd_obj = TMD(str(tmd_file))
            height_map = tmd_obj.height_map()
            metadata = tmd_obj.metadata()
        else:
            # Fallback method if TMD module not available
            from tmd.utils.utils import TMDUtils
            metadata, height_map = TMDUtils.process_tmd_file(tmd_file)
        
        # Original dimensions and stats for reporting
        orig_shape = height_map.shape
        orig_min = height_map.min()
        orig_max = height_map.max()
        orig_size = tmd_file.stat().st_size
        
        # Compress the height map
        compressed_height_map = compress_height_map(
            height_map, 
            mode=mode, 
            scale=scale, 
            levels=levels, 
            method=method
        )
        
        # Save the compressed TMD file
        if TMD is not None:
            # Update metadata to indicate compression
            metadata["compressed"] = True
            metadata["compression_mode"] = mode
            if mode in ["downsample", "both"]:
                metadata["downsample_scale"] = scale
                metadata["downsample_method"] = method
            if mode in ["quantize", "both"]:
                metadata["quantize_levels"] = levels
            
            # Create and save the new TMD object
            compressed_tmd = TMD(compressed_height_map, metadata)
            compressed_tmd.save(str(output), version=version)
        else:
            # Fallback method if TMD module not available
            logger.warning("TMD module not available, using limited export functionality")
            from tmd.utils.files import TMDFileUtilities
            TMDFileUtilities.save_heightmap_to_tmd(compressed_height_map, output, metadata)
        
        # Calculate statistics for reporting
        new_shape = compressed_height_map.shape
        new_size = output.stat().st_size
        size_reduction = 1.0 - (new_size / orig_size)
        
        # Create summary dictionary
        summary = {
            "success": True,
            "input_file": str(tmd_file),
            "output_file": str(output),
            "mode": mode,
            "original_dimensions": f"{orig_shape[1]}x{orig_shape[0]}",
            "compressed_dimensions": f"{new_shape[1]}x{new_shape[0]}",
            "original_size": orig_size,
            "compressed_size": new_size,
            "size_reduction": size_reduction,
            "original_range": (orig_min, orig_max),
            "compressed_range": (compressed_height_map.min(), compressed_height_map.max())
        }
        
        return summary
        
    except Exception as e:
        logger.exception(f"Error compressing TMD file: {e}")
        return {"success": False, "error": str(e)}

def display_compression_summary(summary: Dict[str, Any]) -> None:
    """
    Display a summary of the compression results.
    
    Args:
        summary: Dictionary with compression summary from compress_tmd_file
    """
    if not summary["success"]:
        rprint(f"[bold red]Compression failed:[/bold red] {summary.get('error', 'Unknown error')}")
        return
    
    # Format size values
    orig_size_kb = summary["original_size"] / 1024
    new_size_kb = summary["compressed_size"] / 1024
    reduction_pct = summary["size_reduction"] * 100
    
    # Display summary
    rprint("\n[bold green]Compression successful[/bold green]")
    rprint(f"[bold]Input:[/bold] {summary['input_file']}")
    rprint(f"[bold]Output:[/bold] {summary['output_file']}")
    rprint(f"[bold]Mode:[/bold] {summary['mode']}")
    rprint(f"[bold]Dimensions:[/bold] {summary['original_dimensions']} → {summary['compressed_dimensions']}")
    rprint(f"[bold]File size:[/bold] {orig_size_kb:.1f} KB → {new_size_kb:.1f} KB ({reduction_pct:.1f}% reduction)")
    
    # Add compression-specific details
    if summary["mode"] in ["downsample", "both"]:
        rprint(f"[bold]Scale factor:[/bold] {summary.get('scale', 'N/A')}")
        rprint(f"[bold]Method:[/bold] {summary.get('method', 'N/A')}")
    
    if summary["mode"] in ["quantize", "both"]:
        rprint(f"[bold]Quantization levels:[/bold] {summary.get('levels', 'N/A')}")
