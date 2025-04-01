#!/usr/bin/env python3
"""
Compression utilities for TMD CLI tools.

This module provides specialized utilities for compression features
of TMD command-line tools.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Optional, Any, Dict, List, Union, Tuple

from tmd.cli.core.ui import print_error, print_warning, HAS_RICH

logger = logging.getLogger(__name__)

def perform_downsampling(height_map: np.ndarray, new_height: int, new_width: int, method: str) -> Optional[np.ndarray]:
    """
    Downsample a height map to the specified dimensions.
    
    Args:
        height_map: Original height map
        new_height: Target height
        new_width: Target width
        method: Interpolation method (nearest, bilinear, bicubic)
        
    Returns:
        Downsampled height map or None on error
    """
    try:
        # Use scipy for interpolation
        import scipy.ndimage
        
        # Select interpolation order based on method
        if method == "nearest":
            order = 0
        elif method == "bilinear":
            order = 1
        elif method == "bicubic":
            order = 3
        else:
            order = 1  # Default to bilinear
            
        # Calculate zoom factors
        y_factor = new_height / height_map.shape[0]
        x_factor = new_width / height_map.shape[1]
            
        # Resize the height map
        return scipy.ndimage.zoom(
            height_map, 
            (y_factor, x_factor), 
            order=order
        )
    except Exception as e:
        print_error(f"Error during downsampling: {str(e)}")
        logger.error(f"Error during downsampling: {e}")
        return None


def quantize_height_map(height_map: np.ndarray, levels: int) -> Optional[np.ndarray]:
    """
    Quantize height values to reduce precision.
    
    Args:
        height_map: Original height map
        levels: Number of quantization levels
        
    Returns:
        Quantized height map or None on error
    """
    try:
        if levels < 2:
            levels = 2  # Ensure at least 2 levels
            
        # Get height range
        height_min = height_map.min()
        height_max = height_map.max()
        
        # Check if range is valid
        if height_max <= height_min:
            return height_map  # No change needed
        
        # Normalize to 0-1 range
        normalized = (height_map - height_min) / (height_max - height_min)
        
        # Quantize to specified levels
        quantized_normalized = np.round(normalized * (levels - 1)) / (levels - 1)
        
        # Convert back to original range
        quantized = quantized_normalized * (height_max - height_min) + height_min
        
        return quantized
    except Exception as e:
        print_error(f"Error during quantization: {str(e)}")
        logger.error(f"Error during quantization: {e}")
        return None


def calculate_compression_ratio(original_size: int, compressed_size: int) -> float:
    """
    Calculate compression ratio as a percentage.
    
    Args:
        original_size: Size of original file in bytes
        compressed_size: Size of compressed file in bytes
        
    Returns:
        Reduction percentage (0-100)
    """
    if original_size == 0:
        return 0.0
    return (1 - compressed_size / original_size) * 100