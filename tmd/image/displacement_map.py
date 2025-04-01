"""
Displacement map generation module.

This module provides functions for generating displacement maps from height maps.
"""

import os
import numpy as np
import logging
from typing import Optional, Union, Tuple

from .image_io import save_image
from ..model.mesh_utils import ensure_directory_exists

# Set up logging
logger = logging.getLogger(__name__)

def export_displacement_map(
    height_map: np.ndarray,
    output_file: str,
    bit_depth: int = 16,
    normalize: bool = True,
    invert: bool = False,
    **kwargs
) -> Optional[str]:
    """
    Export a height map as a displacement map.
    
    Args:
        height_map: 2D numpy array of height values
        output_file: Path to save the output image
        bit_depth: Bit depth of output image (8 or 16)
        normalize: Whether to normalize the height values to 0-1 range
        invert: Whether to invert the height values
        **kwargs: Additional options
        
    Returns:
        Path to the saved image or None if failed
    """
    try:
        # Ensure output directory exists
        if not ensure_directory_exists(output_file):
            logger.error(f"Failed to create output directory for {output_file}")
            return None
        
        # Process the height map
        disp_map = process_displacement_map(height_map, normalize, invert)
        
        # Save to file
        return save_image(disp_map, output_file, bit_depth=bit_depth)
        
    except Exception as e:
        logger.error(f"Error exporting displacement map: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_displacement_map(
    height_map: np.ndarray,
    normalize: bool = True,
    invert: bool = False
) -> np.ndarray:
    """
    Process a height map into a displacement map.
    
    Args:
        height_map: 2D numpy array of height values
        normalize: Whether to normalize the height values to 0-1 range
        invert: Whether to invert the height values
        
    Returns:
        Processed displacement map as 2D numpy array
    """
    # Make a copy to avoid modifying original
    disp_map = height_map.copy()
    
    # Normalize if requested
    if normalize:
        h_min = np.min(disp_map)
        h_max = np.max(disp_map)
        
        if h_max > h_min:
            disp_map = (disp_map - h_min) / (h_max - h_min)
        else:
            disp_map = np.zeros_like(disp_map)
    
    # Invert if requested
    if invert:
        disp_map = 1.0 - disp_map
    
    return disp_map


def convert_heightmap_to_displacement_map(
    height_map: np.ndarray,
    output_file: str,
    **kwargs
) -> Optional[str]:
    """
    Convert a height map to a displacement map.
    
    Args:
        height_map: 2D numpy array of height values
        output_file: Path to save the output image
        **kwargs: Additional options for export_displacement_map
        
    Returns:
        Path to the saved image or None if failed
    """
    return export_displacement_map(height_map, output_file, **kwargs)
