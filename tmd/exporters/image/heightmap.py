"""
Heightmap export module.

This module provides functions for exporting heightmaps to image formats.
"""

import os
import numpy as np
import logging
from typing import Optional, Union

from .image_io import save_heightmap as save_image_heightmap
from .utils import ensure_directory_exists, normalize_heightmap, handle_nan_values

# Set up logging
logger = logging.getLogger(__name__)


def convert_heightmap_to_heightmap(
    height_map: np.ndarray,
    output_file: str,
    bit_depth: int = 16,
    normalize: bool = True,
    **kwargs
) -> Optional[str]:
    """
    Convert a heightmap to an image file.
    
    This function essentially passes through a heightmap for saving as an image,
    applying any requested normalization and other processing.
    
    Args:
        height_map: 2D numpy array of height values
        output_file: Path to save the output image
        bit_depth: Bit depth of output image (8 or 16)
        normalize: Whether to normalize the height values
        **kwargs: Additional options
        
    Returns:
        Path to the saved image or None if failed
    """
    try:
        # Ensure output directory exists
        if not ensure_directory_exists(output_file):
            logger.error(f"Failed to create output directory for {output_file}")
            return None
        
        # Process the heightmap
        processed_map = height_map.copy()
        
        # Handle NaN values if present
        if np.any(np.isnan(processed_map)):
            processed_map = handle_nan_values(processed_map)
        
        # Save to file
        return save_image_heightmap(
            processed_map, 
            output_file, 
            bit_depth=bit_depth,
            normalize=normalize,
            **kwargs
        )
        
    except Exception as e:
        logger.error(f"Error exporting heightmap: {e}")
        import traceback
        traceback.print_exc()
        return None


def export_heightmap(
    height_map: np.ndarray,
    output_file: str,
    **kwargs
) -> Optional[str]:
    """
    Export a heightmap to an image file.
    
    Args:
        height_map: 2D numpy array of height values
        output_file: Path to save the output file
        **kwargs: Additional options for convert_heightmap_to_heightmap
        
    Returns:
        Path to the saved image or None if failed
    """
    return convert_heightmap_to_heightmap(height_map, output_file, **kwargs)
