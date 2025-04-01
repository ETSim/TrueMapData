"""
Bump map generation module for heightmaps.

This module provides functions for generating bump maps from height maps.
Bump maps are used in 3D rendering to add surface detail with minimal performance impact.
"""

import os
import logging
import numpy as np
from typing import Optional, Union

from PIL import Image, ImageFilter
from .utils import ensure_directory_exists, normalize_heightmap, handle_nan_values

# Set up logging
logger = logging.getLogger(__name__)


def convert_heightmap_to_bump_map(
    height_map: np.ndarray,
    filename: str = "bump_map.png",
    strength: float = 1.0,
    blur_radius: float = 1.0,
    bit_depth: int = 8,
    **kwargs
) -> Union[str, Image.Image]:
    """
    Convert a height map to a bump map.
    
    Args:
        height_map: 2D numpy array of height values
        filename: Path to save the output image
        strength: Factor to control the strength of bumps
        blur_radius: Radius for Gaussian blur to smooth the result
        bit_depth: Bit depth of output image (8 or 16)
        **kwargs: Additional options
        
    Returns:
        Path to the saved file or PIL Image object
    """
    try:
        # Ensure output directory exists
        if not ensure_directory_exists(filename):
            logger.error(f"Failed to create output directory for {filename}")
            return None
            
        # Process height map
        if np.any(np.isnan(height_map)):
            height_map = handle_nan_values(height_map)
            
        # Normalize height map to 0-1 range
        height_norm = normalize_heightmap(height_map)
        
        # Use Sobel filter to detect edges (approximates slope)
        from scipy import ndimage
        sobel_x = ndimage.sobel(height_norm, axis=1)
        sobel_y = ndimage.sobel(height_norm, axis=0)
        
        # Apply strength factor
        sobel_x *= strength
        sobel_y *= strength
        
        # Compute gradient magnitude
        gradient = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Scale to 0-1
        gradient_norm = normalize_heightmap(gradient)
        
        # Convert to desired bit depth
        if bit_depth == 16:
            bump_map = (gradient_norm * 65535).astype(np.uint16)
        else:
            bump_map = (gradient_norm * 255).astype(np.uint8)
        
        # Convert to PIL Image
        if bit_depth == 16:
            # Create 16-bit grayscale image
            img = Image.fromarray(bump_map, mode='I;16')
        else:
            # Create 8-bit grayscale image
            img = Image.fromarray(bump_map, mode='L')
        
        # Apply Gaussian blur if requested
        if blur_radius > 0:
            img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        # Save to file
        img.save(filename)
        
        # Store filename in the image object for stats reporting
        img.filename = filename
        
        logger.info(f"Bump map saved to {filename}")
        return img
        
    except Exception as e:
        logger.error(f"Error creating bump map: {e}")
        import traceback
        traceback.print_exc()
        return None
