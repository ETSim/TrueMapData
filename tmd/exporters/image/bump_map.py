""".

Bump map generation module.

This module provides functions for converting heightmaps to bump maps.
"""

import os
import logging
import numpy as np
from PIL import Image
from scipy import ndimage

logger = logging.getLogger(__name__)

def convert_heightmap_to_bump_map(height_map, filename="bump_map.png", strength=1.0, blur_radius=1.0):
    """.

    Converts the height map to a bump map with optional blurring.

    Args:
        height_map: 2D numpy array of height values.
        filename: Name of the output PNG file.
        strength: Strength factor for the bump effect.
        blur_radius: Radius for Gaussian blur to smooth the result.

    Returns:
        PIL Image object of the bump map.
    """
    height_map = height_map.astype(np.float32)
    h_min = np.min(height_map)
    h_max = np.max(height_map)
    # Normalize height map
    bump_map = (
        ((height_map - h_min) / (h_max - h_min)) if h_max > h_min else np.zeros_like(height_map)
    )
    bump_map *= strength

    if blur_radius > 0:
        bump_map = ndimage.gaussian_filter(bump_map, sigma=blur_radius)

    b_min = np.min(bump_map)
    b_max = np.max(bump_map)
    bump_map = ((bump_map - b_min) / (b_max - b_min)) if b_max > b_min else bump_map
    bump_map = (bump_map * 255).astype(np.uint8)

    im = Image.fromarray(bump_map)
    
    if filename:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        im.save(filename)
        logger.info(f"Bump map saved to {filename}")
        
    return im
