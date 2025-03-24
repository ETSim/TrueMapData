""".

Hillshade generation module.

This module provides functions for creating hillshade visualizations from heightmaps.
"""

import os
import logging
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

def generate_hillshade(height_map, filename="hillshade.png", altitude=45, azimuth=0, z_factor=1.0):
    """.

    Generates a hillshade image from a height map.

    Hillshading simulates the cast of shadows on terrain, highlighting ridges and valleys.
    The light source position is defined by altitude and azimuth angles.

    Args:
        height_map: 2D numpy array of height values.
        filename: Name of the output PNG file.
        altitude: Light source altitude in degrees above the horizon (0-90).
        azimuth: Light source azimuth in degrees clockwise from North (0-360).
        z_factor: Vertical exaggeration factor to enhance terrain features.

    Returns:
        PIL Image object of the hillshade.
    """
    # Ensure proper data type
    height_map = height_map.astype(np.float32)
    rows, cols = height_map.shape

    # Convert illumination angles from degrees to radians
    altitude = np.radians(altitude)
    azimuth = np.radians(azimuth)

    # Calculate slope and aspect using 3x3 windows
    dx = np.zeros_like(height_map)
    dy = np.zeros_like(height_map)

    # Calculate gradients using central differences for interior cells
    dx[1:-1, 1:-1] = ((height_map[1:-1, 2:] - height_map[1:-1, :-2]) * z_factor) / 2.0
    dy[1:-1, 1:-1] = ((height_map[2:, 1:-1] - height_map[:-2, 1:-1]) * z_factor) / 2.0

    # Calculate slope (in radians)
    slope = np.arctan(np.sqrt(dx**2 + dy**2))

    # Calculate aspect (in radians)
    aspect = np.arctan2(dy, dx)

    # Calculate hillshade using the formula:
    # hillshade = cos(zenith_angle) * cos(slope) + sin(zenith_angle) * sin(slope) * cos(azimuth - aspect)
    zenith = np.pi / 2 - altitude
    hillshade = np.cos(zenith) * np.cos(slope) + np.sin(zenith) * np.sin(slope) * np.cos(
        azimuth - aspect
    )

    # Scale hillshade values to the range [0, 255]
    hillshade = np.clip(hillshade, 0, 1)
    hillshade = (hillshade * 255).astype(np.uint8)

    # Create and save the image
    im = Image.fromarray(hillshade)

    # Add metadata for the hillshade parameters
    metadata = {
        "altitude": f"{np.degrees(altitude):.1f}°",
        "azimuth": f"{np.degrees(azimuth):.1f}°",
        "z_factor": str(z_factor),
    }
    im.info = metadata

    if filename:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        im.save(filename)
        logger.info(f"Hillshade image saved to {filename}")

    return im
