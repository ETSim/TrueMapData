""".

Displacement map generation module.

This module provides functions for converting heightmaps to displacement maps.
"""

import os
import logging
import numpy as np
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

def convert_heightmap_to_displacement_map(height_map, filename="displacement_map.png", units=None):
    """.

    Converts the height map into a grayscale displacement map (PNG).

    Args:
        height_map: 2D numpy array of height values.
        filename: Name of the output PNG file.
        units: Physical units information (e.g., "µm", "nm").

    Returns:
        PIL Image object of the displacement map.
    """
    hmin = float(height_map.min())
    hmax = float(height_map.max())
    norm = (height_map - hmin) / (hmax - hmin) * 255.0
    norm = norm.astype(np.uint8)
    im = Image.fromarray(norm)

    # Add physical units to metadata if provided
    if units:
        metadata = {"Height_Range": f"{hmin:.2f} to {hmax:.2f} {units}", "Units": units}
        im.info = {k: str(v) for k, v in metadata.items()}

        # Add text annotation
        try:
            im_rgba = im.convert("RGBA")
            overlay = Image.new("RGBA", im_rgba.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            text = f"Range: {hmin:.2f} to {hmax:.2f} {units}"
            draw.text((10, 10), text, fill=(255, 255, 255, 128))
            im_rgba = Image.alpha_composite(im_rgba, overlay)
            im = im_rgba.convert(im.mode)
        except Exception as e:
            logger.warning(f"Text annotation failed: {e}")

    if filename:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        im.save(filename)
        logger.info(f"Displacement map saved to {filename}")
        
    return im
