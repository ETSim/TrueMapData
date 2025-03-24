""".

NPY format export/import for TMD data.

This module provides functions to export heightmap data to the NumPy .npy format,
which is a simple binary format for saving NumPy arrays.
"""

import os
import logging
import numpy as np

logger = logging.getLogger(__name__)

def export_to_npy(data, output_path):
    """.

    Export height map data to NumPy .npy format.

    Args:
        data: Height map as 2D numpy array or dictionary containing height_map
        output_path: Path to save the .npy file

    Returns:
        Path to the saved file
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # If data is a dictionary, extract the height map
    if isinstance(data, dict) and "height_map" in data:
        height_map = data["height_map"]
    else:
        height_map = data

    # Check if height_map is a numpy array
    if not isinstance(height_map, np.ndarray):
        raise TypeError("Height map must be a NumPy array")

    # Save as .npy file
    np.save(output_path, height_map)
    logger.info(f"Height map exported to {output_path}")

    return output_path

def load_from_npy(file_path):
    """.

    Load height map data from a .npy file.

    Args:
        file_path: Path to the .npy file

    Returns:
        NumPy array containing the height map
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        data = np.load(file_path)
        return data
    except Exception as e:
        logger.error(f"Error loading NPY file: {e}")
        raise
