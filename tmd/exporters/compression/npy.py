""".

NPY format export/import for TMD data.

This module provides functions to export heightmap data to the NumPy .npy format,
which is a simple binary format for saving NumPy arrays.

Examples:
    >>> height_map = np.random.rand(100, 100)
    >>> export_to_npy(height_map, "heightmap.npy")
    >>> loaded_map = load_from_npy("heightmap.npy")
"""

import os
import logging
from typing import Dict, Union, Any
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


def export_to_npy(data: Union[np.ndarray, Dict[str, Any]], output_path: str) -> str:
    """.

    Export height map data to NumPy .npy format.

    Args:
        data: Height map as 2D numpy array or dictionary containing height_map key
        output_path: Path to save the .npy file

    Returns:
        Path to the saved file

    Raises:
        TypeError: If height map is not a NumPy array or cannot be extracted from data
        ValueError: If height map is not 2D

    Examples:
        >>> height_map = np.random.rand(100, 100)
        >>> export_to_npy(height_map, "heightmap.npy")
        'heightmap.npy'
    """
    # Ensure output directory exists
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # If data is a dictionary, extract the height map
    if isinstance(data, dict) and "height_map" in data:
        height_map = data["height_map"]
    else:
        height_map = data

    # Check if height_map is a numpy array
    if not isinstance(height_map, np.ndarray):
        raise TypeError("Height map must be a NumPy array")

    # Validate that height_map is 2D (optional validation)
    if height_map.ndim != 2:
        raise ValueError(f"Height map should be 2D, got {height_map.ndim}D array")

    # Save as .npy file
    np.save(output_path, height_map)
    logger.info(f"Height map exported to {output_path}")

    return output_path


def load_from_npy(file_path: str) -> np.ndarray:
    """.

    Load height map data from a .npy file.

    Args:
        file_path: Path to the .npy file

    Returns:
        NumPy array containing the height map

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file is not a valid .npy file
        
    Examples:
        >>> height_map = load_from_npy("heightmap.npy")
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        data = np.load(file_path)
        
        # Validate that loaded data is 2D (optional validation)
        if data.ndim != 2:
            logger.warning(f"Loaded height map is not 2D ({data.ndim}D). This may cause issues.")
            
        return data
    except Exception as e:
        logger.error(f"Error loading NPY file {file_path}: {e}")
        raise
