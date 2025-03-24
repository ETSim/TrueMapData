""".

NPZ format export/import for TMD data.

This module provides functions to export TMD data (heightmap and metadata) to the NumPy .npz format,
which is a compressed archive format for storing multiple NumPy arrays.
"""

import os
import json
import logging
import numpy as np

logger = logging.getLogger(__name__)

def export_to_npz(data, output_path, compress=True):
    """.

    Export TMD data to NumPy .npz format.

    Args:
        data: Dictionary containing height map and metadata
        output_path: Path to save the .npz file
        compress: Whether to use compression (default: True)

    Returns:
        Path to the saved file
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Check if data is a dictionary
    if not isinstance(data, dict):
        raise TypeError("Data must be a dictionary")

    # Extract height map
    if "height_map" not in data:
        raise ValueError("Data dictionary must contain 'height_map'")

    # Create a copy of the data without the height map for separate storage
    metadata = {k: v for k, v in data.items() if k != "height_map"}

    # Convert non-array types to arrays or strings for npz compatibility
    for key, value in metadata.items():
        if not isinstance(value, (np.ndarray, str, int, float, bool)) or value is None:
            metadata[key] = str(value)

    # Add metadata as a string for easier recovery
    try:
        metadata_str = json.dumps(metadata, default=str)
    except TypeError:
        # Fallback if json conversion fails
        metadata_str = str(metadata)

    # Save to .npz file
    try:
        if compress:
            np.savez_compressed(output_path, height_map=data["height_map"], metadata=metadata_str)
        else:
            np.savez(output_path, height_map=data["height_map"], metadata=metadata_str)

        logger.info(f"TMD data exported to {output_path}" + (" (compressed)" if compress else ""))
        return output_path
    except Exception as e:
        logger.error(f"Error saving NPZ file: {e}")
        raise

def load_from_npz(file_path):
    """.

    Load TMD data from a .npz file.

    Args:
        file_path: Path to the .npz file

    Returns:
        Dictionary containing height map and metadata
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        # Load the .npz file
        npz_data = np.load(file_path)

        # Extract height map
        height_map = npz_data["height_map"]

        # Extract metadata
        try:
            metadata_str = str(npz_data["metadata"])
            metadata = json.loads(metadata_str)
        except (KeyError, json.JSONDecodeError):
            # Fallback if metadata is missing or invalid
            logger.warning("Could not parse metadata from NPZ file. Using empty metadata.")
            metadata = {}

        # Combine into a single dictionary
        result = metadata.copy()
        result["height_map"] = height_map

        return result
    except Exception as e:
        logger.error(f"Error loading NPZ file: {e}")
        raise
