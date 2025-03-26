""".

NPZ format export/import for TMD data.

This module provides functions to export TMD data (heightmap and metadata) to the NumPy .npz format,
which is a compressed archive format for storing multiple NumPy arrays.

Examples:
    >>> data = {"height_map": np.random.rand(100, 100), "version": "1.0"}
    >>> export_to_npz(data, "terrain_data.npz")
    >>> loaded_data = load_from_npz("terrain_data.npz")
"""

import os
import json
import logging
from typing import Dict, Any, Union, Optional
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


def export_to_npz(data: Dict[str, Any], output_path: str, compress: bool = True) -> str:
    """.

    Export TMD data to NumPy .npz format.

    Args:
        data: Dictionary containing height map and metadata
        output_path: Path to save the .npz file
        compress: Whether to use compression (default: True)

    Returns:
        Path to the saved file

    Raises:
        TypeError: If data is not a dictionary
        ValueError: If data dictionary doesn't contain 'height_map'
        OSError: If there's an error creating the output directory or saving the file

    Examples:
        >>> data = {"height_map": np.random.rand(100, 100), "version": "1.0"}
        >>> export_to_npz(data, "terrain_data.npz")
        'terrain_data.npz'
    """
    # Ensure output directory exists
    output_path = os.path.abspath(output_path)
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create output directory: {e}")
        raise

    # Check if data is a dictionary
    if not isinstance(data, dict):
        raise TypeError("Data must be a dictionary")

    # Extract height map
    if "height_map" not in data:
        raise ValueError("Data dictionary must contain 'height_map' key")
    
    if not isinstance(data["height_map"], np.ndarray):
        raise TypeError("Height map must be a NumPy array")

    # Create a copy of the data without the height map for separate storage
    metadata = {k: v for k, v in data.items() if k != "height_map"}

    # Convert non-array types to arrays or strings for npz compatibility
    sanitized_metadata = {}
    for key, value in metadata.items():
        if isinstance(value, (np.ndarray, str, int, float, bool)) and value is not None:
            sanitized_metadata[key] = value
        else:
            sanitized_metadata[key] = str(value)

    # Add metadata as a string for easier recovery
    try:
        metadata_str = json.dumps(sanitized_metadata, default=str)
    except TypeError as e:
        logger.warning(f"Failed to JSON encode metadata, using string representation: {e}")
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


def load_from_npz(file_path: str) -> Dict[str, Any]:
    """.

    Load TMD data from a .npz file.

    Args:
        file_path: Path to the .npz file

    Returns:
        Dictionary containing height map and metadata

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file is not a valid .npz file or doesn't contain required data

    Examples:
        >>> data = load_from_npz("terrain_data.npz")
        >>> height_map = data["height_map"]
        >>> version = data.get("version")
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        # Load the .npz file
        npz_data = np.load(file_path, allow_pickle=True)
        
        # Check if required keys exist
        if "height_map" not in npz_data:
            raise ValueError("NPZ file does not contain 'height_map' data")

        # Extract height map
        height_map = npz_data["height_map"]
        
        # Validate height map
        if not isinstance(height_map, np.ndarray):
            raise ValueError("Height map data is not a valid NumPy array")

        # Extract metadata
        result = {"height_map": height_map}
        
        try:
            if "metadata" in npz_data:
                metadata_str = str(npz_data["metadata"])
                # Remove any leading/trailing quotes that might be in the string representation
                metadata_str = metadata_str.strip("'\"")
                metadata = json.loads(metadata_str)
                result.update(metadata)
        except json.JSONDecodeError as e:
            logger.warning(f"Could not parse metadata from NPZ file: {e}. Using empty metadata.")
        except Exception as e:
            logger.warning(f"Error processing metadata: {e}. Using height map only.")

        return result
    except Exception as e:
        logger.error(f"Error loading NPZ file: {e}")
        raise
