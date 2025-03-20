"""
metadata.py

This module provides functions for computing statistics on TMD height maps
and exporting metadata to a text file.
"""

from typing import Any, Dict

import numpy as np


def compute_stats(height_map: np.ndarray) -> Dict[str, Any]:
    """
    Calculate statistics for a given height map.

    Args:
        height_map: 2D numpy array of height values.

    Returns:
        Dictionary containing statistics (min, max, mean, median, std, shape, etc.).
    """
    stats = {
        "min": float(height_map.min()),
        "max": float(height_map.max()),
        "mean": float(height_map.mean()),
        "median": float(np.median(height_map)),
        "std": float(height_map.std()),
        "shape": height_map.shape,
        "non_nan": int(np.count_nonzero(~np.isnan(height_map))),
        "nan_count": int(np.count_nonzero(np.isnan(height_map))),
    }
    return stats


def export_metadata(metadata: Dict[str, Any], stats: Dict[str, Any], output_path: str) -> str:
    """
    Export metadata and height map statistics to a text file.

    Args:
        metadata: Dictionary containing metadata (excluding the height map).
        stats: Dictionary containing computed statistics.
        output_path: File path to save the metadata.

    Returns:
        The output path to the saved metadata file.
    """
    with open(output_path, "w") as f:
        f.write(f"TMD File: {metadata.get('file_path', 'N/A')}\n")
        f.write("=" * 80 + "\n\n")
        for key, value in metadata.items():
            if key != "file_path":
                f.write(f"{key}: {value}\n")
        f.write("\nHeight Map Statistics\n")
        f.write("-" * 20 + "\n")
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    return output_path


def export_metadata_txt(data_dict, filename="tmd_metadata.txt"):
    """
    Exports TMD metadata to a human-readable text file.

    Args:
        data_dict: Dictionary containing TMD data
        filename: Name of the output text file

    Returns:
        Path to the saved file
    """
    with open(filename, "w") as f:
        f.write("TMD File Metadata\n")
        f.write("================\n\n")

        # Write metadata values
        for key, value in data_dict.items():
            if key != "height_map":  # Skip the height map
                f.write(f"{key}: {value}\n")

        # Write height map statistics
        if "height_map" in data_dict:
            height_map = data_dict["height_map"]
            f.write("\nHeight Map Statistics\n")
            f.write("====================\n")
            f.write(f"Shape: {height_map.shape}\n")
            f.write(f"Min: {height_map.min()}\n")
            f.write(f"Max: {height_map.max()}\n")
            f.write(f"Mean: {height_map.mean()}\n")
            f.write(f"Std Dev: {height_map.std()}\n")

    print(f"TMD metadata saved to text file: {filename}")
    return filename
