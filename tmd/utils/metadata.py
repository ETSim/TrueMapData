"""
metadata.py

This module provides functions for computing statistics on TMD height maps
and exporting metadata to a text file.
"""

import numpy as np
from typing import Dict, Any

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
