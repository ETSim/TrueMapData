import numpy as np
import os
import pickle
import gzip

def export_to_npy(height_map, filename="height_map.npy", compress=False):
    """
    Exports the height map to a NumPy .npy file.

    Args:
        height_map: 2D numpy array of height values
        filename: Name of the output .npy file
        compress: Whether to use compression (uses savez_compressed if True)

    Returns:
        Path to the saved file
    """
    # Ensure output has .npy extension if not compressing, or .npz if compressing
    if not filename.lower().endswith(".npy") and not compress:
        filename += ".npy"
    elif not filename.lower().endswith(".npz") and compress:
        filename += ".npz"

    if compress:
        np.savez_compressed(filename, height_map=height_map)
        print(f"Height map saved to compressed NPZ file: {filename}")
    else:
        np.save(filename, height_map)
        print(f"Height map saved to NPY file: {filename}")

    return filename


def export_to_npz(data_dict, filename="tmd_data.npz", compress=True):
    """
    Exports TMD data to a NumPy .npz file with multiple arrays.

    Args:
        data_dict: Dictionary containing TMD data (height_map, metadata, etc.)
        filename: Name of the output .npz file
        compress: Whether to use compression

    Returns:
        Path to the saved file
    """
    # Ensure output has .npz extension
    if not filename.lower().endswith(".npz"):
        filename += ".npz"

    # Prepare export dictionary
    export_dict = {}

    # Add height map
    if "height_map" in data_dict:
        export_dict["height_map"] = data_dict["height_map"]

    # Add metadata as separate arrays
    for key, value in data_dict.items():
        if key != "height_map":
            # Convert string metadata to arrays if needed
            if isinstance(value, str):
                export_dict[key] = np.array([value])
            elif isinstance(value, (int, float)):
                export_dict[key] = np.array([value])
            else:
                export_dict[key] = np.array(value)

    # Save to NPZ file with or without compression
    if compress:
        np.savez_compressed(filename, **export_dict)
    else:
        np.savez(filename, **export_dict)

    print(f"TMD data saved to {'compressed ' if compress else ''}NPZ file: {filename}")
    return filename


def export_height_map_with_metadata_pickle_gzip(height_map, metadata, stats=None, images=None, filename="height_map.pkl.gz"):
    """
    Exports the height map and metadata to a compressed pickle file.

    Args:
        height_map: 2D numpy array of height values
        metadata: Dictionary containing metadata
        stats: Dictionary containing height map statistics
        images: List of image paths to include
        filename: Name of the output .pkl.gz file

    Returns:
        Path to the saved file
    """

    # Prepare data dictionary
    data_dict = {
        "height_map": height_map,
        "metadata": metadata,
        "stats": stats,
        "images": images
    }

    with gzip.open(filename, "wb") as f:
        pickle.dump(data_dict, f)

    print(f"Height map and metadata saved to compressed pickle file: {filename}")
    return filename
