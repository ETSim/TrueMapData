"""
Compression exporters for TMD data.

This module provides functions to export TMD data in compressed formats.
"""

import os
import numpy as np
import json


def export_to_npy(data, output_path):
    """
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
    if isinstance(data, dict) and 'height_map' in data:
        height_map = data['height_map']
    else:
        height_map = data
    
    # Check if height_map is a numpy array
    if not isinstance(height_map, np.ndarray):
        raise TypeError("Height map must be a NumPy array")
    
    # Save as .npy file
    np.save(output_path, height_map)
    print(f"Height map exported to {output_path}")
    
    return output_path


def export_to_npz(data, output_path, compress=True):
    """
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
    if 'height_map' not in data:
        raise ValueError("Data dictionary must contain 'height_map'")
    
    # Create a copy of the data without the height map for separate storage
    metadata = {k: v for k, v in data.items() if k != 'height_map'}
    
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
    if compress:
        np.savez_compressed(
            output_path,
            height_map=data['height_map'],
            metadata=metadata_str
        )
    else:
        np.savez(
            output_path,
            height_map=data['height_map'],
            metadata=metadata_str
        )
    
    print(f"TMD data exported to {output_path}" + (" (compressed)" if compress else ""))
    return output_path


def load_from_npy(file_path):
    """
    Load height map data from a .npy file.
    
    Args:
        file_path: Path to the .npy file
        
    Returns:
        NumPy array containing the height map
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    return np.load(file_path)


def load_from_npz(file_path):
    """
    Load TMD data from a .npz file.
    
    Args:
        file_path: Path to the .npz file
        
    Returns:
        Dictionary containing height map and metadata
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Load the .npz file
    npz_data = np.load(file_path)
    
    # Extract height map
    height_map = npz_data['height_map']
    
    # Extract metadata
    try:
        metadata_str = str(npz_data['metadata'])
        metadata = json.loads(metadata_str)
    except (KeyError, json.JSONDecodeError):
        # Fallback if metadata is missing or invalid
        metadata = {}
    
    # Combine into a single dictionary
    result = metadata.copy()
    result['height_map'] = height_map
    
    return result
