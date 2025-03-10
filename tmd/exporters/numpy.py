"""
Functions for exporting TMD height maps to NumPy array formats (.npy, .npz).
"""
import numpy as np
import os

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
    # Ensure output has .npy extension
    if not filename.lower().endswith('.npy') and not compress:
        filename += '.npy'
    elif not filename.lower().endswith('.npz') and compress:
        filename += '.npz'
    
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
    if not filename.lower().endswith('.npz'):
        filename += '.npz'
    
    # Extract metadata to save alongside height map
    export_dict = {}
    
    # Add height map
    if 'height_map' in data_dict:
        export_dict['height_map'] = data_dict['height_map']
    
    # Add metadata as separate arrays
    for key, value in data_dict.items():
        if key != 'height_map':
            # Convert string metadata to arrays if needed
            if isinstance(value, str):
                export_dict[key] = np.array([value])
            elif isinstance(value, (int, float)):
                export_dict[key] = np.array([value])
            else:
                export_dict[key] = np.array(value)
    
    # Save to NPZ file
    if compress:
        np.savez_compressed(filename, **export_dict)
    else:
        np.savez(filename, **export_dict)
    
    print(f"TMD data saved to {'compressed ' if compress else ''}NPZ file: {filename}")
    return filename

def export_metadata_txt(data_dict, filename="tmd_metadata.txt"):
    """
    Exports TMD metadata to a human-readable text file.
    
    Args:
        data_dict: Dictionary containing TMD data
        filename: Name of the output text file
        
    Returns:
        Path to the saved file
    """
    with open(filename, 'w') as f:
        f.write("TMD File Metadata\n")
        f.write("================\n\n")
        
        # Write metadata values
        for key, value in data_dict.items():
            if key != 'height_map':  # Skip the height map
                f.write(f"{key}: {value}\n")
        
        # Write height map statistics
        if 'height_map' in data_dict:
            height_map = data_dict['height_map']
            f.write("\nHeight Map Statistics\n")
            f.write("====================\n")
            f.write(f"Shape: {height_map.shape}\n")
            f.write(f"Min: {height_map.min()}\n")
            f.write(f"Max: {height_map.max()}\n")
            f.write(f"Mean: {height_map.mean()}\n")
            f.write(f"Std Dev: {height_map.std()}\n")
    
    print(f"TMD metadata saved to text file: {filename}")
    return filename
