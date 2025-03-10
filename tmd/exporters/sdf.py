"""
Functions for exporting TMD height maps as signed distance fields (SDFs).
"""
import numpy as np
import struct
import os
from typing import Tuple, Optional, Union, Dict, Any


def export_to_sdf(height_map: np.ndarray, 
                 filename: str, 
                 scaling_factor: float = 1.0,
                 invert: bool = False,
                 metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    Export a height map as a signed distance field (SDF) file.
    
    Args:
        height_map: 2D numpy array of height values
        filename: Name of the output SDF file
        scaling_factor: Factor to scale height values
        invert: If True, invert the sign (positive inside, negative outside)
        metadata: Optional dictionary of metadata to include in the file
        
    Returns:
        Path to the created SDF file
    """
    # Ensure the filename has .sdf extension
    if not filename.lower().endswith('.sdf'):
        filename += '.sdf'
        
    # Convert height map to SDF
    sdf_data = height_map.copy() * scaling_factor
    if invert:
        sdf_data = -sdf_data
    
    # Get dimensions
    height, width = sdf_data.shape
    
    # Write SDF file
    with open(filename, 'wb') as f:
        # Write magic number and format version
        f.write(b'SDF1')
        
        # Write dimensions
        f.write(struct.pack('<II', width, height))
        
        # Write metadata size
        metadata_str = ""
        if metadata:
            for key, value in metadata.items():
                metadata_str += f"{key}={value};"
        
        metadata_bytes = metadata_str.encode('utf-8')
        f.write(struct.pack('<I', len(metadata_bytes)))
        
        # Write metadata
        if metadata_bytes:
            f.write(metadata_bytes)
        
        # Write SDF data as float32
        f.write(sdf_data.astype(np.float32).tobytes())
    
    print(f"SDF file saved to {filename}")
    return filename


def read_sdf_file(filename: str) -> Tuple[np.ndarray, Dict[str, str]]:
    """
    Read an SDF file and return the data and metadata.
    
    Args:
        filename: Path to the SDF file
        
    Returns:
        Tuple of (sdf_data, metadata)
    """
    with open(filename, 'rb') as f:
        # Read magic number and check
        magic = f.read(4)
        if magic != b'SDF1':
            raise ValueError(f"Not a valid SDF file: {filename}")
        
        # Read dimensions
        width, height = struct.unpack('<II', f.read(8))
        
        # Read metadata size
        metadata_size = struct.unpack('<I', f.read(4))[0]
        
        # Read metadata
        metadata = {}
        if metadata_size > 0:
            metadata_str = f.read(metadata_size).decode('utf-8')
            for item in metadata_str.split(';'):
                if '=' in item:
                    key, value = item.split('=', 1)
                    metadata[key] = value
        
        # Read SDF data
        data_bytes = f.read(width * height * 4)  # 4 bytes per float32
        sdf_data = np.frombuffer(data_bytes, dtype=np.float32).reshape((height, width))
    
    return sdf_data, metadata
