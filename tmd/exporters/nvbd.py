import numpy as np
import os
import struct

def export_heightmap_to_nvbd(heightmap, output_file, scale=1.0, offset=0.0, chunk_size=16):
    """
    Export a height map to NVBD format (NVIDIA Blast Destruction).
    
    Parameters:
    -----------
    heightmap : numpy.ndarray
        2D array containing height values
    output_file : str
        Path to the output NVBD file
    scale : float, optional
        Scale factor for the height values
    offset : float, optional
        Offset value for the height values
    chunk_size : int, optional
        Size of chunks for destruction simulation
    
    Returns:
    --------
    bool
        True if export was successful, False otherwise
        
    Raises:
    -------
    ValueError
        If output_file is a directory
    """
    # Check if output_file is a directory - this check needs to happen outside the try-except
    if os.path.isdir(output_file):
        raise ValueError(f"Output path is a directory: {output_file}")
    
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Apply scale and offset to the height map
        scaled_heights = heightmap * scale + offset
        
        # Get dimensions
        height, width = heightmap.shape
        
        # Calculate number of chunks
        chunks_x = (width + chunk_size - 1) // chunk_size
        chunks_y = (height + chunk_size - 1) // chunk_size
        
        with open(output_file, 'wb') as f:
            # Write header (format: magic number, version, dimensions, chunk size)
            f.write(struct.pack('4siiii', b'NVBD', 1, width, height, chunk_size))
            
            # Write chunk data
            for cy in range(chunks_y):
                for cx in range(chunks_x):
                    # Calculate chunk boundaries
                    x_start = cx * chunk_size
                    y_start = cy * chunk_size
                    x_end = min(x_start + chunk_size, width)
                    y_end = min(y_start + chunk_size, height)
                    
                    # Extract chunk data
                    chunk_data = scaled_heights[y_start:y_end, x_start:x_end]
                    
                    # Write chunk metadata
                    f.write(struct.pack('iiii', cx, cy, x_end - x_start, y_end - y_start))
                    
                    # Write chunk height values
                    for y in range(y_start, y_end):
                        for x in range(x_start, x_end):
                            f.write(struct.pack('f', scaled_heights[y, x]))
        
        print(f"NVBD file exported successfully to {output_file}")
        return True
        
    except Exception as e:
        print(f"Error exporting NVBD file: {str(e)}")
        return False