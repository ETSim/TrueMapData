import numpy as np
import os
import struct

def export_heightmap_to_sdf(heightmap, output_file, scale=1.0, offset=0.0):
    """
    Export a height map to SDF (Signed Distance Field) format.
    
    Parameters:
    -----------
    heightmap : numpy.ndarray
        2D array containing height values
    output_file : str
        Path to the output SDF file
    scale : float, optional
        Scale factor for the height values
    offset : float, optional
        Offset value for the height values
    
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
        
        # Create SDF data
        # For a basic SDF from height map, we'll consider each height value
        # as a distance from a reference plane
        with open(output_file, 'wb') as f:
            # Write header (format: magic number, version, dimensions)
            f.write(struct.pack('4siii', b'SDF1', 1, width, height))
            
            # Write the SDF data
            for y in range(height):
                for x in range(width):
                    f.write(struct.pack('f', scaled_heights[y, x]))
        
        print(f"SDF file exported successfully to {output_file}")
        return True
        
    except Exception as e:
        print(f"Error exporting SDF file: {str(e)}")
        return False