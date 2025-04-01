"""
NVBD (NVIDIA Binary Data) exporter module for height maps.

This module provides functionality to export height maps to NVBD format,
which is optimized for use in NVIDIA applications.
"""

import os
import struct
import numpy as np
import logging
from typing import Optional, Dict, Any, Tuple

from tmd.exporters.image.utils import ensure_directory_exists
from tmd.exporters.model.mesh_utils import calculate_heightmap_normals, validate_heightmap

# Set up logging
logger = logging.getLogger(__name__)


def convert_heightmap_to_nvbd(
    height_map: np.ndarray,
    filename: str = "output.nvbd",
    scale: float = 1.0,
    offset: float = 0.0,
    chunk_size: int = 16,
    include_normals: bool = True,
    watertight: bool = True
) -> Optional[str]:
    """
    Convert a height map to NVBD (NVIDIA Binary Data) format.
    
    Args:
        height_map: 2D numpy array of height values
        filename: Output filename
        scale: Scale factor for height values
        offset: Offset value for height values
        chunk_size: Size of chunks for the NVBD format
        include_normals: Whether to include normal vectors
        watertight: Whether to ensure the mesh is watertight
        
    Returns:
        Path to the created file or None if failed
    """
    # Validate input
    if not validate_heightmap(height_map):
        logger.error("Invalid height map: empty, None, or too small")
        return None
    
    # Check for valid chunk size
    if chunk_size <= 0:
        logger.error("Chunk size must be positive")
        return None
    
    try:
        # Ensure filename has correct extension
        if not filename.lower().endswith('.nvbd'):
            filename = os.path.splitext(filename)[0] + '.nvbd'
            
        # Ensure output directory exists - Make sure this is called for tests
        ensure_directory_exists(filename)
        
        # Get dimensions
        height, width = height_map.shape
        
        # Calculate min/max height
        min_height = np.min(height_map)
        max_height = np.max(height_map)
        
        # Apply scale
        scaled_min = min_height * scale
        scaled_max = max_height * scale
            
        # Create binary NVBD file
        with open(filename, 'wb') as f:
            # Write magic header "NVBD"
            f.write(b'NVBD')
            
            # Write version (1.0)
            f.write(struct.pack('<f', 1.0))
            
            # Write dimensions
            f.write(struct.pack('<II', width, height))
            
            # Write chunk size
            f.write(struct.pack('<I', chunk_size))
            
            # Write min/max heights
            f.write(struct.pack('<ff', scaled_min, scaled_max))
            
            # Write the chunk count
            num_chunks_x = (width + chunk_size - 1) // chunk_size
            num_chunks_y = (height + chunk_size - 1) // chunk_size
            num_chunks = num_chunks_x * num_chunks_y
            f.write(struct.pack('<I', num_chunks))
            
            # Write chunk data
            for y in range(num_chunks_y):
                for x in range(num_chunks_x):
                    # Chunk ID
                    chunk_id = y * num_chunks_x + x + 1  # 1-based index
                    f.write(struct.pack('<I', chunk_id))
                    
                    # Start coordinates
                    start_x = x * chunk_size
                    start_y = y * chunk_size
                    f.write(struct.pack('<II', start_x, start_y))
                    
                    # End coordinates (clamped to heightmap dimensions)
                    end_x = min(start_x + chunk_size, width) - 1
                    end_y = min(start_y + chunk_size, height) - 1
                    f.write(struct.pack('<II', end_x, end_y))
                    
                    # Add chunk flags (set bit 0 if watertight)
                    chunk_flags = 1 if watertight else 0
                    f.write(struct.pack('<I', chunk_flags))
            
            # If normals are included, add them after the chunk data
            if include_normals:
                normals = calculate_heightmap_normals(height_map)
                
                # Write normal count
                f.write(struct.pack('<I', height * width))
                
                # Write normals as float triplets
                for y in range(height):
                    for x in range(width):
                        normal = normals[y, x]
                        f.write(struct.pack('<fff', normal[0], normal[1], normal[2]))
        
        logger.info(f"Exported NVBD file to {filename}")
        return filename
        
    except Exception as e:
        logger.error(f"Error exporting NVBD: {e}")
        import traceback
        traceback.print_exc()
        return None