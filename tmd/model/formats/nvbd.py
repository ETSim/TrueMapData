"""
NVBD (NVIDIA Binary Data) exporter implementation for TMD.

This module provides the NVBDExporter class and related functions for exporting
height maps to NVBD format, which is optimized for use in NVIDIA applications.
"""

import os
import struct
import numpy as np
import logging
from typing import Optional, Dict, Any, Tuple, Union, List

from ..base import ModelExporter, ExportConfig
from ..utils import validate_heightmap, ensure_directory_exists
from ..utils.heightmap import calculate_heightmap_normals
from ..registry import register_exporter

# Set up logging
logger = logging.getLogger(__name__)


@register_exporter
class NVBDExporter(ModelExporter):
    """Exporter for NVIDIA Binary Data (NVBD) format."""
    
    # Class attributes
    format_name = "NVIDIA Binary Data (NVBD)"
    file_extensions = ["nvbd"]
    binary_supported = True
    
    @classmethod
    def export(cls, 
               height_map: np.ndarray, 
               filename: str, 
               config: ExportConfig) -> Optional[str]:
        """
        Export a heightmap to NVBD format.
        
        Args:
            height_map: 2D numpy array of height values
            filename: Output filename
            config: Export configuration
            
        Returns:
            Path to the created file if successful, None otherwise
        """
        # Validate input
        if not validate_heightmap(height_map):
            logger.error("Invalid height map: empty, None, or too small")
            return None
        
        # Get NVBD-specific parameters from config
        chunk_size = config.extra.get('chunk_size', 16)
        include_normals = config.extra.get('include_normals', True)
        watertight = config.extra.get('watertight', True)
        
        # Ensure filename has correct extension
        filename = cls.ensure_extension(filename)
            
        # Ensure output directory exists
        if not ensure_directory_exists(filename):
            logger.error(f"Failed to create directory for {filename}")
            return None
        
        try:
            # Export to NVBD format (doesn't use mesh, works directly with heightmap)
            return export_heightmap_to_nvbd(
                height_map=height_map,
                filename=filename,
                scale=config.z_scale,
                offset=config.base_height,
                chunk_size=chunk_size,
                include_normals=include_normals,
                watertight=watertight
            )
            
        except Exception as e:
            logger.error(f"Error exporting NVBD: {e}")
            import traceback
            traceback.print_exc()
            return None


def export_heightmap_to_nvbd(
    height_map: np.ndarray,
    filename: str = "output.nvbd",
    scale: float = 1.0,
    offset: float = 0.0,
    chunk_size: int = 16,
    include_normals: bool = True,
    watertight: bool = True
) -> Optional[str]:
    """
    Export a height map to NVBD (NVIDIA Binary Data) format.
    
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
    # Check for valid chunk size
    if chunk_size <= 0:
        logger.error("Chunk size must be positive")
        return None
    
    try:
        # Ensure filename has correct extension
        if not filename.lower().endswith('.nvbd'):
            filename = os.path.splitext(filename)[0] + '.nvbd'
            
        # Ensure output directory exists
        if not ensure_directory_exists(filename):
            logger.error(f"Failed to create directory for {filename}")
            return None
        
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


def binary_serialize_heightmap(
    height_map: np.ndarray,
    scale: float = 1.0,
    include_normals: bool = True
) -> bytes:
    """
    Serialize a heightmap to NVBD binary format without writing to disk.
    
    This is useful for in-memory serialization or for testing.
    
    Args:
        height_map: 2D numpy array of height values
        scale: Scale factor for height values
        include_normals: Whether to include normal vectors
        
    Returns:
        Binary data as bytes
    """
    # Validate input
    if not validate_heightmap(height_map):
        raise ValueError("Invalid height map: empty, None, or too small")
    
    # Default chunk size for serialization
    chunk_size = 16
    
    # Get dimensions
    height, width = height_map.shape
    
    # Calculate min/max height
    min_height = np.min(height_map)
    max_height = np.max(height_map)
    
    # Apply scale
    scaled_min = min_height * scale
    scaled_max = max_height * scale
    
    # Create binary data buffer
    data = bytearray()
    
    # Write magic header "NVBD"
    data.extend(b'NVBD')
    
    # Write version (1.0)
    data.extend(struct.pack('<f', 1.0))
    
    # Write dimensions
    data.extend(struct.pack('<II', width, height))
    
    # Write chunk size
    data.extend(struct.pack('<I', chunk_size))
    
    # Write min/max heights
    data.extend(struct.pack('<ff', scaled_min, scaled_max))
    
    # Calculate chunks
    num_chunks_x = (width + chunk_size - 1) // chunk_size
    num_chunks_y = (height + chunk_size - 1) // chunk_size
    num_chunks = num_chunks_x * num_chunks_y
    data.extend(struct.pack('<I', num_chunks))
    
    # Write chunk data
    for y in range(num_chunks_y):
        for x in range(num_chunks_x):
            # Chunk ID
            chunk_id = y * num_chunks_x + x + 1  # 1-based index
            data.extend(struct.pack('<I', chunk_id))
            
            # Start coordinates
            start_x = x * chunk_size
            start_y = y * chunk_size
            data.extend(struct.pack('<II', start_x, start_y))
            
            # End coordinates (clamped to heightmap dimensions)
            end_x = min(start_x + chunk_size, width) - 1
            end_y = min(start_y + chunk_size, height) - 1
            data.extend(struct.pack('<II', end_x, end_y))
            
            # Add chunk flags (always watertight for serialization)
            data.extend(struct.pack('<I', 1))
    
    # If normals are included, add them after the chunk data
    if include_normals:
        normals = calculate_heightmap_normals(height_map)
        
        # Write normal count
        data.extend(struct.pack('<I', height * width))
        
        # Write normals as float triplets
        for y in range(height):
            for x in range(width):
                normal = normals[y, x]
                data.extend(struct.pack('<fff', normal[0], normal[1], normal[2]))
    
    return bytes(data)