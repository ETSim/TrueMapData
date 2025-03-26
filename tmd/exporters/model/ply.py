"""
PLY exporter module for height maps.

This module provides functions for converting height maps to PLY files,
which are commonly used for storing 3D scanned data.
"""

import os
import numpy as np
import logging
import struct
from typing import Optional, Tuple, List, Dict, Any, Union

from .base import create_mesh_from_heightmap
from .mesh_utils import (
    calculate_vertex_normals, 
    validate_heightmap,
    ensure_directory_exists,
    generate_uv_coordinates
)

# Set up logging
logger = logging.getLogger(__name__)


def convert_heightmap_to_ply(
    height_map: np.ndarray,
    filename: str = "output.ply",
    x_offset: float = 0,
    y_offset: float = 0,
    x_length: float = 1,
    y_length: float = 1,
    z_scale: float = 1,
    base_height: float = 0.0,
    calculate_normals: bool = True,
    add_color: bool = True,
    color_map: str = 'terrain',
    **kwargs
) -> Optional[str]:
    """
    Convert a height map to PLY format.
    
    Args:
        height_map: 2D numpy array of height values
        filename: Output filename
        x_offset: X-axis offset for the model
        y_offset: Y-axis offset for the model
        x_length: Physical length in X direction
        y_length: Physical length in Y direction
        z_scale: Scale factor for Z-axis values
        base_height: Height of solid base to add below the model
        calculate_normals: Whether to calculate vertex normals
        add_color: Whether to add color based on height values
        color_map: Name of the colormap to use for colors
        **kwargs: Additional options
        
    Returns:
        Path to the created file or None if failed
    """
    # Validate input
    if not validate_heightmap(height_map):
        logger.error("Invalid height map: empty, None, or too small")
        return None
        
    # Ensure filename has correct extension
    if not filename.lower().endswith('.ply'):
        filename = f"{os.path.splitext(filename)[0]}.ply"
        
    # Ensure output directory exists
    if not ensure_directory_exists(filename):
        return None
    
    try:
        # Create mesh from heightmap
        vertices, faces = create_mesh_from_heightmap(
            height_map, 
            x_offset, 
            y_offset, 
            x_length, 
            y_length, 
            z_scale, 
            base_height
        )
        
        if not vertices or not faces:
            logger.error("Failed to generate mesh from heightmap")
            return None
        
        # Convert to numpy arrays for easier processing
        vertices_array = np.array(vertices)
        faces_array = np.array(faces)
        
        # Calculate normals if requested
        normals = None
        if calculate_normals:
            normals = calculate_vertex_normals(vertices_array, faces_array)
        
        # Calculate vertex colors if requested
        colors = None
        if add_color:
            colors = _generate_vertex_colors(vertices_array, height_map, color_map)
        
        # Write binary PLY file
        with open(filename, 'wb') as f:
            _write_binary_ply(f, vertices_array, faces_array, normals, colors)
        
        logger.info(f"Exported PLY file to {filename}")
        return filename
        
    except Exception as e:
        logger.error(f"Error exporting PLY: {e}")
        import traceback
        traceback.print_exc()
        return None


def _write_binary_ply(
    file_obj,
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: Optional[np.ndarray] = None,
    colors: Optional[np.ndarray] = None
) -> None:
    """
    Write mesh data as binary PLY.
    
    Args:
        file_obj: File object to write to
        vertices: Nx3 array of vertex positions
        faces: Mx3 array of face indices
        normals: Nx3 array of vertex normals (optional)
        colors: Nx3 array of vertex colors (optional)
    """
    # Write header as ASCII (binary data starts after header)
    file_obj.write(b"ply\n")
    file_obj.write(b"format binary_little_endian 1.0\n")
    file_obj.write(f"element vertex {len(vertices)}\n".encode())
    file_obj.write(b"property float x\n")
    file_obj.write(b"property float y\n")
    file_obj.write(b"property float z\n")
    
    # Add normal properties if provided
    if normals is not None:
        file_obj.write(b"property float nx\n")
        file_obj.write(b"property float ny\n")
        file_obj.write(b"property float nz\n")
    
    # Add color properties if provided
    if colors is not None:
        file_obj.write(b"property uchar red\n")
        file_obj.write(b"property uchar green\n")
        file_obj.write(b"property uchar blue\n")
    
    # Define face element
    file_obj.write(f"element face {len(faces)}\n".encode())
    file_obj.write(b"property list uchar int vertex_indices\n")
    
    # End of header
    file_obj.write(b"end_header\n")
    
    # Write vertex data
    for i in range(len(vertices)):
        # Write position
        file_obj.write(struct.pack('<fff', vertices[i, 0], vertices[i, 1], vertices[i, 2]))
        
        # Write normals if present
        if normals is not None:
            file_obj.write(struct.pack('<fff', normals[i, 0], normals[i, 1], normals[i, 2]))
        
        # Write colors if present
        if colors is not None:
            file_obj.write(struct.pack('<BBB', colors[i, 0], colors[i, 1], colors[i, 2]))
    
    # Write face data
    for face in faces:
        file_obj.write(struct.pack('<BIII', 3, face[0], face[1], face[2]))


def _generate_vertex_colors(
    vertices: np.ndarray, 
    height_map: np.ndarray, 
    color_map: str = 'terrain'
) -> np.ndarray:
    """
    Generate vertex colors based on height values.
    
    Args:
        vertices: Nx3 array of vertex positions
        height_map: 2D height map array
        color_map: Name of the colormap to use
        
    Returns:
        Nx3 array of RGB colors (0-255)
    """
    try:
        from matplotlib import cm
        
        # Get z range to normalize heights
        z_min = np.min(vertices[:, 2])
        z_max = np.max(vertices[:, 2])
        
        # Avoid division by zero
        z_range = z_max - z_min
        if z_range < 1e-10:
            z_range = 1.0
            
        # Normalize z values to [0, 1]
        normalized_z = (vertices[:, 2] - z_min) / z_range
        
        # Apply colormap
        cmap = cm.get_cmap(color_map)
        rgba_colors = cmap(normalized_z)
        
        # Convert to 8-bit RGB
        rgb_colors = (rgba_colors[:, :3] * 255).astype(np.uint8)
        
        return rgb_colors
        
    except ImportError as e:
        logger.warning(f"Matplotlib not available, using grayscale colors: {e}")
        
        # Fall back to grayscale
        z_min = np.min(vertices[:, 2])
        z_max = np.max(vertices[:, 2])
        z_range = max(z_max - z_min, 1e-10)
        
        normalized_z = (vertices[:, 2] - z_min) / z_range
        grayscale = (normalized_z * 255).astype(np.uint8)
        
        # Duplicate for RGB channels
        rgb_colors = np.column_stack([grayscale, grayscale, grayscale])
        return rgb_colors
