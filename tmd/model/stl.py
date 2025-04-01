"""
STL exporter module for height maps.

This module provides functions for converting height maps to STL files,
which are widely used for 3D printing and manufacturing.
"""

import os
import numpy as np
import struct
import logging
from typing import Optional, List, Tuple, Union

from .base import create_mesh_from_heightmap
from .mesh_utils import (
    calculate_vertex_normals,
    validate_heightmap,
    ensure_directory_exists
)

# Set up logging
logger = logging.getLogger(__name__)


def convert_heightmap_to_stl(
    height_map: np.ndarray,
    filename: str = "output.stl",
    x_offset: float = 0,
    y_offset: float = 0,
    x_length: float = 1,
    y_length: float = 1,
    z_scale: float = 1,
    base_height: float = 0.0,
    ascii_format: bool = False,
    **kwargs
) -> Optional[str]:
    """
    Convert a height map to STL format.
    
    Args:
        height_map: 2D numpy array of height values
        filename: Output filename
        x_offset: X-axis offset for the model
        y_offset: Y-axis offset for the model
        x_length: Physical length in X direction
        y_length: Physical length in Y direction
        z_scale: Scale factor for Z-axis values
        base_height: Height of solid base to add below the model
        ascii_format: Whether to save in ASCII STL format
        **kwargs: Additional options
        
    Returns:
        Path to the created file or None if failed
    """
    # Validate input
    if not validate_heightmap(height_map):
        logger.error("Invalid height map: empty, None, or too small")
        return None

    # Ensure filename has correct extension
    if not filename.lower().endswith('.stl'):
        filename = f"{os.path.splitext(filename)[0]}.stl"
            
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
            
        # Ensure the mesh is watertight
        vertices, faces = _ensure_watertight_mesh(vertices, faces, base_height)
        
        # Convert to numpy arrays for easier processing
        vertices_array = np.array(vertices, dtype=np.float32)
        faces_array = np.array(faces, dtype=np.int32)
        
        # Write to STL file
        if ascii_format:
            write_ascii_stl(vertices_array, faces_array, filename)
        else:
            write_binary_stl(vertices_array, faces_array, filename)
            
        logger.info(f"Exported STL file to {filename}")
        return filename
        
    except Exception as e:
        logger.error(f"Error exporting to STL: {e}")
        import traceback
        traceback.print_exc()
        return None


def write_binary_stl(vertices: np.ndarray, faces: np.ndarray, filename: str) -> None:
    """
    Write mesh data to a binary STL file.
    
    Args:
        vertices: Array of vertex coordinates
        faces: Array of face indices
        filename: Output filename
    """
    with open(filename, 'wb') as f:
        # Write STL header (80 bytes)
        f.write(b'TMD STL Exporter' + b' ' * (80 - 15))
        
        # Write number of triangles (4 bytes)
        f.write(struct.pack('<I', len(faces)))
        
        # Calculate normals
        normals = calculate_face_normals(vertices, faces)
        
        # Write each triangle
        for i, face in enumerate(faces):
            # Normal vector
            f.write(struct.pack('<fff', *normals[i]))
            
            # Vertices (3 points)
            for idx in face:
                f.write(struct.pack('<fff', *vertices[idx]))
            
            # Attribute byte count (2 bytes, usually zero)
            f.write(struct.pack('<H', 0))


def write_ascii_stl(vertices: np.ndarray, faces: np.ndarray, filename: str) -> None:
    """
    Write mesh data to an ASCII STL file.
    
    Args:
        vertices: Array of vertex coordinates
        faces: Array of face indices
        filename: Output filename
    """
    with open(filename, 'w') as f:
        # Write header
        f.write("solid TMDExport\n")
        
        # Calculate normals
        normals = calculate_face_normals(vertices, faces)
        
        # Write each triangle
        for i, face in enumerate(faces):
            # Start facet
            f.write(f"  facet normal {normals[i][0]} {normals[i][1]} {normals[i][2]}\n")
            f.write("    outer loop\n")
            
            # Write vertices
            for idx in face:
                v = vertices[idx]
                f.write(f"      vertex {v[0]} {v[1]} {v[2]}\n")
            
            # End facet
            f.write("    endloop\n")
            f.write("  endfacet\n")
        
        # Write footer
        f.write("endsolid TMDExport\n")


def calculate_face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Calculate normal vectors for each face.
    
    Args:
        vertices: Array of vertex coordinates
        faces: Array of face indices
        
    Returns:
        Array of normal vectors
    """
    normals = np.zeros((len(faces), 3))
    
    for i, face in enumerate(faces):
        # Get vertices of this face
        v0 = vertices[face[0]]
        v1 = vertices[face[1]]
        v2 = vertices[face[2]]
        
        # Calculate edges
        edge1 = v1 - v0
        edge2 = v2 - v0
        
        # Calculate normal using cross product
        normal = np.cross(edge1, edge2)
        
        # Normalize
        norm = np.linalg.norm(normal)
        if norm > 0:
            normal = normal / norm
            
        normals[i] = normal
    
    return normals


def export_stl(height_map: np.ndarray, output_file: str, **kwargs) -> Optional[str]:
    """
    Export a height map to STL format.
    
    Args:
        height_map: 2D numpy array of height values
        output_file: Output filename
        **kwargs: Additional options passed to convert_heightmap_to_stl
        
    Returns:
        Path to the created file or None if failed
    """
    return convert_heightmap_to_stl(
        height_map=height_map,
        filename=output_file,
        **kwargs
    )


def _ensure_watertight_mesh(vertices, faces, base_height=0.0):
    """
    Ensure that a mesh is watertight by adding a base if necessary.
    
    Args:
        vertices: List of vertex coordinates
        faces: List of triangular faces
        base_height: Height of base to add
        
    Returns:
        Tuple of (vertices, faces) with base added
    """
    # Import here to avoid circular import
    from .mesh_utils import ensure_watertight_mesh
    
    # If base_height is specified, use ensure_watertight_mesh
    if base_height > 0:
        return ensure_watertight_mesh(
            vertices=np.array(vertices),
            faces=np.array(faces),
            min_base_height=base_height
        )
        
    # Otherwise return the original mesh
    return np.array(vertices), np.array(faces)
