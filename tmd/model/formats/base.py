"""Base functionality for mesh generation from heightmaps."""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging

# Set up logging
logger = logging.getLogger(__name__)

def create_mesh_from_heightmap(
    height_map: np.ndarray,
    x_offset: float = 0,
    y_offset: float = 0,
    x_length: float = 1,
    y_length: float = 1,
    z_scale: float = 1,
    base_height: float = 0.0
) -> Tuple[List[List[float]], List[List[int]]]:
    """
    Create a basic mesh from a heightmap.
    
    Args:
        height_map: 2D numpy array of height values
        x_offset, y_offset: Offset for vertex positions
        x_length, y_length: Physical dimensions of the mesh
        z_scale: Scale factor for height values
        base_height: Height of solid base below model
        
    Returns:
        Tuple of (vertices, faces) where vertices is a list of [x,y,z] coordinates
        and faces is a list of triangle indices.
    """
    try:
        rows, cols = height_map.shape
        
        # Calculate vertex spacing
        dx = x_length / (cols - 1)
        dy = y_length / (rows - 1)
        
        # Create vertices array
        vertices = []
        for i in range(rows):
            for j in range(cols):
                x = j * dx + x_offset
                y = i * dy + y_offset
                z = float(height_map[i, j]) * z_scale
                vertices.append([x, y, z])
        
        # Create faces (triangles)
        faces = []
        for i in range(rows - 1):
            for j in range(cols - 1):
                # Get vertex indices for this quad
                v0 = i * cols + j
                v1 = v0 + 1
                v2 = v0 + cols
                v3 = v2 + 1
                
                # Create two triangles
                faces.append([v0, v2, v1])  # First triangle
                faces.append([v1, v2, v3])  # Second triangle
        
        # Add base if requested
        if base_height > 0:
            vertices, faces = _add_base(vertices, faces, base_height)
        
        return vertices, faces
        
    except Exception as e:
        logger.error(f"Error creating mesh from heightmap: {e}")
        return [], []

def _add_base(
    vertices: List[List[float]],
    faces: List[List[int]],
    base_height: float
) -> Tuple[List[List[float]], List[List[int]]]:
    """Add a solid base to the mesh."""
    try:
        # Get the number of original vertices
        num_orig_vertices = len(vertices)
        
        # Find the boundary vertices (minimum z-coordinate for each x,y position)
        boundary_verts = {}
        for i, v in enumerate(vertices):
            key = (v[0], v[1])
            if key not in boundary_verts or v[2] < vertices[boundary_verts[key]][2]:
                boundary_verts[key] = i
        
        # Create base vertices by duplicating boundary vertices at base_height
        base_vertices = []
        vert_map = {}  # Maps original index to base vertex index
        
        for orig_idx in boundary_verts.values():
            v = vertices[orig_idx]
            new_idx = len(vertices)
            vertices.append([v[0], v[1], v[2] - base_height])
            vert_map[orig_idx] = new_idx
        
        # Create faces for base walls
        for face in faces:
            # Check if this is a boundary face
            boundary_edges = []
            for i in range(3):
                v1 = face[i]
                v2 = face[(i + 1) % 3]
                if v1 in boundary_verts and v2 in boundary_verts:
                    boundary_edges.append((v1, v2))
            
            # Create wall faces for boundary edges
            for edge in boundary_edges:
                v1, v2 = edge
                faces.append([v1, vert_map[v1], v2])
                faces.append([vert_map[v1], vert_map[v2], v2])
        
        # Add bottom face
        bottom_verts = list(vert_map.values())
        if len(bottom_verts) > 2:
            # Simple triangulation - fan from first vertex
            v0 = bottom_verts[0]
            for i in range(1, len(bottom_verts) - 1):
                faces.append([v0, bottom_verts[i+1], bottom_verts[i]])
        
        return vertices, faces
        
    except Exception as e:
        logger.error(f"Error adding base to mesh: {e}")
        return vertices, faces

def validate_mesh(vertices: List[List[float]], faces: List[List[int]]) -> bool:
    """
    Validate a mesh for common errors.
    
    Args:
        vertices: List of vertex coordinates
        faces: List of triangle indices
        
    Returns:
        True if mesh is valid, False otherwise
    """
    try:
        # Check for empty mesh
        if not vertices or not faces:
            logger.error("Empty mesh")
            return False
            
        # Check vertex indices
        num_vertices = len(vertices)
        for face in faces:
            if len(face) != 3:
                logger.error(f"Invalid face: {face}")
                return False
            if any(i < 0 or i >= num_vertices for i in face):
                logger.error(f"Face references invalid vertex: {face}")
                return False
            if len(set(face)) != 3:
                logger.error(f"Degenerate face: {face}")
                return False
                
        return True
        
    except Exception as e:
        logger.error(f"Error validating mesh: {e}")
        return False
