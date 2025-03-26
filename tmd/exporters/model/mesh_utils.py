"""
Utility functions for handling 3D meshes.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
import os

def calculate_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Calculate vertex normals for a mesh.
    
    Args:
        vertices: Nx3 array of vertex positions
        faces: Mx3 array of vertex indices
        
    Returns:
        Nx3 array of vertex normals
    """
    # Initialize normals array
    normals = np.zeros_like(vertices, dtype=np.float32)
    
    # Compute face normals and accumulate on vertices
    for face in faces:
        # Skip degenerate faces (faces with repeated vertices)
        if face[0] == face[1] or face[1] == face[2] or face[2] == face[0]:
            continue
            
        # Get vertices of this face
        v0 = vertices[face[0]]
        v1 = vertices[face[1]]
        v2 = vertices[face[2]]
        
        # Compute face normal using cross product
        edge1 = v1 - v0
        edge2 = v2 - v0
        
        # Check for zero-length edges
        if np.allclose(edge1, 0) or np.allclose(edge2, 0):
            continue
            
        # Calculate normal
        face_normal = np.cross(edge1, edge2)
        
        # Skip faces with zero-area (resulting in zero normal)
        if np.allclose(face_normal, 0):
            continue
            
        # Add to each vertex of this face
        normals[face[0]] += face_normal
        normals[face[1]] += face_normal
        normals[face[2]] += face_normal
    
    # Normalize all vertex normals
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    
    # Handle zero normals to avoid division by zero
    mask = norms[:, 0] > 1e-10
    normals[mask] = normals[mask] / norms[mask]
    
    # For vertices with zero normals, use default up vector
    normals[~mask] = np.array([0, 0, 1])
    
    return normals

def calculate_face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Calculate face normals for a mesh.
    
    Args:
        vertices: Array of 3D vertices
        faces: Array of face indices
        
    Returns:
        Array of normal vectors for each face
    """
    normals = np.zeros((len(faces), 3), dtype=np.float32)
    
    for i, face in enumerate(faces):
        # Get vertices of this face
        v0 = vertices[face[0]]
        v1 = vertices[face[1]]
        v2 = vertices[face[2]]
        
        # Calculate face normal
        e1 = v1 - v0
        e2 = v2 - v0
        normal = np.cross(e1, e2)
        
        # Normalize
        norm = np.linalg.norm(normal)
        if norm > 1e-10:
            normal = normal / norm
            
        normals[i] = normal
    
    return normals

def calculate_heightmap_normals(height_map: np.ndarray) -> np.ndarray:
    """
    Calculate normal vectors for a height map.
    
    Args:
        height_map: 2D array of height values
        
    Returns:
        3D array of normal vectors
    """
    height, width = height_map.shape
    normals = np.zeros((height, width, 3), dtype=np.float32)
    
    # Calculate gradients
    gradient_x = np.zeros_like(height_map)
    gradient_y = np.zeros_like(height_map)
    
    # Interior points
    gradient_x[1:-1, 1:-1] = (height_map[1:-1, 2:] - height_map[1:-1, :-2]) / 2.0
    gradient_y[1:-1, 1:-1] = (height_map[2:, 1:-1] - height_map[:-2, 1:-1]) / 2.0
    
    # Boundary points - forward/backward differences for better accuracy
    # Left & right edges
    gradient_x[1:-1, 0] = height_map[1:-1, 1] - height_map[1:-1, 0]
    gradient_x[1:-1, -1] = height_map[1:-1, -1] - height_map[1:-1, -2]
    
    # Top & bottom edges
    gradient_y[0, 1:-1] = height_map[1, 1:-1] - height_map[0, 1:-1]
    gradient_y[-1, 1:-1] = height_map[-1, 1:-1] - height_map[-2, 1:-1]
    
    # Corners - diagonal differences
    gradient_x[0, 0] = height_map[0, 1] - height_map[0, 0]
    gradient_y[0, 0] = height_map[1, 0] - height_map[0, 0]
    
    gradient_x[0, -1] = height_map[0, -1] - height_map[0, -2]
    gradient_y[0, -1] = height_map[1, -1] - height_map[0, -1]
    
    gradient_x[-1, 0] = height_map[-1, 1] - height_map[-1, 0]
    gradient_y[-1, 0] = height_map[-1, 0] - height_map[-2, 0]
    
    gradient_x[-1, -1] = height_map[-1, -1] - height_map[-1, -2]
    gradient_y[-1, -1] = height_map[-1, -1] - height_map[-2, -1]
    
    # Construct normals (-grad_x, -grad_y, 1)
    normals[:, :, 0] = -gradient_x
    normals[:, :, 1] = -gradient_y
    normals[:, :, 2] = 1.0
    
    # Normalize
    norm = np.sqrt(np.sum(normals**2, axis=2, keepdims=True))
    # Avoid division by zero
    norm[norm < 1e-10] = 1.0
    normals /= norm
    
    return normals.astype(np.float32)  # Ensure float32 type

def ensure_watertight_mesh(
    vertices: np.ndarray, 
    faces: np.ndarray, 
    min_base_height: float = 0.001
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ensure a mesh is watertight by adding a minimal base if needed.
    
    Args:
        vertices: Array of vertices [x, y, z]
        faces: Array of triangular faces [v1, v2, v3]
        min_base_height: Minimum base height to add
        
    Returns:
        Tuple of (vertices, faces) with base added if needed
    """
    # Find the minimum z coordinate
    min_z = np.min(vertices[:, 2])
    
    # Add a very thin base at the minimum z coordinate
    from .base import _add_base_to_mesh
    return _add_base_to_mesh(vertices.tolist(), faces.tolist(), min_base_height)

def optimize_mesh(
    vertices: np.ndarray, 
    faces: np.ndarray, 
    tolerance: float = 1e-10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Optimize a mesh by merging duplicate vertices and removing degenerate faces.
    
    Args:
        vertices: Array of 3D vertices
        faces: Array of face indices
        tolerance: Distance tolerance for merging vertices
        
    Returns:
        Tuple of (optimized_vertices, optimized_faces)
    """
    # Dictionary to track merged vertices
    vertex_map = {}
    unique_vertices = []
    
    # Process each vertex
    for i, vertex in enumerate(vertices):
        # Convert to tuple for hashability
        v_tuple = tuple(vertex)
        
        # Check if this vertex is already in the map
        if v_tuple in vertex_map:
            vertex_map[i] = vertex_map[v_tuple]
        else:
            # Check if this vertex is close to an existing one
            found = False
            for j, unique in enumerate(unique_vertices):
                if np.sum((vertex - unique) ** 2) < tolerance:
                    vertex_map[i] = j
                    vertex_map[v_tuple] = j
                    found = True
                    break
            
            if not found:
                vertex_map[i] = len(unique_vertices)
                vertex_map[v_tuple] = len(unique_vertices)
                unique_vertices.append(vertex)
    
    # Update face indices
    optimized_faces = []
    for face in faces:
        new_face = [vertex_map[idx] for idx in face]
        
        # Skip degenerate faces (where vertices are duplicated)
        if len(set(new_face)) == len(face):
            optimized_faces.append(new_face)
    
    return np.array(unique_vertices), np.array(optimized_faces)

def validate_heightmap(
    height_map: np.ndarray, 
    min_size: Tuple[int, int] = (2, 2)
) -> bool:
    """
    Validate that a heightmap is suitable for processing.
    
    Args:
        height_map: 2D numpy array of height values
        min_size: Minimum dimensions (height, width)
        
    Returns:
        True if valid, False otherwise
    """
    if height_map is None:
        return False
    
    if not isinstance(height_map, np.ndarray):
        return False
    
    if height_map.size == 0:
        return False
        
    if height_map.ndim != 2:
        return False
        
    if height_map.shape[0] < min_size[0] or height_map.shape[1] < min_size[1]:
        return False
        
    return True

def ensure_directory_exists(filepath: str) -> bool:
    """
    Ensure the directory for the given file path exists.
    
    Args:
        filepath: Path to a file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        directory = os.path.dirname(os.path.abspath(filepath))
        os.makedirs(directory, exist_ok=True)
        return True
    except (PermissionError, OSError) as e:
        print(f"Error creating directory for {filepath}: {e}")
        return False

def generate_uv_coordinates(vertices: np.ndarray) -> np.ndarray:
    """
    Generate UV coordinates for vertices based on their X/Y positions.
    
    Args:
        vertices: Nx3 array of vertex positions
        
    Returns:
        Nx2 array of UV coordinates
    """
    # Find min/max x and y coordinates
    min_x = np.min(vertices[:, 0])
    max_x = np.max(vertices[:, 0])
    min_y = np.min(vertices[:, 1])
    max_y = np.max(vertices[:, 1])
    
    # Calculate ranges (avoid division by zero)
    x_range = max_x - min_x if max_x > min_x else 1.0
    y_range = max_y - min_y if max_y > min_y else 1.0
    
    # Generate normalized UVs
    u = (vertices[:, 0] - min_x) / x_range
    v = (vertices[:, 1] - min_y) / y_range
    
    # Flip v for texture coordinates (standard in many formats)
    v = 1.0 - v
    
    return np.column_stack((u, v))
