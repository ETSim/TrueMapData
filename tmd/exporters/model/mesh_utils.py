""".

Mesh utility functions for 3D model creation and manipulation.

This module provides utility functions for working with meshes, such as calculating
normals, smoothing, and other mesh operations.
"""

import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict, Set, Optional

def calculate_vertex_normals(vertices, faces):
    """
    Calculate vertex normals for a mesh.
    
    Args:
        vertices: List of vertices (x, y, z)
        faces: List of faces (each face is a list of vertex indices)
        
    Returns:
        List of vertex normals (nx, ny, nz)
    """
    import numpy as np
    
    # Convert to numpy arrays if not already
    vertices = np.array(vertices)
    faces = np.array(faces)
    
    # Initialize vertex normals with zeros
    normals = np.zeros_like(vertices, dtype=float)
    
    # Calculate face normals and add to vertex normals
    for face in faces:
        # Get the three vertices of this face
        v0 = vertices[face[0]]
        v1 = vertices[face[1]]
        v2 = vertices[face[2]]
        
        # Calculate face normal using cross product
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        
        # Normalize face normal
        length = np.linalg.norm(normal)
        if length > 0:
            normal = normal / length
        
        # Add face normal to all vertices of this face
        normals[face[0]] += normal
        normals[face[1]] += normal
        normals[face[2]] += normal
    
    # Normalize vertex normals
    for i in range(len(normals)):
        length = np.linalg.norm(normals[i])
        if length > 0:
            normals[i] = normals[i] / length
    
    return normals

def calculate_face_normals(vertices: List[List[float]], faces: List[List[int]]) -> List[List[float]]:
    """.

    Calculate normals for each face in a mesh.
    
    Args:
        vertices: List of [x,y,z] coordinates
        faces: List of [v1,v2,v3] vertex indices
        
    Returns:
        List of normal vectors [nx,ny,nz] for each face
    """
    normals = []
    
    for face in faces:
        # Get vertex positions
        v0 = np.array(vertices[face[0]])
        v1 = np.array(vertices[face[1]])
        v2 = np.array(vertices[face[2]])
        
        # Calculate face normal using cross product
        edge1 = v1 - v0
        edge2 = v2 - v0
        face_normal = np.cross(edge1, edge2)
        
        # Normalize
        norm = np.linalg.norm(face_normal)
        if norm > 0:
            face_normal = face_normal / norm
        else:
            # Default normal if degenerate
            face_normal = np.array([0.0, 0.0, 1.0])
        
        normals.append(face_normal.tolist())
    
    return normals

def calculate_edge_lengths(vertices: List[List[float]], faces: List[List[int]]) -> List[List[float]]:
    """.

    Calculate the edge lengths for each face.
    
    Args:
        vertices: List of [x,y,z] coordinates
        faces: List of [v1,v2,v3] vertex indices
        
    Returns:
        List of [e1,e2,e3] edge lengths for each face
    """
    edge_lengths = []
    
    for face in faces:
        # Get vertex positions
        v0 = np.array(vertices[face[0]])
        v1 = np.array(vertices[face[1]])
        v2 = np.array(vertices[face[2]])
        
        # Calculate edge lengths
        e1 = np.linalg.norm(v1 - v0)
        e2 = np.linalg.norm(v2 - v1)
        e3 = np.linalg.norm(v0 - v2)
        
        edge_lengths.append([e1, e2, e3])
    
    return edge_lengths

def optimize_mesh(vertices, faces, tolerance=0.0001):
    """.

    Optimize a mesh by merging duplicate vertices and removing degenerate faces.
    
    Args:
        vertices: List of [x,y,z] coordinates.
        faces: List of [v1,v2,v3] vertex indices.
        tolerance: Distance tolerance for merging vertices.
        
    Returns:
        tuple: (optimized_vertices, optimized_faces)
    """
    # Convert to numpy for faster processing
    vertices_array = np.array(vertices)
    faces_array = np.array(faces)
    
    # Create a mapping from old to new vertex indices
    unique_vertices = []
    vertex_map = {}
    
    for i, vertex in enumerate(vertices_array):
        # Check if vertex is already in the unique list (within tolerance)
        found = False
        for j, unique_vertex in enumerate(unique_vertices):
            if np.linalg.norm(vertex - unique_vertex) < tolerance:
                vertex_map[i] = j
                found = True
                break
        
        if not found:
            vertex_map[i] = len(unique_vertices)
            unique_vertices.append(vertex)
    
    # Remap faces
    optimized_faces = []
    for face in faces_array:
        new_face = [vertex_map[idx] for idx in face]
        
        # Skip degenerate faces (where vertices are mapped to the same index)
        if len(set(new_face)) == 3:
            optimized_faces.append(new_face)
    
    return unique_vertices, optimized_faces

def triangulate_quad_mesh(vertices, quads):
    """.

    Convert a quad mesh to triangles.
    
    Args:
        vertices: List of [x,y,z] coordinates.
        quads: List of [v1,v2,v3,v4] vertex indices.
        
    Returns:
        list: List of triangular faces [v1,v2,v3]
    """
    triangles = []
    for quad in quads:
        if len(quad) != 4:
            continue  # Skip non-quad faces
        
        # Split quad into two triangles
        triangles.append([quad[0], quad[1], quad[2]])
        triangles.append([quad[0], quad[2], quad[3]])
    
    return triangles
