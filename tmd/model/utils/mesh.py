"""Mesh manipulation utilities."""

import os
import time
import logging
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union, Callable
from ..core.mesh import MeshData, MeshOperationError

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
    Create a triangular mesh from a heightmap.
    
    Args:
        height_map: 2D numpy array of height values
        x_offset: X-axis offset for the model
        y_offset: Y-axis offset for the model
        x_length: Physical length in X direction
        y_length: Physical length in Y direction
        z_scale: Z-scale factor for height values
        base_height: Height of solid base to add below the model
        
    Returns:
        Tuple of (vertices, faces)
    """
    # Normalize height map if needed
    height_map_norm = height_map.copy()
    
    rows, cols = height_map_norm.shape
    vertices = []
    faces = []
    
    # Calculate grid spacings
    dx = x_length / (cols - 1) if cols > 1 else x_length
    dy = y_length / (rows - 1) if rows > 1 else y_length
    
    # Create vertices
    vertex_indices = {}
    index = 0
    
    for r in range(rows):
        for c in range(cols):
            # Calculate x and y positions
            x = x_offset + c * dx
            y = y_offset + r * dy
            
            # Get height value and apply z-scale
            z = height_map_norm[r, c] * z_scale
            
            # Add vertex
            vertices.append([x, y, z])
            vertex_indices[(r, c)] = index
            index += 1
    
    # Create faces (triangles)
    for r in range(rows - 1):
        for c in range(cols - 1):
            # Get vertex indices for this grid cell
            v0 = vertex_indices[(r, c)]
            v1 = vertex_indices[(r, c+1)]
            v2 = vertex_indices[(r+1, c)]
            v3 = vertex_indices[(r+1, c+1)]
            
            # Create two triangles for this grid cell
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])
    
    # Add base if requested
    if base_height > 0:
        vertices, faces = _add_base_to_mesh(vertices, faces, base_height)
    
    return vertices, faces


def _add_base_to_mesh(vertices, faces, base_height):
    """
    Add a solid base below the mesh with minimal triangles.
    
    Args:
        vertices: List of [x,y,z] coordinates.
        faces: List of [v1,v2,v3] vertex indices.
        base_height: Height of the base to add (in model units).
        
    Returns:
        tuple: (new_vertices, new_faces) with base added.
    """
    if base_height <= 0:
        return vertices, faces
    
    # Find mesh dimensions
    vertices_array = np.array(vertices)
    x_coords = vertices_array[:, 0]
    y_coords = vertices_array[:, 1]
    z_coords = vertices_array[:, 2]
    
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    z_min = np.min(z_coords)
    
    # Create base coordinates (below the mesh)
    base_z = z_min - base_height
    
    # Copy all existing vertices and faces
    new_vertices = vertices.copy()
    new_faces = faces.copy()
    
    # For minimal triangle count, create only 5 base vertices:
    # - One at the center of the base
    # - Four at the corners of the base
    
    # Add center vertex
    center_idx = len(new_vertices)
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    new_vertices.append([center_x, center_y, base_z])
    
    # Add corner vertices
    bl_idx = len(new_vertices)  # bottom left
    new_vertices.append([x_min, y_min, base_z])
    
    br_idx = len(new_vertices)  # bottom right
    new_vertices.append([x_max, y_min, base_z])
    
    tr_idx = len(new_vertices)  # top right
    new_vertices.append([x_max, y_max, base_z])
    
    tl_idx = len(new_vertices)  # top left
    new_vertices.append([x_min, y_max, base_z])
    
    # Identify perimeter vertices
    # A vertex is on the perimeter if it's part of only one triangle's edge
    edge_count = {}
    perimeter_vertices = set()
    
    for face in faces:
        edges = [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]
        for edge in edges:
            # Sort the vertex indices to ensure we count the edge properly
            sorted_edge = tuple(sorted(edge))
            edge_count[sorted_edge] = edge_count.get(sorted_edge, 0) + 1
    
    # Edges that appear only once are on the boundary
    perimeter_edges = [edge for edge, count in edge_count.items() if count == 1]
    
    # Extract unique vertices on the perimeter
    for edge in perimeter_edges:
        perimeter_vertices.add(edge[0])
        perimeter_vertices.add(edge[1])
    
    # Group perimeter vertices by side
    left_side = []
    right_side = []
    bottom_side = []
    top_side = []
    
    for v_idx in perimeter_vertices:
        x, y, z = vertices[v_idx]
        
        # Assign to appropriate side (allowing for small floating-point differences)
        if abs(x - x_min) < 1e-5:
            left_side.append(v_idx)
        elif abs(x - x_max) < 1e-5:
            right_side.append(v_idx)
            
        if abs(y - y_min) < 1e-5:
            bottom_side.append(v_idx)
        elif abs(y - y_max) < 1e-5:
            top_side.append(v_idx)
    
    # Create triangles for the base (just 4 triangles)
    new_faces.append([center_idx, bl_idx, br_idx])  # Center to bottom edge
    new_faces.append([center_idx, br_idx, tr_idx])  # Center to right edge
    new_faces.append([center_idx, tr_idx, tl_idx])  # Center to top edge
    new_faces.append([center_idx, tl_idx, bl_idx])  # Center to left edge
    
    # Connect perimeter vertices to appropriate base corners
    for v_idx in perimeter_vertices:
        x, y, _ = vertices[v_idx]
        
        # Determine closest corner
        dist_bl = (x - x_min)**2 + (y - y_min)**2
        dist_br = (x - x_max)**2 + (y - y_min)**2
        dist_tr = (x - x_max)**2 + (y - y_max)**2
        dist_tl = (x - x_min)**2 + (y - y_max)**2
        
        closest_corner_idx = min(
            (dist_bl, bl_idx), 
            (dist_br, br_idx),
            (dist_tr, tr_idx),
            (dist_tl, tl_idx)
        )[1]
        
        # Find adjacent perimeter vertices
        for edge in perimeter_edges:
            if v_idx in edge:
                other_idx = edge[0] if edge[1] == v_idx else edge[1]
                # Create a triangle from the two perimeter vertices to the base corner
                new_faces.append([v_idx, other_idx, closest_corner_idx])
    
    return new_vertices, new_faces


def calculate_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Calculate unit normal vectors for each vertex using weighted contribution from adjacent faces.
    
    Args:
        vertices: Nx3 array of vertex positions
        faces: Mx3 array of vertex indices
        
    Returns:
        Nx3 array of unit normal vectors for each vertex
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
            
        # Calculate normal and face area (half the cross product magnitude)
        face_normal = np.cross(edge1, edge2)
        face_area = np.linalg.norm(face_normal) * 0.5
        
        # Skip faces with zero-area (resulting in zero normal)
        if face_area < 1e-10:
            continue
        
        # Normalize the face normal
        face_normal = face_normal / (2.0 * face_area)
            
        # Add to each vertex of this face, weighted by face area
        normals[face[0]] += face_normal * face_area
        normals[face[1]] += face_normal * face_area
        normals[face[2]] += face_normal * face_area
    
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
    Calculate normal vectors for each face in a mesh.
    
    Args:
        vertices: Array of 3D vertices
        faces: Array of face indices
        
    Returns:
        Array of unit normal vectors for each face
    """
    normals = np.zeros((len(faces), 3), dtype=np.float32)
    
    for i, face in enumerate(faces):
        # Get vertices of this face
        v0 = vertices[face[0]]
        v1 = vertices[face[1]]
        v2 = vertices[face[2]]
        
        # Calculate face normal using cross product of edges
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        
        # Normalize
        norm = np.linalg.norm(normal)
        if norm > 1e-10:
            normal = normal / norm
        else:
            # For degenerate faces, use a default normal
            normal = np.array([0, 0, 1], dtype=np.float32)
            
        normals[i] = normal
    
    return normals


def generate_uv_coordinates(
    vertices: np.ndarray,
    method: str = 'planar'
) -> np.ndarray:
    """
    Generate UV coordinates for vertices based on their positions.
    
    Args:
        vertices: Nx3 array of vertex positions
        method: UV mapping method ('planar', 'cylindrical', or 'spherical')
        
    Returns:
        Nx2 array of UV coordinates
    """
    if method == 'cylindrical':
        return _generate_cylindrical_uvs(vertices)
    elif method == 'spherical':
        return _generate_spherical_uvs(vertices)
    else:  # Default to planar
        return _generate_planar_uvs(vertices)


def _generate_planar_uvs(vertices: np.ndarray) -> np.ndarray:
    """Generate planar UV mapping based on X and Y coordinates."""
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


def _generate_cylindrical_uvs(vertices: np.ndarray) -> np.ndarray:
    """Generate cylindrical UV mapping using polar coordinates."""
    # Find center of the mesh in XY plane
    center_x = (np.max(vertices[:, 0]) + np.min(vertices[:, 0])) / 2
    center_y = (np.max(vertices[:, 1]) + np.min(vertices[:, 1])) / 2
    
    # Find min/max z coordinate
    min_z = np.min(vertices[:, 2])
    max_z = np.max(vertices[:, 2])
    z_range = max_z - min_z if max_z > min_z else 1.0
    
    # Calculate UV coordinates
    uvs = np.zeros((len(vertices), 2), dtype=np.float32)
    
    for i, vertex in enumerate(vertices):
        # Calculate polar angle in XY plane
        dx = vertex[0] - center_x
        dy = vertex[1] - center_y
        
        # Calculate angle in radians, then normalize to [0, 1]
        angle = np.arctan2(dy, dx)
        u = angle / (2 * np.pi) + 0.5
        
        # Normalize Z coordinate for v
        v = 1.0 - (vertex[2] - min_z) / z_range
        
        uvs[i] = [u, v]
    
    return uvs


def _generate_spherical_uvs(vertices: np.ndarray) -> np.ndarray:
    """Generate spherical UV mapping."""
    # Find center of the mesh
    center_x = (np.max(vertices[:, 0]) + np.min(vertices[:, 0])) / 2
    center_y = (np.max(vertices[:, 1]) + np.min(vertices[:, 1])) / 2
    center_z = (np.max(vertices[:, 2]) + np.min(vertices[:, 2])) / 2
    
    # Calculate UV coordinates
    uvs = np.zeros((len(vertices), 2), dtype=np.float32)
    
    for i, vertex in enumerate(vertices):
        # Vector from center to vertex
        dx = vertex[0] - center_x
        dy = vertex[1] - center_y
        dz = vertex[2] - center_z
        
        # Convert to spherical coordinates
        radius = np.sqrt(dx*dx + dy*dy + dz*dz)
        if radius < 1e-10:
            # Avoid division by zero for vertices at center
            uvs[i] = [0.5, 0.5]
            continue
            
        # Calculate spherical angles
        theta = np.arccos(dz / radius)  # Polar angle
        phi = np.arctan2(dy, dx)        # Azimuthal angle
        
        # Convert to UV space
        u = phi / (2 * np.pi) + 0.5
        v = 1.0 - theta / np.pi
        
        uvs[i] = [u, v]
    
    return uvs


def optimize_mesh(vertices: Union[np.ndarray, List[List[float]]], 
                 faces: Union[np.ndarray, List[List[int]]],
                 tolerance: float = 1e-6) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Optimize a mesh by merging close vertices and removing degenerate faces."""
    try:
        # Convert inputs to numpy arrays
        vertices = np.asarray(vertices, dtype=np.float32)
        faces = np.asarray(faces, dtype=np.int32)

        # Find unique vertices using a higher precision for merging
        # Round to fixed number of decimal places instead of using tolerance
        decimals = int(-np.log10(tolerance))
        rounded = np.round(vertices, decimals=decimals)
        unique_verts, index_map = np.unique(rounded, axis=0, return_inverse=True)

        # Remap face indices and filter degenerate faces
        remapped_faces = index_map[faces]
        valid_faces = []
        
        for face in remapped_faces:
            # Check if face has 3 unique vertices
            if (face[0] != face[1] and face[1] != face[2] and face[2] != face[0]):
                # Check triangle area
                v1, v2, v3 = unique_verts[face]
                edge1 = v2 - v1
                edge2 = v3 - v1
                area = np.linalg.norm(np.cross(edge1, edge2)) / 2
                if area > tolerance:
                    valid_faces.append(face)

        if not valid_faces:
            return None

        optimized_faces = np.array(valid_faces, dtype=np.int32)
        
        # Keep only used vertices
        used_verts = np.unique(optimized_faces)
        vert_map = {old: new for new, old in enumerate(used_verts)}
        optimized_verts = unique_verts[used_verts]
        
        # Update face indices
        final_faces = np.array([[vert_map[v] for v in face] for face in optimized_faces])

        return optimized_verts, final_faces

    except Exception as e:
        logger.error(f"Mesh optimization failed: {e}")
        return None


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
    return _add_base_to_mesh(vertices.tolist(), faces.tolist(), min_base_height)