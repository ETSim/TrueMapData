"""
Base classes for model exporters.

This module defines the abstract base class for all model exporters and
provides common utility functions for creating meshes from heightmaps.
"""

import os
import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import Dict, Type, Optional, Any, List, Tuple, Union

# Set up logging
logger = logging.getLogger(__name__)

class ModelExporter(ABC):
    """
    Abstract base class for all model exporters.
    """
    
    @classmethod
    @abstractmethod
    def export(cls, 
               height_map: np.ndarray, 
               filename: str, 
               x_offset: float = 0.0,
               y_offset: float = 0.0,
               x_length: float = 1.0,
               y_length: float = 1.0,
               z_scale: float = 1.0,
               base_height: float = 0.0,
               **kwargs) -> Optional[str]:
        """
        Export a height map to a 3D model file.
        
        Args:
            height_map: 2D numpy array of height values
            filename: Output filename
            x_offset: X-axis offset for the model
            y_offset: Y-axis offset for the model
            x_length: Physical length in X direction
            y_length: Physical length in Y direction
            z_scale: Scale factor for Z-axis values
            base_height: Height of solid base to add below the model
            **kwargs: Additional format-specific parameters
            
        Returns:
            Path to the created file if successful, None otherwise
        """
        pass
    
    @classmethod
    def get_extension(cls) -> str:
        """
        Get the file extension for this exporter format.
        
        Returns:
            File extension without leading dot (e.g., 'stl', 'obj')
        """
        return ""
    
    @classmethod
    def get_format_name(cls) -> str:
        """
        Get the human-readable format name.
        
        Returns:
            Format name (e.g., 'STL', 'Wavefront OBJ')
        """
        return ""
    
    @classmethod
    def supports_binary(cls) -> bool:
        """
        Check if this format supports binary export.
        
        Returns:
            True if binary export is supported, False otherwise
        """
        return False
    
    @classmethod
    def ensure_extension(cls, filename: str) -> str:
        """
        Ensure filename has the correct extension for this format.
        
        Args:
            filename: Original filename
            
        Returns:
            Filename with correct extension
        """
        ext = cls.get_extension()
        if not ext:
            return filename
            
        if not filename.lower().endswith(f".{ext.lower()}"):
            filename = f"{filename}.{ext}"
            
        return filename


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


def calculate_vertex_normals(vertices: List[List[float]], faces: List[List[int]]) -> List[List[float]]:
    """
    Calculate unit normal vectors for each vertex.
    
    Args:
        vertices: List of vertex positions [x, y, z]
        faces: List of face indices [a, b, c]
        
    Returns:
        List of normal vectors [nx, ny, nz] for each vertex
    """
    # Initialize normal list
    num_vertices = len(vertices)
    normals = [[0.0, 0.0, 0.0] for _ in range(num_vertices)]
    counts = [0] * num_vertices
    
    # Calculate face normals and accumulate to vertices
    for face in faces:
        if len(face) >= 3:
            # Get three vertices of this face
            v1 = vertices[face[0]]
            v2 = vertices[face[1]]
            v3 = vertices[face[2]]
            
            # Calculate face normal using cross product
            # Vectors from v1 to v2 and v1 to v3
            vec1 = [v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]]
            vec2 = [v3[0] - v1[0], v3[1] - v1[1], v3[2] - v1[2]]
            
            # Cross product
            normal = [
                vec1[1] * vec2[2] - vec1[2] * vec2[1],
                vec1[2] * vec2[0] - vec1[0] * vec2[2],
                vec1[0] * vec2[1] - vec1[1] * vec2[0]
            ]
            
            # Normalize face normal
            length = np.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2)
            if length > 1e-10:
                normal = [normal[0]/length, normal[1]/length, normal[2]/length]
            
            # Add face normal to each vertex of this face
            for idx in face:
                normals[idx][0] += normal[0]
                normals[idx][1] += normal[1]
                normals[idx][2] += normal[2]
                counts[idx] += 1
    
    # Average and normalize vertex normals
    for i in range(num_vertices):
        if counts[i] > 0:
            # Average by the number of adjacent faces
            normals[i][0] /= counts[i]
            normals[i][1] /= counts[i]
            normals[i][2] /= counts[i]
            
            # For heightmaps, ensure Z is positive
            if normals[i][2] < 0:
                normals[i][0] *= -1
                normals[i][1] *= -1
                normals[i][2] *= -1
            
            # Normalize to unit length
            length = np.sqrt(normals[i][0]**2 + normals[i][1]**2 + normals[i][2]**2)
            if length > 1e-10:
                normals[i][0] /= length
                normals[i][1] /= length
                normals[i][2] /= length
            else:
                normals[i] = [0.0, 0.0, 1.0]  # Default to straight up if zero length
        else:
            # Default to up vector if no faces
            normals[i] = [0.0, 0.0, 1.0]
    
    return normals


def export_heightmap_to_model(
    height_map: np.ndarray,
    filename: str,
    format_name: str,
    x_offset: float = 0,
    y_offset: float = 0,
    x_length: float = 1,
    y_length: float = 1,
    z_scale: float = 1,
    base_height: float = 0.0,
    binary: bool = False,
    **kwargs
) -> Optional[str]:
    """
    Export a height map to a 3D model file.

    Args:
        height_map: 2D numpy array of height values
        filename: Output filename
        format_name: Format name for the model (e.g., 'stl', 'obj', 'ply')
        x_offset: X-axis offset for the model
        y_offset: Y-axis offset for the model
        x_length: Physical length in X direction
        y_length: Physical length in Y direction
        z_scale: Scale factor for Z-axis values
        base_height: Height of solid base to add below the model
        binary: Whether to use binary format if supported
        **kwargs: Additional keyword arguments to pass to the specific exporter

    Returns:
        str: Path to the created file or None if failed
    """
    try:
        # Import converters here to avoid circular imports
        if format_name.lower() == "nvbd":
            from .nvbd import convert_heightmap_to_nvbd
            return convert_heightmap_to_nvbd(
                height_map=height_map,
                filename=filename,
                scale=z_scale,
                offset=base_height,
                chunk_size=16,
                include_normals=True
            )
        elif format_name.lower() == "usdz":
            from .usd import convert_heightmap_to_usdz
            return convert_heightmap_to_usdz(
                height_map=height_map,
                filename=filename,
                x_offset=x_offset,
                y_offset=y_offset,
                x_length=x_length,
                y_length=y_length,
                z_scale=z_scale,
                base_height=base_height,
                add_texture=True,
                texture_resolution=None
            )
        elif format_name.lower() == "glb":
            from .gltf import convert_heightmap_to_glb
            return convert_heightmap_to_glb(
                height_map=height_map,
                filename=filename,
                x_offset=x_offset,
                y_offset=y_offset,
                x_length=x_length,
                y_length=y_length,
                z_scale=z_scale,
                base_height=base_height
            )
        elif format_name.lower() == "ply":
            from .ply import convert_heightmap_to_ply
            return convert_heightmap_to_ply(
                height_map=height_map,
                filename=filename,
                x_offset=x_offset,
                y_offset=y_offset,
                x_length=x_length,
                y_length=y_length,
                z_scale=z_scale,
                base_height=base_height,
                binary=binary
            )
        elif format_name.lower() == "obj":
            from .obj import convert_heightmap_to_obj
            return convert_heightmap_to_obj(
                height_map=height_map,
                filename=filename,
                x_offset=x_offset,
                y_offset=y_offset,
                x_length=x_length,
                y_length=y_length,
                z_scale=z_scale,
                base_height=base_height
            )
        elif format_name.lower() == "stl":
            from .stl import convert_heightmap_to_stl
            return convert_heightmap_to_stl(
                height_map=height_map,
                filename=filename,
                x_offset=x_offset,
                y_offset=y_offset,
                x_length=x_length,
                y_length=y_length,
                z_scale=z_scale,
                base_height=base_height,
                ascii_format=not binary
            )
        else:
            logger.error(f"Unsupported model format: {format_name}")
            return None
    except ImportError as e:
        logger.error(f"Failed to import required module: {e}")
        return None