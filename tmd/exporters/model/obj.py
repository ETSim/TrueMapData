""".

OBJ exporter module for height maps.

This module provides functions for converting height maps to OBJ files
for 3D visualization and rendering.
"""

import os
import numpy as np
import logging

from .base import _create_mesh_from_heightmap

# Set up logging
logger = logging.getLogger(__name__)

def convert_heightmap_to_obj(
    height_map,
    filename="output.obj",
    x_offset=0,
    y_offset=0,
    x_length=1,
    y_length=1,
    z_scale=1,
    base_height=0.0
):
    """.

    Converts a height map into an OBJ file.

    Args:
        height_map: 2D numpy array of height values.
        filename: Name of the output OBJ file.
        x_offset: X-axis offset for the model.
        y_offset: Y-axis offset for the model.
        x_length: Physical length in the X direction.
        y_length: Physical length in the Y direction.
        z_scale: Scale factor for Z-axis values.
        base_height: Height of solid base to add below the model (0 = no base).

    Returns:
        str: Path to the created file or None if failed.
    """
    # Ensure directory exists
    try:
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    except (PermissionError, OSError) as e:
        logger.error(f"Error creating directory for {filename}: {e}")
        return None
    
    # Generate mesh
    mesh_result = _create_mesh_from_heightmap(
        height_map, x_offset, y_offset, x_length, y_length, z_scale, base_height
    )
    
    if not mesh_result:
        logger.error("Height map too small to generate OBJ")
        return None
    
    vertices, faces = mesh_result
    
    # Calculate normals for each face
    normals = []
    for face in faces:
        v0 = np.array(vertices[face[0]])
        v1 = np.array(vertices[face[1]])
        v2 = np.array(vertices[face[2]])
        
        # Calculate normal using cross product
        normal = np.cross(v1 - v0, v2 - v0)
        norm_val = np.linalg.norm(normal)
        if norm_val > 0:
            normal = normal / norm_val
        else:
            normal = np.array([0, 0, 1.0])
        
        normals.append(normal)
    
    # Write OBJ file
    try:
        with open(filename, "w") as f:
            # Write header
            f.write(f"# OBJ file generated from heightmap\n")
            f.write(f"# Original dimensions: {height_map.shape[0]}x{height_map.shape[1]}\n")
            
            # Write vertices
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
            # Write vertex normals (calculate average normals per vertex)
            vertex_to_normals = {}
            for i, face in enumerate(faces):
                for vertex_idx in face:
                    if vertex_idx not in vertex_to_normals:
                        vertex_to_normals[vertex_idx] = []
                    vertex_to_normals[vertex_idx].append(normals[i])
            
            # Average normals for each vertex
            for v_idx in range(len(vertices)):
                if v_idx in vertex_to_normals:
                    v_normals = vertex_to_normals[v_idx]
                    avg_normal = np.mean(v_normals, axis=0)
                    norm_val = np.linalg.norm(avg_normal)
                    if norm_val > 0:
                        avg_normal = avg_normal / norm_val
                    f.write(f"vn {avg_normal[0]:.6f} {avg_normal[1]:.6f} {avg_normal[2]:.6f}\n")
                else:
                    # Fall back to default normal if vertex has no faces
                    f.write(f"vn 0.000000 0.000000 1.000000\n")
            
            # Write faces (OBJ is 1-indexed)
            for i, face in enumerate(faces):
                # Format is: f v1//vn1 v2//vn2 v3//vn3
                f.write(f"f {face[0]+1}//{face[0]+1} {face[1]+1}//{face[1]+1} {face[2]+1}//{face[2]+1}\n")
        
        logger.info(f"OBJ file{' with base' if base_height > 0 else ''} saved to {filename}")
        return filename
    except Exception as e:
        logger.error(f"Error writing OBJ file: {e}")
        return None

def convert_heightmap_to_obj_meshio(
    height_map,
    filename="meshio_output.obj",
    x_offset=0,
    y_offset=0,
    x_length=1,
    y_length=1,
    z_scale=1,
    base_height=0.0
):
    """.

    Converts a height map into an OBJ file using meshio.

    Args:
        height_map: 2D numpy array of height values.
        filename: Output OBJ filename.
        x_offset: X-axis offset.
        y_offset: Y-axis offset.
        x_length: Physical length in the X direction.
        y_length: Physical length in the Y direction.
        z_scale: Scale factor for Z-axis.
        base_height: Height of solid base to add below the model (0 = no base).
    
    Returns:
        str: Path to the created file or None if failed.
    """
    try:
        from .base import _export_with_meshio
        return _export_with_meshio(
            height_map, filename, "obj",
            x_offset, y_offset, x_length, y_length, z_scale, base_height
        )
    except ImportError:
        logger.error("meshio library not found. Please install with 'pip install meshio'.")
        return None
