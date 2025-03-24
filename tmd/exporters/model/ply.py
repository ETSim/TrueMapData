""".

PLY exporter module for height maps.

This module provides functionality for converting height maps to PLY (Polygon File Format)
files, which are commonly used for storing 3D model data.
"""

import os
import logging
import numpy as np
from typing import Dict, Any, Optional, Union

# Change to import from base instead of mesh_generation
from .base import create_mesh_from_heightmap, _calculate_vertex_normals, _add_base_to_mesh

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
    binary: bool = False
) -> Optional[str]:
    """.

    Convert a height map to a PLY (Polygon File Format) file.
    
    Args:
        height_map: 2D numpy array of height values
        filename: Output filename
        x_offset: X-axis offset for the model
        y_offset: Y-axis offset for the model
        x_length: Physical length in X direction
        y_length: Physical length in Y direction
        z_scale: Scale factor for Z-axis values
        base_height: Height of solid base to add below the model
        binary: Whether to use binary PLY format
        
    Returns:
        Path to the created file or None if failed
    """
    try:
        # Ensure we have a valid heightmap
        if height_map is None or height_map.size == 0 or height_map.shape[0] < 2 or height_map.shape[1] < 2:
            logger.error("Invalid heightmap: empty, None, or too small")
            return None
            
        # Fix extension if needed
        if not filename.lower().endswith('.ply'):
            filename = filename + '.ply'
        
        # Generate mesh (vertices and faces)
        vertices, faces = create_mesh_from_heightmap(
            height_map=height_map,
            x_offset=x_offset,
            y_offset=y_offset,
            x_length=x_length,
            y_length=y_length,
            z_scale=z_scale,
            base_height=base_height
        )
        
        # Check if valid mesh was created
        if not vertices or not faces or len(vertices) < 3 or len(faces) < 1:
            logger.error("Failed to generate valid mesh from height map")
            return None
        
        # Calculate normals
        normals = _calculate_vertex_normals(vertices, faces)
        
        # Ensure output directory exists
        try:
            os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        except (PermissionError, OSError) as e:
            logger.error(f"Error creating directory for {filename}: {e}")
            return None
        
        # Write PLY file
        with open(filename, 'wb' if binary else 'w') as f:
            # Write header
            if not binary:
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"element vertex {len(vertices)}\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                f.write("property float nx\n")
                f.write("property float ny\n")
                f.write("property float nz\n")
                f.write(f"element face {len(faces)}\n")
                f.write("property list uchar int vertex_indices\n")
                f.write("end_header\n")
                
                # Write vertices with normals
                for i, v in enumerate(vertices):
                    n = normals[i]
                    f.write(f"{v[0]} {v[1]} {v[2]} {n[0]} {n[1]} {n[2]}\n")
                
                # Write faces
                for face in faces:
                    f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
            else:
                # Binary PLY format
                try:
                    import struct
                    
                    # Write header as ASCII
                    f.write(b"ply\n")
                    f.write(b"format binary_little_endian 1.0\n")
                    f.write(f"element vertex {len(vertices)}\n".encode())
                    f.write(b"property float x\n")
                    f.write(b"property float y\n")
                    f.write(b"property float z\n")
                    f.write(b"property float nx\n")
                    f.write(b"property float ny\n")
                    f.write(b"property float nz\n")
                    f.write(f"element face {len(faces)}\n".encode())
                    f.write(b"property list uchar int vertex_indices\n")
                    f.write(b"end_header\n")
                    
                    # Write vertices with normals
                    for i, v in enumerate(vertices):
                        n = normals[i]
                        f.write(struct.pack("<ffffff", 
                                           float(v[0]), float(v[1]), float(v[2]),
                                           float(n[0]), float(n[1]), float(n[2])))
                    
                    # Write faces
                    for face in faces:
                        f.write(struct.pack("<Biii", 3, int(face[0]), int(face[1]), int(face[2])))
                        
                except ImportError:
                    logger.warning("struct module not available for binary PLY. Using ASCII instead.")
                    # Fall back to ASCII format
                    f.close()
                    return convert_heightmap_to_ply(
                        height_map, filename, x_offset, y_offset, 
                        x_length, y_length, z_scale, base_height, binary=False
                    )
        
        logger.info(f"PLY file saved to {filename}")
        return filename
    
    except Exception as e:
        logger.error(f"Error creating PLY file: {e}")
        return None

def export_heightmap_to_ply(
    height_map: np.ndarray,
    filename: str,
    x_offset: float = 0,
    y_offset: float = 0,
    x_length: float = 1,
    y_length: float = 1,
    z_scale: float = 1,
    base_height: float = 0.0,
    binary: bool = False
) -> bool:
    """.

    Export a height map to a PLY file.
    
    Args:
        height_map: 2D numpy array of height values
        filename: Output filename
        x_offset: X-axis offset for the model
        y_offset: Y-axis offset for the model
        x_length: Physical length in X direction
        y_length: Physical length in Y direction
        z_scale: Scale factor for Z-axis values
        base_height: Height of solid base to add below the model
        binary: Whether to use binary PLY format
        
    Returns:
        True if successful, False otherwise
    """
    result = convert_heightmap_to_ply(
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
    
    return result is not None
