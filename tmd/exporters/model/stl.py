"""

STL exporter module for height maps.

This module provides functions for converting height maps to STL files
for 3D printing and rendering.
"""

import os
import struct
import numpy as np
import logging
from typing import Optional, Tuple, Union

from .base import _create_mesh_from_heightmap
from .adaptive_mesh import convert_heightmap_to_adaptive_mesh
from .backends import generate_mesh_with_backend, ModelBackend

# Set up logging
logger = logging.getLogger(__name__)


def convert_heightmap_to_stl(
    height_map,
    filename="output.stl",
    x_offset=0,
    y_offset=0,
    x_length=1,
    y_length=1,
    z_scale=1,
    ascii=False,  # Default to binary for efficiency
    base_height=0.0,
    adaptive=False,
    max_subdivisions=10,
    error_threshold=0.01,
    max_triangles=None,
    progress_callback=None,
    backend: Union[str, ModelBackend] = None,
    x_scale=1.0,
    y_scale=1.0,
    coordinate_system="right-handed",
    origin_at_zero=True,
    preserve_orientation=True,
    invert_base=False,
    # Deprecated parameters but kept for backwards compatibility
    mirror_x=None,
    mirror_y=None
) -> Optional[str]:
    """

    Converts a height map into an STL file for 3D printing.

    Args:
        height_map: 2D numpy array of height values.
        filename: Name of the output STL file.
        x_offset: X-axis offset for the model.
        y_offset: Y-axis offset for the model.
        x_length: Physical length in the X direction.
        y_length: Physical length in the Y direction.
        z_scale: Scale factor for Z-axis values.
        ascii: If True, creates ASCII STL; if False, creates binary STL.
        base_height: Height of solid base to add below the model (0 = no base).
        adaptive: Whether to use adaptive mesh generation.
        max_subdivisions: Maximum subdivision level for adaptive meshing.
        error_threshold: Error threshold for adaptive meshing.
        max_triangles: Maximum number of triangles for adaptive meshing.
        progress_callback: Function to call with progress updates (0-100).
        backend: Backend to use for mesh generation (overrides adaptive parameter).
        x_scale: Scaling factor for x axis.
        y_scale: Scaling factor for y axis.
        coordinate_system: Coordinate system ("right-handed" or "left-handed").
        origin_at_zero: Whether to place origin at zero.
        preserve_orientation: Whether to preserve the original heightmap orientation.
        invert_base: Whether to invert the base to create a mold/negative.
        mirror_x, mirror_y: Deprecated parameters, kept for backwards compatibility.

    Returns:
        str: Path to the created file or None if failed.
    """
    # Check for valid height map
    if height_map is None or height_map.size == 0 or height_map.shape[0] < 2 or height_map.shape[1] < 2:
        logger.error("Invalid height map: empty, None, or too small")
        return None
        
    # Ensure directory exists
    try:
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    except (PermissionError, OSError) as e:
        logger.error(f"Error creating directory for {filename}: {e}")
        return None
    
    # Determine which backend to use
    if backend is not None:
        # If backend is specified, use it
        result, triangle_count = generate_mesh_with_backend(
            height_map=height_map,
            filename=filename,
            backend=backend,
            x_offset=x_offset,
            y_offset=y_offset,
            x_length=x_length,
            y_length=y_length,
            z_scale=z_scale,
            base_height=base_height,
            ascii=ascii,
            max_subdivisions=max_subdivisions,
            error_threshold=error_threshold,
            max_triangles=max_triangles,
            binary=not ascii
        )
        
        if result:
            logger.info(f"Generated STL with {triangle_count} triangles using {backend} backend")
            return result
        else:
            logger.error(f"Failed to generate STL using {backend} backend")
            return None
    
    # Check for large heightmaps
    large_heightmap = height_map.size > 1000000  # Over 1 million pixels
    
    # Use adaptive mesh generation if explicitly requested or for large heightmaps
    if adaptive or large_heightmap:
        logger.info(f"Using adaptive triangulation with error threshold: {error_threshold}")
        
        # Prepare parameters for adaptive mesh conversion
        adaptive_params = {
            "height_map": height_map,
            "output_file": filename,
            "z_scale": z_scale,
            "base_height": base_height,
            "x_scale": x_scale,
            "y_scale": y_scale,
            "max_subdivisions": max_subdivisions,
            "error_threshold": error_threshold,
            "max_triangles": max_triangles,
            "progress_callback": progress_callback,
            "ascii": ascii,
            "coordinate_system": coordinate_system,
            "origin_at_zero": origin_at_zero,
            "preserve_orientation": preserve_orientation,
            "invert_base": invert_base
        }
        
        # Call adaptive mesh converter with full parameter set
        result = convert_heightmap_to_adaptive_mesh(**adaptive_params)
        
        # If result is a tuple, extract the filename
        if isinstance(result, tuple) and len(result) >= 3:
            return result[2]  # The third element should be the filename
        return result
    
    # Otherwise use regular mesh generation
    mesh_result = _create_mesh_from_heightmap(
        height_map, x_offset, y_offset, x_length, y_length, z_scale, base_height
    )
    
    if not mesh_result:
        logger.error("Height map too small to generate STL")
        return None
    
    vertices, faces = mesh_result
    vertices_array = np.array(vertices)
    
    # Calculate and display triangle count
    triangle_count = len(faces)
    logger.info(f"Generated STL with {triangle_count} triangles using standard meshing")
    
    # Write the STL file (ASCII or binary)
    try:
        if ascii:
            # Write ASCII STL
            with open(filename, "w") as f:
                f.write("solid displacement\n")
                
                for face in faces:
                    v0 = vertices_array[face[0]]
                    v1 = vertices_array[face[1]]
                    v2 = vertices_array[face[2]]
                    
                    # Calculate normal
                    n = np.cross(v1 - v0, v2 - v0)
                    norm_val = np.linalg.norm(n)
                    if norm_val < 1e-10:
                        n = np.array([0, 0, 1.0])
                    else:
                        n = n / norm_val
                    
                    # Write facet
                    f.write(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n")
                    f.write("    outer loop\n")
                    f.write(f"      vertex {v0[0]:.6e} {v0[1]:.6e} {v0[2]:.6e}\n")
                    f.write(f"      vertex {v1[0]:.6e} {v1[1]:.6e} {v1[2]:.6e}\n")
                    f.write(f"      vertex {v2[0]:.6e} {v2[1]:.6e} {v2[2]:.6e}\n")
                    f.write("    endloop\n")
                    f.write("  endfacet\n")
                
                f.write("endsolid displacement\n")
                
            logger.info(f"ASCII STL file{' with base' if base_height > 0 else ''} saved to {filename}")
        else:
            # Write binary STL
            with open(filename, "wb") as f:
                # Write header (80 bytes)
                header = b"TMD Processor Generated Binary STL"
                header = header.ljust(80, b" ")
                f.write(header)
                
                # Write number of triangles (4 bytes)
                f.write(struct.pack("<I", len(faces)))
                
                # Write each triangle
                for face in faces:
                    v0 = vertices_array[face[0]]
                    v1 = vertices_array[face[1]]
                    v2 = vertices_array[face[2]]
                    
                    # Calculate normal
                    n = np.cross(v1 - v0, v2 - v0)
                    norm_val = np.linalg.norm(n)
                    if norm_val > 0:
                        n = n / norm_val
                    else:
                        n = np.array([0, 0, 1.0])
                    
                    # Write triangle data
                    f.write(struct.pack("<fff", *n))      # normal
                    f.write(struct.pack("<fff", *v0))     # vertex 1
                    f.write(struct.pack("<fff", *v1))     # vertex 2
                    f.write(struct.pack("<fff", *v2))     # vertex 3
                    f.write(struct.pack("<H", 0))         # attribute byte count
                
            logger.info(f"Binary STL file{' with base' if base_height > 0 else ''} saved to {filename}")
        
        return filename
    except Exception as e:
        logger.error(f"Error writing STL file: {e}")
        return None


def convert_heightmap_to_stl_threaded(
    height_map,
    filename="output_threaded.stl",
    x_offset=0,
    y_offset=0,
    x_length=1,
    y_length=1,
    z_scale=1,
    ascii=False,
    base_height=0.0,
    num_threads=None
):
    """.

    Converts a height map into an STL file using parallel processing.

    Args:
        height_map: 2D numpy array of height values.
        filename: Name of the output STL file.
        x_offset: X-axis offset for the model.
        y_offset: Y-axis offset for the model.
        x_length: Physical length in the X direction.
        y_length: Physical length in the Y direction.
        z_scale: Scale factor for Z-axis values.
        ascii: If True, creates ASCII STL; if False, creates binary STL.
        base_height: Height of solid base to add below the model (0 = no base).
        num_threads: Number of threads to use for processing (None = auto).

    Returns:
        str: Path to the created file or None if failed.
    """
    # Ensure directory exists
    try:
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    except (PermissionError, OSError) as e:
        logger.error(f"Error creating directory for {filename}: {e}")
        return None
    
    # Generate mesh using chunked processing
    from .base import _create_mesh_from_heightmap_chunked
    mesh_result = _create_mesh_from_heightmap_chunked(
        height_map, x_offset, y_offset, x_length, y_length, z_scale, base_height, num_threads
    )
    
    if not mesh_result:
        logger.error("Height map too small to generate STL")
        return None
    
    vertices, faces = mesh_result
    vertices_array = np.array(vertices)

    # Write the STL file (ASCII or binary)
    try:
        if ascii:
            # Write ASCII STL (same as in convert_heightmap_to_stl)
            with open(filename, "w") as f:
                f.write("solid displacement\n")
                
                for face in faces:
                    v0 = vertices_array[face[0]]
                    v1 = vertices_array[face[1]]
                    v2 = vertices_array[face[2]]
                    
                    # Calculate normal
                    n = np.cross(v1 - v0, v2 - v0)
                    norm_val = np.linalg.norm(n)
                    if norm_val < 1e-10:
                        n = np.array([0, 0, 1.0])
                    else:
                        n = n / norm_val
                    
                    # Write facet
                    f.write(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n")
                    f.write("    outer loop\n")
                    f.write(f"      vertex {v0[0]:.6e} {v0[1]:.6e} {v0[2]:.6e}\n")
                    f.write(f"      vertex {v1[0]:.6e} {v1[1]:.6e} {v1[2]:.6e}\n")
                    f.write(f"      vertex {v2[0]:.6e} {v2[1]:.6e} {v2[2]:.6e}\n")
                    f.write("    endloop\n")
                    f.write("  endfacet\n")
                
                f.write("endsolid displacement\n")
                
            logger.info(f"Binary STL file saved to {filename}")
        else:
            # Write binary STL (same as in convert_heightmap_to_stl)
            with open(filename, "wb") as f:
                # Write header (80 bytes)
                header = b"TMD Processor Generated Binary STL"
                header = header.ljust(80, b" ")
                f.write(header)
                
                # Write number of triangles (4 bytes)
                f.write(struct.pack("<I", len(faces)))
                
                # Write each triangle
                for face in faces:
                    v0 = vertices_array[face[0]]
                    v1 = vertices_array[face[1]]
                    v2 = vertices_array[face[2]]
                    
                    # Calculate normal
                    n = np.cross(v1 - v0, v2 - v0)
                    norm_val = np.linalg.norm(n)
                    if norm_val > 0:
                        n = n / norm_val
                    else:
                        n = np.array([0, 0, 1.0])
                    
                    # Write triangle data
                    f.write(struct.pack("<fff", *n))      # normal
                    f.write(struct.pack("<fff", *v0))     # vertex 1
                    f.write(struct.pack("<fff", *v1))     # vertex 2
                    f.write(struct.pack("<fff", *v2))     # vertex 3
                    f.write(struct.pack("<H", 0))         # attribute byte count
                
            logger.info(f"Binary STL file saved to {filename}")
        
        return filename
    except Exception as e:
        logger.error(f"Error writing STL file: {e}")
        return None


def convert_heightmap_to_stl_streamed(
    height_map,
    filename="output_streamed.stl",
    x_offset=0,
    y_offset=0,
    x_length=1,
    y_length=1,
    z_scale=1,
    ascii=False,
    base_height=0.0,
    chunk_size=500
):
    """.

    Converts a height map into an STL file using memory-efficient streaming.
    Useful for very large height maps that don't fit in memory.

    Args:
        height_map: 2D numpy array of height values.
        filename: Name of the output STL file.
        x_offset: X-axis offset for the model.
        y_offset: Y-axis offset for the model.
        x_length: Physical length in the X direction.
        y_length: Physical length in the Y direction.
        z_scale: Scale factor for Z-axis values.
        ascii: If True, creates ASCII STL; if False, creates binary STL.
        base_height: Height of solid base to add below the model (0 = no base).
        chunk_size: Size of chunks to process at once.

    Returns:
        str: Path to the created file or None if failed.
    """
    # Ensure directory exists
    try:
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    except (PermissionError, OSError) as e:
        logger.error(f"Error creating directory for {filename}: {e}")
        return None
    
    rows, cols = height_map.shape
    
    if ascii:
        # Process in streaming mode for ASCII format
        try:
            with open(filename, "w") as f:
                # Write header
                f.write("solid displacement\n")
                
                # Calculate scale factors
                x_scale = x_length / max(1, cols - 1)
                y_scale = y_length / max(1, rows - 1)
                
                # Process height map in chunks
                for i in range(0, rows-1, chunk_size):
                    chunk_end = min(i + chunk_size, rows - 1)
                    for j in range(0, cols-1, chunk_size):
                        chunk_right = min(j + chunk_size, cols - 1)
                        
                        # Process current chunk
                        for ci in range(i, chunk_end):
                            for cj in range(j, chunk_right):
                                # Calculate vertex positions
                                v0 = [
                                    x_offset + cj * x_scale,
                                    y_offset + ci * y_scale,
                                    height_map[ci, cj] * z_scale
                                ]
                                v1 = [
                                    x_offset + (cj + 1) * x_scale,
                                    y_offset + ci * y_scale,
                                    height_map[ci, cj + 1] * z_scale
                                ]
                                v2 = [
                                    x_offset + (cj + 1) * x_scale,
                                    y_offset + (ci + 1) * y_scale,
                                    height_map[ci + 1, cj + 1] * z_scale
                                ]
                                v3 = [
                                    x_offset + cj * x_scale,
                                    y_offset + (ci + 1) * y_scale,
                                    height_map[ci + 1, cj] * z_scale
                                ]
                                
                                # First triangle
                                normal = np.cross(
                                    np.array(v1) - np.array(v0),
                                    np.array(v2) - np.array(v0)
                                )
                                norm_val = np.linalg.norm(normal)
                                if norm_val > 0:
                                    normal = normal / norm_val
                                else:
                                    normal = np.array([0, 0, 1.0])
                                
                                f.write(f"  facet normal {normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}\n")
                                f.write("    outer loop\n")
                                f.write(f"      vertex {v0[0]:.6e} {v0[1]:.6e} {v0[2]:.6e}\n")
                                f.write(f"      vertex {v1[0]:.6e} {v1[1]:.6e} {v1[2]:.6e}\n")
                                f.write(f"      vertex {v2[0]:.6e} {v2[1]:.6e} {v2[2]:.6e}\n")
                                f.write("    endloop\n")
                                f.write("  endfacet\n")
                                
                                # Second triangle
                                normal = np.cross(
                                    np.array(v2) - np.array(v0),
                                    np.array(v3) - np.array(v0)
                                )
                                norm_val = np.linalg.norm(normal)
                                if norm_val > 0:
                                    normal = normal / norm_val
                                else:
                                    normal = np.array([0, 0, 1.0])
                                
                                f.write(f"  facet normal {normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}\n")
                                f.write("    outer loop\n")
                                f.write(f"      vertex {v0[0]:.6e} {v0[1]:.6e} {v0[2]:.6e}\n")
                                f.write(f"      vertex {v2[0]:.6e} {v2[1]:.6e} {v2[2]:.6e}\n")
                                f.write(f"      vertex {v3[0]:.6e} {v3[1]:.6e} {v3[2]:.6e}\n")
                                f.write("    endloop\n")
                                f.write("  endfacet\n")
                
                # Base not yet implemented for streamed version
                if base_height > 0:
                    logger.warning("Base not implemented for streamed version")
                
                f.write("endsolid displacement\n")
                
            logger.info(f"ASCII STL file (streamed) saved to {filename}")
            return filename
        except Exception as e:
            logger.error(f"Error writing STL file: {e}")
            return None
    else:
        # For binary STL, we need to know the face count upfront
        total_triangles = 2 * (rows - 1) * (cols - 1)
        
        try:
            with open(filename, "wb") as f:
                # Write header
                header = b"TMD Processor Generated Binary STL"
                header = header.ljust(80, b" ")
                f.write(header)
                
                # Write number of triangles (4 bytes)
                f.write(struct.pack("<I", total_triangles))
                
                # Calculate scale factors
                x_scale = x_length / max(1, cols - 1)
                y_scale = y_length / max(1, rows - 1)
                
                # Process height map in chunks
                for i in range(0, rows-1, chunk_size):
                    chunk_end = min(i + chunk_size, rows - 1)
                    for j in range(0, cols-1, chunk_size):
                        chunk_right = min(j + chunk_size, cols - 1)
                        
                        # Process current chunk
                        for ci in range(i, chunk_end):
                            for cj in range(j, chunk_right):
                                # Calculate vertex positions
                                v0 = np.array([
                                    x_offset + cj * x_scale,
                                    y_offset + ci * y_scale,
                                    height_map[ci, cj] * z_scale
                                ])
                                v1 = np.array([
                                    x_offset + (cj + 1) * x_scale,
                                    y_offset + ci * y_scale,
                                    height_map[ci, cj + 1] * z_scale
                                ])
                                v2 = np.array([
                                    x_offset + (cj + 1) * x_scale,
                                    y_offset + (ci + 1) * y_scale,
                                    height_map[ci + 1, cj + 1] * z_scale
                                ])
                                v3 = np.array([
                                    x_offset + cj * x_scale,
                                    y_offset + (ci + 1) * y_scale,
                                    height_map[ci + 1, cj] * z_scale
                                ])
                                
                                # First triangle
                                normal = np.cross(v1 - v0, v2 - v0)
                                norm_val = np.linalg.norm(normal)
                                if norm_val > 0:
                                    normal = normal / norm_val
                                else:
                                    normal = np.array([0, 0, 1.0])
                                
                                f.write(struct.pack("<fff", *normal))  # normal
                                f.write(struct.pack("<fff", *v0))     # vertex 1
                                f.write(struct.pack("<fff", *v1))     # vertex 2
                                f.write(struct.pack("<fff", *v2))     # vertex 3
                                f.write(struct.pack("<H", 0))         # attribute byte count
                                
                                # Second triangle
                                normal = np.cross(v2 - v0, v3 - v0)
                                norm_val = np.linalg.norm(normal)
                                if norm_val > 0:
                                    normal = normal / norm_val
                                else:
                                    normal = np.array([0, 0, 1.0])
                                
                                f.write(struct.pack("<fff", *normal))  # normal
                                f.write(struct.pack("<fff", *v0))     # vertex 1
                                f.write(struct.pack("<fff", *v2))     # vertex 2
                                f.write(struct.pack("<fff", *v3))     # vertex 3
                                f.write(struct.pack("<H", 0))         # attribute byte count
                
            logger.info(f"Binary STL file (streamed) saved to {filename}")
            return filename
        except Exception as e:
            logger.error(f"Error writing STL file: {e}")
            return None
