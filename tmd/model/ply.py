#!/usr/bin/env python3
"""
PLY exporter module for height maps.

This module provides functions for converting height maps to PLY files,
which are commonly used for storing 3D scanned data.
Now with optional Open3D support and adaptive normal estimation.
"""

import os
import numpy as np
import logging
import struct
from typing import Optional

# Import Open3D for improved mesh handling and processing
import open3d as o3d

from .base import create_mesh_from_heightmap
from .mesh_utils import (
    calculate_vertex_normals,
    validate_heightmap,
    ensure_directory_exists,
    generate_uv_coordinates  # if used elsewhere
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
    use_open3d: bool = True,
    **kwargs
) -> Optional[str]:
    """
    Convert a height map to PLY format using Open3D for export.
    
    Args:
        height_map: 2D numpy array of height values.
        filename: Output filename.
        x_offset: X-axis offset for the model.
        y_offset: Y-axis offset for the model.
        x_length: Physical length in X direction.
        y_length: Physical length in Y direction.
        z_scale: Scale factor for Z-axis values.
        base_height: Height of solid base to add below the model.
        calculate_normals: Whether to calculate vertex normals.
        add_color: Whether to add color based on height values.
        color_map: Name of the colormap to use for colors.
        use_open3d: If True, uses Open3D to export the mesh; otherwise, falls back to manual binary export.
        **kwargs: Additional options.
        
    Returns:
        Path to the created file or None if failed.
    """
    # Validate the input height map
    if not validate_heightmap(height_map):
        logger.error("Invalid height map: empty, None, or too small")
        return None

    # Ensure the output filename ends with .ply and the output directory exists
    filename = filename if filename.lower().endswith('.ply') else f"{os.path.splitext(filename)[0]}.ply"
    if not ensure_directory_exists(filename):
        return None

    try:
        # Create mesh from the height map (vertices and faces)
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
            logger.error("Failed to generate mesh from height map")
            return None

        # Convert lists to numpy arrays
        vertices_array = np.array(vertices)
        faces_array = np.array(faces)

        # Calculate normals if requested
        normals = None
        if calculate_normals:
            normals = calculate_vertex_normals(vertices_array, faces_array)

        # Calculate vertex colors if requested
        colors = _generate_vertex_colors(vertices_array, height_map, color_map) if add_color else None

        # Export the mesh using Open3D or fallback to custom binary writer
        if use_open3d:
            if _export_with_open3d(filename, vertices_array, faces_array, normals, colors):
                logger.info(f"Exported PLY file to {filename} using Open3D")
                return filename
            else:
                logger.error("Open3D failed to write the triangle mesh.")
                return None
        else:
            with open(filename, 'wb') as f:
                _write_binary_ply(f, vertices_array, faces_array, normals, colors)
            logger.info(f"Exported PLY file to {filename} using custom binary writer")
            return filename

    except Exception as e:
        logger.error(f"Error exporting PLY: {e}")
        import traceback
        traceback.print_exc()
        return None


def _export_with_open3d(
    filename: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: Optional[np.ndarray],
    colors: Optional[np.ndarray]
) -> bool:
    """
    Export the mesh using Open3D's TriangleMesh and its I/O functions.
    
    Args:
        filename: Destination file path.
        vertices: Nx3 numpy array of vertex positions.
        faces: Mx3 numpy array of face indices.
        normals: Nx3 numpy array of vertex normals (optional).
        colors: Nx3 numpy array of vertex colors (optional).
        
    Returns:
        True if export succeeds, False otherwise.
    """
    try:
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces)

        # Set normals if provided; otherwise compute them
        if normals is not None:
            mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
        else:
            mesh.compute_vertex_normals()

        # Set vertex colors if provided (Open3D expects colors in [0, 1])
        if colors is not None:
            colors_normalized = colors.astype(np.float64) / 255.0
            mesh.vertex_colors = o3d.utility.Vector3dVector(colors_normalized)

        # Write the mesh to file in binary PLY format
        return o3d.io.write_triangle_mesh(filename, mesh, write_ascii=False)
    except Exception as e:
        logger.error(f"Error during Open3D export: {e}")
        return False


def _write_binary_ply(
    file_obj,
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: Optional[np.ndarray] = None,
    colors: Optional[np.ndarray] = None
) -> None:
    """
    Write mesh data as a binary PLY file.
    
    Args:
        file_obj: Open file object to write to.
        vertices: Nx3 numpy array of vertex positions.
        faces: Mx3 numpy array of face indices.
        normals: Nx3 numpy array of vertex normals (optional).
        colors: Nx3 numpy array of vertex colors (optional).
    """
    # Write the header (in ASCII)
    file_obj.write(b"ply\n")
    file_obj.write(b"format binary_little_endian 1.0\n")
    file_obj.write(f"element vertex {len(vertices)}\n".encode())
    file_obj.write(b"property float x\n")
    file_obj.write(b"property float y\n")
    file_obj.write(b"property float z\n")

    if normals is not None:
        file_obj.write(b"property float nx\n")
        file_obj.write(b"property float ny\n")
        file_obj.write(b"property float nz\n")

    if colors is not None:
        file_obj.write(b"property uchar red\n")
        file_obj.write(b"property uchar green\n")
        file_obj.write(b"property uchar blue\n")

    file_obj.write(f"element face {len(faces)}\n".encode())
    file_obj.write(b"property list uchar int vertex_indices\n")
    file_obj.write(b"end_header\n")

    # Write vertex data
    for i in range(len(vertices)):
        file_obj.write(struct.pack('<fff', vertices[i, 0], vertices[i, 1], vertices[i, 2]))
        if normals is not None:
            file_obj.write(struct.pack('<fff', normals[i, 0], normals[i, 1], normals[i, 2]))
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
        vertices: Nx3 numpy array of vertex positions.
        height_map: 2D height map array (used to determine overall height range).
        color_map: Name of the colormap to use.
        
    Returns:
        Nx3 numpy array of RGB colors (0-255).
    """
    try:
        from matplotlib import cm

        # Compute the height range from the Z-coordinate of vertices
        z_min = np.min(vertices[:, 2])
        z_max = np.max(vertices[:, 2])
        z_range = z_max - z_min if z_max - z_min > 1e-10 else 1.0

        # Normalize Z values to the range [0, 1]
        normalized_z = (vertices[:, 2] - z_min) / z_range

        # Apply the colormap to obtain RGBA values, then convert to 8-bit RGB
        cmap = cm.get_cmap(color_map)
        rgba_colors = cmap(normalized_z)
        rgb_colors = (rgba_colors[:, :3] * 255).astype(np.uint8)

        return rgb_colors
    except ImportError as e:
        logger.warning(f"Matplotlib not available, using grayscale colors: {e}")
        # Fallback: generate grayscale colors based on height
        z_min = np.min(vertices[:, 2])
        z_max = np.max(vertices[:, 2])
        z_range = max(z_max - z_min, 1e-10)
        normalized_z = (vertices[:, 2] - z_min) / z_range
        grayscale = (normalized_z * 255).astype(np.uint8)
        rgb_colors = np.column_stack([grayscale, grayscale, grayscale])
        return rgb_colors


def adaptive_normal_estimation(
    pcd: o3d.geometry.PointCloud,
    k: int = 30,
    adaptive_factor: float = 2.0
) -> o3d.geometry.PointCloud:
    """
    Estimate normals adaptively for a given Open3D point cloud.
    
    This function computes the average distance to the k-th nearest neighbor for each point,
    then uses the global average (multiplied by adaptive_factor) as the search radius for normal estimation.
    
    Args:
        pcd: Open3D PointCloud.
        k: Number of nearest neighbors to consider.
        adaptive_factor: Factor to scale the average k-th neighbor distance to determine the search radius.
        
    Returns:
        The input point cloud with estimated and consistently oriented normals.
    """
    num_points = len(pcd.points)
    if num_points < k:
        k = num_points

    kdtree = o3d.geometry.KDTreeFlann(pcd)
    kth_distances = []

    # Loop over each point to compute the k-th nearest neighbor distance.
    for point in pcd.points:
        [_, _, distances] = kdtree.search_knn_vector_3d(point, k)
        kth_distance = np.sqrt(distances[-1])
        kth_distances.append(kth_distance)

    avg_kth_distance = np.mean(kth_distances)
    search_radius = avg_kth_distance * adaptive_factor

    logger.info(
        f"Adaptive normal estimation: using search radius = {search_radius:.4f} "
        f"(avg kth distance = {avg_kth_distance:.4f} * factor {adaptive_factor})"
    )

    # Estimate normals using the computed adaptive search radius
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=search_radius, max_nn=k))
    # Orient normals consistently using a tangent plane approach
    pcd.orient_normals_consistent_tangent_plane(k)

    return pcd
