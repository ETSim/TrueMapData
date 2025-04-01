"""
Utility functions for handling 3D meshes.

This module provides common functions for mesh manipulation, validation,
and optimization used by the various model exporters.
"""

import os
import time
import logging
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union, Callable

# Set up logging
logger = logging.getLogger(__name__)


def calculate_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Calculate vertex normals for a mesh using weighted contribution from adjacent faces.
    
    This function computes face normals and then averages them at shared vertices,
    weighting by face area for more accurate results on irregular meshes.
    
    Args:
        vertices: Nx3 array of vertex positions
        faces: Mx3 array of vertex indices
        
    Returns:
        Nx3 array of unit vertex normals
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


def calculate_heightmap_normals(height_map: np.ndarray, z_scale: float = 1.0) -> np.ndarray:
    """
    Calculate normal vectors for a height map using gradient-based approach.
    
    Args:
        height_map: 2D array of height values
        z_scale: Optional scaling factor for height values affecting normal steepness
        
    Returns:
        3D array of normal vectors with shape (height, width, 3)
    """
    height, width = height_map.shape
    normals = np.zeros((height, width, 3), dtype=np.float32)
    
    # Scale factor affects the normal direction (higher values make terrain appear steeper)
    xy_scale = 1.0
    
    # Calculate gradients using optimized numpy operations
    gradient_x = np.zeros_like(height_map, dtype=np.float32)
    gradient_y = np.zeros_like(height_map, dtype=np.float32)
    
    # Interior points - central differences for better accuracy
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
    
    # Apply scaling to control normal direction
    gradient_x *= xy_scale / z_scale 
    gradient_y *= xy_scale / z_scale
    
    # Construct normals (-grad_x, -grad_y, 1)
    normals[:, :, 0] = -gradient_x
    normals[:, :, 1] = -gradient_y
    normals[:, :, 2] = 1.0
    
    # Normalize to unit length
    norm = np.sqrt(np.sum(normals**2, axis=2, keepdims=True))
    # Avoid division by zero
    norm[norm < 1e-10] = 1.0
    normals /= norm
    
    return normals.astype(np.float32)


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
    tolerance: float = 1e-10,
    remove_isolated: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Optimize a mesh by merging duplicate vertices and removing degenerate faces.
    
    Args:
        vertices: Array of 3D vertices
        faces: Array of face indices
        tolerance: Distance tolerance for merging vertices
        remove_isolated: Whether to remove isolated vertices not used in any face
        
    Returns:
        Tuple of (optimized_vertices, optimized_faces)
    """
    # Dictionary to track merged vertices
    vertex_map = {}
    unique_vertices = []
    
    # Process each vertex
    for i, vertex in enumerate(vertices):
        # Convert to tuple for hashability (with rounding to handle floating point precision)
        v_tuple = tuple(np.round(vertex, decimals=int(-np.log10(tolerance))))
        
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
        try:
            new_face = [vertex_map[idx] for idx in face]
            
            # Skip degenerate faces (where vertices are duplicated)
            if len(set(new_face)) == len(face):
                optimized_faces.append(new_face)
        except KeyError:
            # Skip faces with invalid indices
            logger.warning("Skipping face with invalid vertex indices")
            continue
    
    # If there are no faces, return empty arrays
    if not optimized_faces:
        return np.array(unique_vertices), np.array([], dtype=np.int32).reshape(0, 3)
    
    optimized_faces = np.array(optimized_faces)
    
    # Remove isolated vertices if requested
    if remove_isolated:
        # Find all used vertices
        used_vertices = set(optimized_faces.flatten())
        
        # Create a new vertex array and mapping
        final_vertices = []
        remap = {}
        
        for i in range(len(unique_vertices)):
            if i in used_vertices:
                remap[i] = len(final_vertices)
                final_vertices.append(unique_vertices[i])
        
        # Update face indices
        final_faces = [[remap[v] for v in face] for face in optimized_faces]
        
        return np.array(final_vertices), np.array(final_faces)
    
    return np.array(unique_vertices), optimized_faces


def validate_heightmap(
    height_map: np.ndarray, 
    min_size: Tuple[int, int] = (2, 2),
    check_nan: bool = True
) -> bool:
    """
    Validate that a heightmap is suitable for processing.
    
    Args:
        height_map: 2D numpy array of height values
        min_size: Minimum dimensions (height, width)
        check_nan: Whether to check for NaN or infinity values
        
    Returns:
        True if valid, False otherwise
    """
    if height_map is None:
        logger.error("Height map is None")
        return False
    
    if not isinstance(height_map, np.ndarray):
        logger.error(f"Height map is not a numpy array, got {type(height_map)}")
        return False
    
    if height_map.size == 0:
        logger.error("Height map is empty")
        return False
        
    if height_map.ndim != 2:
        logger.error(f"Height map must be 2D, got {height_map.ndim}D")
        return False
        
    if height_map.shape[0] < min_size[0] or height_map.shape[1] < min_size[1]:
        logger.error(f"Height map too small: {height_map.shape}, minimum: {min_size}")
        return False
    
    if check_nan and (np.isnan(height_map).any() or np.isinf(height_map).any()):
        logger.error("Height map contains NaN or infinity values")
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
        # Handle empty string or None
        if not filepath:
            logger.error("Empty file path provided")
            return False
            
        directory = os.path.dirname(os.path.abspath(filepath))
        
        # If the directory is empty (current directory), return True
        if not directory:
            return True
            
        os.makedirs(directory, exist_ok=True)
        return True
    except (PermissionError, OSError) as e:
        logger.error(f"Error creating directory for {filepath}: {e}")
        return False


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


def simplify_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_percent: float,
    preserve_boundary: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simplify a mesh by reducing the number of faces.
    
    Args:
        vertices: Nx3 array of vertex positions
        faces: Mx3 array of face indices
        target_percent: Target percentage of faces to keep (0.0-1.0)
        preserve_boundary: Whether to preserve boundary edges
        
    Returns:
        Tuple of (simplified_vertices, simplified_faces)
    """
    try:
        import trimesh
        
        # Create a trimesh object
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # Calculate target face count
        target_faces = max(4, int(len(faces) * target_percent))
        
        # Simplify the mesh
        simplified_mesh = mesh.simplify_quadratic_decimation(target_faces)
        
        return np.array(simplified_mesh.vertices), np.array(simplified_mesh.faces)
    except ImportError:
        logger.warning("Trimesh not available, falling back to basic simplification")
        
        # Basic simplification: skip every N faces
        if target_percent >= 1.0:
            return vertices, faces
            
        skip = int(1.0 / target_percent)
        simplified_faces = faces[::skip]
        
        # Rebuild vertex array to only include used vertices
        used_vertices = set(simplified_faces.flatten())
        new_vertices = []
        vertex_map = {}
        
        for i in used_vertices:
            vertex_map[i] = len(new_vertices)
            new_vertices.append(vertices[i])
        
        # Remap face indices
        remapped_faces = np.array([[vertex_map[v] for v in face] for face in simplified_faces])
        
        return np.array(new_vertices), remapped_faces


def compute_mesh_metrics(vertices: np.ndarray, faces: np.ndarray) -> Dict[str, float]:
    """
    Compute basic metrics for a mesh.
    
    Args:
        vertices: Nx3 array of vertex positions
        faces: Mx3 array of face indices
        
    Returns:
        Dictionary with metrics
    """
    metrics = {}
    
    # Basic counts
    metrics['vertex_count'] = len(vertices)
    metrics['face_count'] = len(faces)
    
    # Bounding box
    metrics['bbox_min'] = vertices.min(axis=0).tolist()
    metrics['bbox_max'] = vertices.max(axis=0).tolist()
    metrics['bbox_size'] = (vertices.max(axis=0) - vertices.min(axis=0)).tolist()
    
    # Calculate total surface area
    total_area = 0.0
    for face in faces:
        v0 = vertices[face[0]]
        v1 = vertices[face[1]]
        v2 = vertices[face[2]]
        
        # Calculate face area using cross product
        edge1 = v1 - v0
        edge2 = v2 - v0
        face_normal = np.cross(edge1, edge2)
        face_area = np.linalg.norm(face_normal) * 0.5
        total_area += face_area
    
    metrics['surface_area'] = total_area
    
    # Compute mesh density (vertices per unit area)
    if total_area > 0:
        metrics['vertex_density'] = len(vertices) / total_area
    else:
        metrics['vertex_density'] = 0
    
    return metrics


class ProgressTracker:
    """Helper class for tracking and reporting progress."""
    
    def __init__(
        self, 
        callback: Optional[Callable[[float], None]] = None, 
        total_steps: int = 100, 
        description: str = ""
    ):
        """
        Initialize progress tracker.
        
        Args:
            callback: Function to call with progress updates (0.0-1.0)
            total_steps: Total number of steps to complete
            description: Description of the operation being tracked
        """
        self.callback = callback
        self.total_steps = max(1, total_steps)
        self.current_step = 0
        self.description = description
        self.start_time = time.time()
        self.last_update_time = 0
        self.min_update_interval = 0.1  # seconds
    
    def update(self, increment: int = 1, force: bool = False) -> None:
        """
        Update progress by the specified increment.
        
        Args:
            increment: Number of steps to increment by
            force: Force a callback update regardless of time interval
        """
        self.current_step += increment
        
        # Limit update frequency to avoid callback overhead
        current_time = time.time()
        if not force and current_time - self.last_update_time < self.min_update_interval:
            return
        
        progress = min(1.0, self.current_step / self.total_steps)
        
        if self.callback:
            self.callback(progress)
            
        self.last_update_time = current_time
        
    def finish(self) -> float:
        """
        Mark progress as complete and return elapsed time.
        
        Returns:
            Elapsed time in seconds
        """
        if self.callback:
            self.callback(1.0)
        
        elapsed = time.time() - self.start_time
        logger.info(f"Completed {self.description} in {elapsed:.2f}s")
        return elapsed


def calculate_terrain_complexity(
    height_map: np.ndarray, 
    smoothing: float = 1.0
) -> np.ndarray:
    """
    Calculate terrain complexity map based on gradients and curvature.
    
    This function computes a complexity map that identifies areas with high detail
    or significant features in the heightmap, which can guide adaptive triangulation.
    
    Args:
        height_map: 2D numpy array of height values
        smoothing: Optional smoothing factor (higher = smoother complexity map)
        
    Returns:
        2D array representing terrain complexity (normalized to 0.0-1.0 range)
    """
    from scipy.ndimage import gaussian_filter, sobel
    
    # Optional smoothing to reduce noise
    if smoothing > 0:
        smoothed = gaussian_filter(height_map, sigma=smoothing)
    else:
        smoothed = height_map
    
    # Calculate gradients in x and y directions
    grad_x = sobel(smoothed, axis=1)
    grad_y = sobel(smoothed, axis=0)
    
    # Gradient magnitude represents slope
    gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # Calculate second derivatives for curvature
    grad_xx = np.diff(np.diff(smoothed, axis=1, append=0), axis=1, append=0)
    grad_yy = np.diff(np.diff(smoothed, axis=0, append=0), axis=0, append=0)
    
    # Use Laplacian as a measure of curvature
    laplacian = np.abs(grad_xx + grad_yy)
    
    # Combine slope and curvature for overall complexity
    complexity = (
        0.7 * gradient_mag + 
        0.3 * laplacian
    )
    
    # Normalize to [0, 1] range
    complexity_min = np.min(complexity)
    complexity_max = np.max(complexity)
    
    if complexity_max > complexity_min:
        complexity = (complexity - complexity_min) / (complexity_max - complexity_min)
    else:
        complexity = np.zeros_like(complexity)
    
    return complexity


def sample_heightmap(
    height_map: np.ndarray, 
    x: float, 
    y: float, 
    interpolation: str = 'bilinear'
) -> float:
    """
    Sample height value from a heightmap with interpolation.
    
    Args:
        height_map: 2D numpy array of height values
        x, y: Coordinates to sample (can be floating point)
        interpolation: Interpolation method ('nearest', 'bilinear', or 'bicubic')
        
    Returns:
        Interpolated height value
        
    Raises:
        ValueError: If an invalid interpolation method is specified
    """
    rows, cols = height_map.shape
    
    # Ensure coordinates are within bounds
    x = min(max(0, x), cols - 1)
    y = min(max(0, y), rows - 1)
    
    if interpolation == 'nearest':
        # Simple nearest neighbor interpolation
        return height_map[int(round(y)), int(round(x))]
    
    elif interpolation == 'bilinear':
        # Bilinear interpolation
        x0, y0 = int(x), int(y)
        x1, y1 = min(x0 + 1, cols - 1), min(y0 + 1, rows - 1)
        
        # Calculate fractional parts
        fx, fy = x - x0, y - y0
        
        # Get corner values
        v00 = height_map[y0, x0]
        v01 = height_map[y0, x1]
        v10 = height_map[y1, x0]
        v11 = height_map[y1, x1]
        
        # Interpolate along x
        v0 = v00 * (1 - fx) + v01 * fx
        v1 = v10 * (1 - fx) + v11 * fx
        
        # Interpolate along y
        return v0 * (1 - fy) + v1 * fy
    
    elif interpolation == 'bicubic':
        try:
            from scipy.interpolate import RectBivariateSpline
            
            # Get bounds for interpolation window
            x0, y0 = max(0, int(x) - 1), max(0, int(y) - 1)
            x1, y1 = min(cols - 1, int(x) + 2), min(rows - 1, int(y) + 2)
            
            # Create local coordinate grid
            x_grid = np.arange(x0, x1 + 1)
            y_grid = np.arange(y0, y1 + 1)
            
            # Extract local patch
            local_patch = height_map[y0:y1+1, x0:x1+1]
            
            # Create spline interpolator
            spline = RectBivariateSpline(y_grid, x_grid, local_patch, kx=3, ky=3)
            
            # Evaluate at requested coordinates
            return float(spline(y, x))
        except ImportError:
            logger.warning("SciPy not available, falling back to bilinear interpolation")
            # Fall back to bilinear
            return sample_heightmap(height_map, x, y, 'bilinear')
    
    else:
        raise ValueError(f"Invalid interpolation method: {interpolation}")


def validate_mesh(
    vertices: np.ndarray, 
    faces: np.ndarray
) -> Tuple[bool, List[str]]:
    """
    Validate mesh data for common issues.
    
    Args:
        vertices: Array of vertex positions
        faces: Array of face indices
        
    Returns:
        Tuple of (is_valid, issues) where issues is a list of problems found
    """
    issues = []
    
    # Check for empty mesh
    if len(vertices) == 0:
        issues.append("Empty mesh (no vertices)")
    
    if len(faces) == 0:
        issues.append("Empty mesh (no faces)")
    
    if not issues:  # Only continue if we have vertices and faces
        # Check for invalid indices
        max_index = np.max(faces) if len(faces) > 0 else -1
        if max_index >= len(vertices):
            issues.append(f"Invalid vertex indices: max index {max_index} >= vertex count {len(vertices)}")
        
        # Check for negative indices
        if np.min(faces) < 0:
            issues.append(f"Invalid negative vertex indices found")
        
        # Check for degenerate faces (duplicated vertices in a face)
        for i, face in enumerate(faces):
            if len(set(face)) != len(face):
                issues.append(f"Degenerate face at index {i}: {face}")
        
        # Check for NaN or infinite values
        if np.isnan(vertices).any():
            issues.append("Mesh contains NaN vertex coordinates")
        
        if np.isinf(vertices).any():
            issues.append("Mesh contains infinite vertex coordinates")
    
    return len(issues) == 0, issues


def export_mesh(
    vertices: np.ndarray, 
    faces: np.ndarray, 
    filepath: str, 
    format: Optional[str] = None
) -> bool:
    """
    Export mesh to file in the specified format.
    
    Args:
        vertices: Array of vertex positions
        faces: Array of face indices
        filepath: Path to output file
        format: Output format (inferred from extension if None)
        
    Returns:
        True if successful, False otherwise
    """
    # Validate mesh before export
    is_valid, issues = validate_mesh(vertices, faces)
    if not is_valid:
        logger.error(f"Cannot export invalid mesh: {', '.join(issues)}")
        return False
    
    # Determine format from extension if not specified
    if format is None:
        format = os.path.splitext(filepath)[1].lower()[1:]
    
    # Ensure output directory exists
    if not ensure_directory_exists(filepath):
        logger.error(f"Failed to create directory for {filepath}")
        return False
    
    try:
        if format == 'obj':
            return _export_obj(vertices, faces, filepath)
        elif format == 'stl':
            return _export_stl(vertices, faces, filepath)
        elif format == 'ply':
            return _export_ply(vertices, faces, filepath)
        else:
            logger.error(f"Unsupported export format: {format}")
            return False
    except Exception as e:
        logger.error(f"Error exporting mesh to {filepath}: {e}")
        return False


def _export_obj(vertices: np.ndarray, faces: np.ndarray, filepath: str) -> bool:
    """Export mesh to Wavefront OBJ format."""
    try:
        with open(filepath, 'w') as f:
            # Write header
            f.write("# OBJ file created by TMD\n")
            
            # Write vertices
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
            # Write faces (OBJ uses 1-based indexing)
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        
        logger.info(f"Exported mesh with {len(vertices)} vertices and {len(faces)} faces to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Failed to export OBJ: {e}")
        return False


def _export_stl(vertices: np.ndarray, faces: np.ndarray, filepath: str) -> bool:
    """Export mesh to STL format."""
    try:
        # Calculate face normals for STL
        normals = calculate_face_normals(vertices, faces)
        
        # Determine if we should use binary or ASCII format
        use_binary = filepath.lower().endswith('.stl')
        
        if use_binary:
            return _export_binary_stl(vertices, faces, normals, filepath)
        else:
            return _export_ascii_stl(vertices, faces, normals, filepath)
    except Exception as e:
        logger.error(f"Failed to export STL: {e}")
        return False


def _export_binary_stl(
    vertices: np.ndarray, 
    faces: np.ndarray, 
    normals: np.ndarray, 
    filepath: str
) -> bool:
    """Export mesh to binary STL format."""
    try:
        with open(filepath, 'wb') as f:
            # Write header (80 bytes)
            f.write(b'TMD Generated STL File'.ljust(80, b'\0'))
            
            # Write number of triangles (4 bytes)
            f.write(np.array([len(faces)], dtype=np.uint32).tobytes())
            
            # Write triangles
            for i, face in enumerate(faces):
                # Normal vector (3 floats)
                f.write(normals[i].astype(np.float32).tobytes())
                
                # Vertex coordinates (9 floats)
                for idx in face:
                    f.write(vertices[idx].astype(np.float32).tobytes())
                
                # Attribute byte count (unused, 2 bytes)
                f.write(np.array([0], dtype=np.uint16).tobytes())
        
        logger.info(f"Exported binary STL with {len(faces)} triangles to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Failed to export binary STL: {e}")
        return False


def _export_ascii_stl(
    vertices: np.ndarray, 
    faces: np.ndarray, 
    normals: np.ndarray, 
    filepath: str
) -> bool:
    """Export mesh to ASCII STL format."""
    try:
        with open(filepath, 'w') as f:
            f.write("solid TMD_Generated\n")
            
            for i, face in enumerate(faces):
                nx, ny, nz = normals[i]
                f.write(f"  facet normal {nx:.6e} {ny:.6e} {nz:.6e}\n")
                f.write("    outer loop\n")
                
                for idx in face:
                    x, y, z = vertices[idx]
                    f.write(f"      vertex {x:.6e} {y:.6e} {z:.6e}\n")
                
                f.write("    endloop\n")
                f.write("  endfacet\n")
            
            f.write("endsolid TMD_Generated\n")
        
        logger.info(f"Exported ASCII STL with {len(faces)} triangles to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Failed to export ASCII STL: {e}")
        return False


def _export_ply(vertices: np.ndarray, faces: np.ndarray, filepath: str) -> bool:
    """Export mesh to Stanford PLY format."""
    try:
        with open(filepath, 'w') as f:
            # Write header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(vertices)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write(f"element face {len(faces)}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")
            
            # Write vertices
            for v in vertices:
                f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
            # Write faces
            for face in faces:
                f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
        
        logger.info(f"Exported PLY with {len(vertices)} vertices and {len(faces)} faces to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Failed to export PLY: {e}")
        return False


class MeshProcessingConfig:
    """Configuration class for mesh processing parameters."""
    
    default_values = {
        'z_scale': 1.0,
        'base_height': 0.0,
        'max_triangles': 100000,
        'error_threshold': 0.001,
        'coordinate_system': 'right-handed',
        'origin_at_zero': True,
        'detail_boost': 1.0,
        'uv_mapping': 'planar',
        'x_scale': 1.0,
        'y_scale': 1.0,
        'invert_base': False,
        'triangulation_method': 'adaptive'
    }
    
    def __init__(self, **kwargs):
        """
        Initialize configuration with default values and overrides.
        
        Args:
            **kwargs: Configuration parameters to override defaults
        
        Raises:
            ValueError: If an unknown parameter is provided
        """
        # Set default values
        for key, value in self.default_values.items():
            setattr(self, key, value)
            
        # Override with provided values
        for key, value in kwargs.items():
            if key in self.default_values:
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
    
    def as_dict(self) -> Dict[str, Any]:
        """
        Get configuration as a dictionary.
        
        Returns:
            Dictionary of configuration parameters
        """
        return {key: getattr(self, key) for key in self.default_values}
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate configuration values.
        
        Returns:
            Tuple of (is_valid, issues) where issues is a list of problems found
        """
        issues = []
        
        # Validate numeric ranges
        if self.z_scale <= 0:
            issues.append(f"z_scale must be positive, got {self.z_scale}")
        
        if self.base_height < 0:
            issues.append(f"base_height cannot be negative, got {self.base_height}")
        
        if self.max_triangles <= 0:
            issues.append(f"max_triangles must be positive, got {self.max_triangles}")
            
        if self.error_threshold <= 0:
            issues.append(f"error_threshold must be positive, got {self.error_threshold}")
            
        # Validate enums
        if self.coordinate_system not in ['right-handed', 'left-handed']:
            issues.append(f"coordinate_system must be 'right-handed' or 'left-handed', got '{self.coordinate_system}'")
            
        if self.uv_mapping not in ['planar', 'cylindrical', 'spherical']:
            issues.append(f"uv_mapping must be 'planar', 'cylindrical', or 'spherical', got '{self.uv_mapping}'")
            
        if self.triangulation_method not in ['adaptive', 'regular', 'quadtree']:
            issues.append(f"triangulation_method must be 'adaptive', 'regular', or 'quadtree', got '{self.triangulation_method}'")
        
        return len(issues) == 0, issues