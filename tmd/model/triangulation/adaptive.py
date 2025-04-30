"""
Adaptive triangulator for heightmaps.

This module provides functionality for adaptive triangulation of heightmaps,
which generates a more efficient mesh by using fewer triangles in flat areas
and more triangles in detailed areas, resulting in optimal model size.
"""

import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any, Set, Union, Callable

from .base import BaseTriangulator
from ..utils.heightmap import calculate_terrain_complexity

# Set up logging
logger = logging.getLogger(__name__)

def to_16bit_grayscale(self, height_map: np.ndarray) -> np.ndarray:
        """
        Convert a heightmap to 16-bit grayscale format.
        
        Args:
            height_map: Input heightmap array
        Returns:
            16-bit normalized heightmap
        """
        
        # Ensure floating point for calculations
        height_map = height_map.astype(np.float32)
        
        # Normalize to [0, 1] range
        min_val = np.min(height_map)
        max_val = np.max(height_map)
        height_range = max_val - min_val
        
        if height_range > 0:
            height_map = (height_map - min_val) / height_range
        else:
            height_map = np.zeros_like(height_map)
        
        # Convert to 16-bit integer range [0, 65535]
        height_map = (height_map * 65535).astype(np.uint16)
        
        # Convert back to float32 but preserve 16-bit precision
        height_map = height_map.astype(np.float32) / 65535.0
        
        logger.debug(f"Converted heightmap: shape={height_map.shape}, dtype={height_map.dtype}, range=[{height_map.min():.3f}, {height_map.max():.3f}]")
        
        return height_map


class AdaptiveTriangulator(BaseTriangulator):
    """
    Class for adaptively triangulating a heightmap.
    
    This triangulator creates a mesh with varying triangle density based on the
    local curvature and detail level of the heightmap. Areas with high detail
    receive more triangles, while flat areas use fewer triangles.
    """
    
    def __init__(
        self,
        height_map: np.ndarray,
        max_triangles: int = 50000,  # Reduced max triangles
        error_threshold: float = 0.01,  # Increased error threshold
        min_area_fraction: float = 0.001,  # Increased min area
        z_scale: float = 1.0,
        detail_boost: float = 0.5,  # Reduced detail boost
        progress_callback: Optional[Callable[[float], None]] = None
    ):
        """
        Initialize the adaptive triangulator.
        
        Args:
            height_map: 2D array of height values
            max_triangles: Maximum number of triangles to generate
            error_threshold: Maximum allowed error for approximation (lower = more detail)
            min_area_fraction: Minimum allowed triangle area as fraction of total area
            z_scale: Z-scaling factor for height values
            detail_boost: Factor to boost detail in high-complexity areas (1.0 = normal)
            progress_callback: Optional callback function for progress reporting
        """
        super().__init__(
            height_map=height_map,
            z_scale=z_scale,
            max_triangles=max_triangles,
            error_threshold=error_threshold,
            progress_callback=progress_callback
        )
        logger.info(f"Initializing AdaptiveTriangulator with {height_map.shape} heightmap")
        logger.info(f"Parameters: max_triangles={max_triangles}, error_threshold={error_threshold}, detail_boost={detail_boost}")
        
        self.min_area = min_area_fraction * height_map.shape[0] * height_map.shape[1]
        self.detail_boost = detail_boost
        
        # Calculate terrain complexity to guide the triangulation
        self.complexity_map = self.calculate_complexity_map()
        
        # Initialize internal state
        self.vertices = []  # List of (x, y, z) vertex coordinates
        self.indices = []   # List of triangle vertex indices
        self.vertex_map = {}  # Maps (x, y) grid coordinates to vertex indices
        
        logger.debug(f"AdaptiveTriangulator initialized with {self.height_map.shape} heightmap")
    
    def calculate_complexity_map(self) -> np.ndarray:
        """
        Calculate a complexity map to guide where to add more detail.
        
        Returns:
            2D array representing local terrain complexity
        """
        return calculate_terrain_complexity(self.height_map, smoothing=1.0)
    
    def _add_vertex(self, row: int, col: int) -> int:
        """
        Add a vertex at the specified grid position.
        
        Args:
            row: Row index in heightmap
            col: Column index in heightmap
            
        Returns:
            Index of the new or existing vertex
        """
        # Ensure row and col are within bounds
        row = max(0, min(row, self.height_map.shape[0] - 1))
        col = max(0, min(col, self.height_map.shape[1] - 1))
        
        # Check if vertex already exists
        if (row, col) in self.vertex_map:
            return self.vertex_map[(row, col)]
        
        # Create new vertex
        z = float(self.height_map[row, col]) * self.z_scale  # Apply z_scale
        vertex_idx = len(self.vertices)
        self.vertices.append([float(col), float(row), z])
        self.vertex_map[(row, col)] = vertex_idx
        
        return vertex_idx
    
    def _needs_subdivision(self, triangle_idx: int) -> bool:
        """
        Determine if a triangle needs to be subdivided based on error and complexity.
        
        Args:
            triangle_idx: Index of the triangle to check
            
        Returns:
            True if the triangle should be subdivided, False otherwise
        """
        # Get triangle vertices
        triangle = self.indices[triangle_idx]
        v1 = self.vertices[triangle[0]]
        v2 = self.vertices[triangle[1]]
        v3 = self.vertices[triangle[2]]
        
        # Calculate triangle properties
        area = self._triangle_area(v1, v2, v3)
        
        # Skip very small triangles
        if area < self.min_area:
            return False
        
        # Calculate local complexity to adjust error threshold
        local_complexity = self._get_triangle_complexity(v1, v2, v3)
        
        # Apply detail boost: higher complexity = lower error threshold = more detail
        adjusted_threshold = self.error_threshold / (1.0 + local_complexity * self.detail_boost)
        
        # Check if the triangle approximation error is too high
        error = self._approximation_error(triangle_idx)
        return error > adjusted_threshold
    
    def _get_triangle_complexity(self, v1: List[float], v2: List[float], v3: List[float]) -> float:
        """
        Get the average complexity within a triangle.
        
        Args:
            v1, v2, v3: Triangle vertices
            
        Returns:
            Average complexity value
        """
        # Convert to grid coordinates
        x1, y1, _ = v1
        x2, y2, _ = v2
        x3, y3, _ = v3
        
        # Calculate triangle bounding box
        min_row = max(0, min(int(y1), int(y2), int(y3)))
        max_row = min(self.complexity_map.shape[0]-1, max(int(y1), int(y2), int(y3)))
        min_col = max(0, min(int(x1), int(x2), int(x3)))
        max_col = min(self.complexity_map.shape[1]-1, max(int(x1), int(x2), int(x3)))
        
        # If triangle is tiny, sample a single point
        if min_row == max_row or min_col == max_col:
            return self.complexity_map[min_row, min_col]
        
        # Sample points and find average complexity
        complexity_sum = 0.0
        point_count = 0
        
        for r in range(min_row, max_row+1):
            for c in range(min_col, max_col+1):
                # Check if point is inside triangle
                if not self._point_in_triangle((c, r), (int(x1), int(y1)), (int(x2), int(y2)), (int(x3), int(y3))):
                    continue
                
                # Add complexity value
                complexity_sum += self.complexity_map[r, c]
                point_count += 1
        
        # Return average complexity (or 0 if no points were inside)
        return complexity_sum / max(1, point_count)
    
    def _approximation_error(self, triangle_idx: int) -> float:
        """
        Calculate the approximation error for a triangle against the actual heightmap.
        
        Args:
            triangle_idx: Index of the triangle to check
            
        Returns:
            Maximum error between triangle plane and actual heightmap
        """
        # Get triangle vertices
        triangle = self.indices[triangle_idx]
        v1 = self.vertices[triangle[0]]
        v2 = self.vertices[triangle[1]]
        v3 = self.vertices[triangle[2]]
        
        # Convert to grid coordinates
        x1, y1, z1 = v1
        x2, y2, z2 = v2
        x3, y3, z3 = v3
        
        # Bounding box for the triangle
        min_row = max(0, min(int(y1), int(y2), int(y3)))
        max_row = min(self.height_map.shape[0]-1, max(int(y1), int(y2), int(y3)))
        min_col = max(0, min(int(x1), int(x2), int(x3)))
        max_col = min(self.height_map.shape[1]-1, max(int(x1), int(x2), int(x3)))
        
        # If triangle is tiny, return a small error
        if min_row == max_row or min_col == max_col:
            return 0.0
        
        # Sample points and find maximum error
        max_error = 0.0
        
        for r in range(min_row, max_row+1):
            for c in range(min_col, max_col+1):
                # Check if point is inside triangle
                if not self._point_in_triangle((c, r), (int(x1), int(y1)), (int(x2), int(y2)), (int(x3), int(y3))):
                    continue
                
                # Get actual height value
                actual_z = float(self.height_map[r, c]) * self.z_scale
                
                # Interpolate height at this point
                interp_z = self._barycentric_interpolate(
                    (c, r), (x1, y1, z1), (x2, y2, z2), (x3, y3, z3)
                )
                
                # Calculate error (weighted by local complexity)
                error = abs(actual_z - interp_z)
                max_error = max(max_error, error)
        
        return max_error
    
    def _barycentric_interpolate(
        self, 
        p: Tuple[float, float], 
        v1: Tuple[float, float, float], 
        v2: Tuple[float, float, float], 
        v3: Tuple[float, float, float]
    ) -> float:
        """
        Interpolate z-value at a point using barycentric coordinates.
        
        Args:
            p: Point to interpolate at (x, y)
            v1, v2, v3: Triangle vertices (x, y, z)
            
        Returns:
            Interpolated z-value
        """
        # Extract coordinates
        px, py = p
        x1, y1, z1 = v1
        x2, y2, z2 = v2
        x3, y3, z3 = v3
        
        # Calculate barycentric coordinates
        denominator = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
        
        if abs(denominator) < 1e-10:  # Avoid division by zero
            return (z1 + z2 + z3) / 3.0  # Return average height
            
        a = ((y2 - y3) * (px - x3) + (x3 - x2) * (py - y3)) / denominator
        b = ((y3 - y1) * (px - x3) + (x1 - x3) * (py - y3)) / denominator
        c = 1.0 - a - b
        
        # Clamp coordinates in case of numerical issues
        a = max(0, min(1, a))
        b = max(0, min(1, b))
        c = max(0, min(1, 1 - a - b))
        
        # Interpolate z using barycentric coordinates
        return a * z1 + b * z2 + c * z3
    
    def _subdivide_triangle(self, triangle_idx: int) -> List[int]:
        """
        Subdivide a triangle by splitting its longest edge.
        
        Args:
            triangle_idx: Index of the triangle to subdivide
            
        Returns:
            List of indices of the new triangles
        """
        # Get the triangle to subdivide
        triangle = self.indices[triangle_idx]
        v1 = self.vertices[triangle[0]]
        v2 = self.vertices[triangle[1]]
        v3 = self.vertices[triangle[2]]
        
        # Find the longest edge
        edge_lengths = [
            (0, 1, self._distance(v1, v2)),  # Edge 0-1
            (1, 2, self._distance(v2, v3)),  # Edge 1-2
            (2, 0, self._distance(v3, v1))   # Edge 2-0
        ]
        
        # Sort by descending length and get longest edge
        edge_lengths.sort(key=lambda e: e[2], reverse=True)
        longest_edge = edge_lengths[0]
        
        # Get indices of the two endpoints of the longest edge
        i1 = triangle[longest_edge[0]]
        i2 = triangle[longest_edge[1]]
        
        # Get the remaining vertex (not part of longest edge)
        i3 = triangle[3 - longest_edge[0] - longest_edge[1]]
        
        # Get grid coordinates for the endpoints
        x1, y1, _ = self.vertices[i1]
        x2, y2, _ = self.vertices[i2]
        
        # Find midpoint in grid coordinates with proper rounding
        mid_row = int(round((y1 + y2) / 2))
        mid_col = int(round((x1 + x2) / 2))
        
        # Add vertex at midpoint
        mid_idx = self._add_vertex(mid_row, mid_col)
        
        # Replace the original triangle with two new ones
        self.indices[triangle_idx] = [i1, mid_idx, i3]
        new_triangle_idx = len(self.indices)
        self.indices.append([i2, mid_idx, i3])
        
        return [triangle_idx, new_triangle_idx]
    
    def _triangle_area(self, v1: List[float], v2: List[float], v3: List[float]) -> float:
        """
        Calculate the area of a triangle.
        
        Args:
            v1, v2, v3: Vertices of the triangle
            
        Returns:
            Area of the triangle
        """
        # Use cross product method for area
        return 0.5 * abs((v1[0] * (v2[1] - v3[1]) + 
                          v2[0] * (v3[1] - v1[1]) + 
                          v3[0] * (v1[1] - v2[1])))
    
    def _distance(self, v1: List[float], v2: List[float]) -> float:
        """
        Calculate the Euclidean distance between two vertices.
        
        Args:
            v1, v2: Vertices to calculate distance between
            
        Returns:
            Distance between vertices
        """
        return ((v1[0] - v2[0]) ** 2 + 
                (v1[1] - v2[1]) ** 2 + 
                (v1[2] - v2[2]) ** 2) ** 0.5
    
    def _point_in_triangle(
        self, 
        p: Tuple[float, float], 
        v1: Tuple[float, float], 
        v2: Tuple[float, float], 
        v3: Tuple[float, float]
    ) -> bool:
        """
        Check if a point is inside a triangle using barycentric coordinates.
        
        Args:
            p: Point to check
            v1, v2, v3: Vertices of the triangle
            
        Returns:
            True if point is inside triangle, False otherwise
        """
        # Use efficient barycentric method
        def compute_sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
        
        d1 = compute_sign(p, v1, v2)
        d2 = compute_sign(p, v2, v3)
        d3 = compute_sign(p, v3, v1)
        
        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        
        # If all have the same sign (all positive or all negative), point is inside
        return not (has_neg and has_pos)
    
    def triangulate(self) -> Tuple[List[List[float]], List[List[int]]]:
        """
        Run the adaptive triangulation algorithm.
        
        Returns:
            Tuple of (vertices, triangles) where vertices is a list of [x, y, z] coordinates
            and triangles is a list of [a, b, c] indices.
        """
        logger.info("Starting adaptive triangulation")
        
        # Report initial progress
        if self.progress_callback:
            self.progress_callback(0.05)  # Initialized
            
        # Start with the corners of the heightmap
        rows, cols = self.height_map.shape
        
        # Create initial vertices at the corners
        self._add_vertex(0, 0)
        self._add_vertex(0, cols-1)
        self._add_vertex(rows-1, 0)
        self._add_vertex(rows-1, cols-1)
        
        # Create initial triangles
        self.indices.append([0, 1, 3])  # Top-left, top-right, bottom-right
        self.indices.append([0, 3, 2])  # Top-left, bottom-right, bottom-left
        
        if self.progress_callback:
            self.progress_callback(0.1)  # Initial triangles created
        
        logger.debug(f"Initial vertices: {len(self.vertices)}")
        
        # Refine the mesh
        triangles_to_check = list(range(len(self.indices)))
        total_checks = 0
        progress_interval = max(1, self.max_triangles // 100)  # For progress reporting
        
        while triangles_to_check and len(self.indices) < self.max_triangles:
            total_checks += 1
            
            # Report progress periodically
            if self.progress_callback and total_checks % progress_interval == 0:
                progress = min(0.9, len(self.indices) / self.max_triangles)
                self.progress_callback(progress)
                
            if total_checks % 1000 == 0:
                logger.debug(f"Processed {total_checks} triangles, current count: {len(self.indices)}")
            
            # Get next triangle to check
            triangle_idx = triangles_to_check.pop(0)
            
            # Check if this triangle needs subdivision
            if self._needs_subdivision(triangle_idx):
                # Skip if we'll exceed max_triangles
                if len(self.indices) + 1 > self.max_triangles:
                    logger.info(f"Reached maximum triangle count ({self.max_triangles})")
                    break
                    
                # Subdivide the triangle
                new_triangle_indices = self._subdivide_triangle(triangle_idx)
                triangles_to_check.extend(new_triangle_indices)
        
        if self.progress_callback:
            self.progress_callback(0.95)  # Subdivision complete
        
        # Update statistics
        vertices_list = [[float(v[0]), float(v[1]), float(v[2])] for v in self.vertices]
        indices_list = [[int(i[0]), int(i[1]), int(i[2])] for i in self.indices]
        
        # Finalize statistics
        self.finalize_stats(vertices_list, indices_list)
        
        logger.info(f"Triangulation complete: {len(self.indices)} triangles from {len(self.vertices)} vertices")
        
        # Final progress report
        if self.progress_callback:
            self.progress_callback(1.0)
        
        return vertices_list, indices_list


def triangulate_heightmap(
    height_map: np.ndarray,
    max_triangles: int = 100000,
    error_threshold: float = 0.001,
    z_scale: float = 1.0,
    detail_boost: float = 1.0,
    progress_callback: Optional[Callable[[float], None]] = None
) -> Tuple[List[List[float]], List[List[int]], Dict[str, Any]]:
    """
    Convenience function to triangulate a heightmap in one call.
    
    Args:
        height_map: 2D numpy array of height values
        max_triangles: Maximum number of triangles to generate
        error_threshold: Maximum allowed error for approximation
        z_scale: Z-scaling factor for height values
        detail_boost: Factor to boost detail in high-complexity areas
        progress_callback: Optional callback function for progress reporting
        
    Returns:
        Tuple of (vertices, faces, statistics)
    """
    triangulator = AdaptiveTriangulator(
        height_map=height_map,
        max_triangles=max_triangles,
        error_threshold=error_threshold,
        z_scale=z_scale,
        detail_boost=detail_boost,
        progress_callback=progress_callback
    )
    
    vertices, faces = triangulator.triangulate()
    stats = triangulator.get_statistics()
    
    return vertices, faces, stats