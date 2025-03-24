""".

Adaptive triangulator for heightmaps.

This module provides functionality for adaptive triangulation of heightmaps,
which generates a more efficient mesh by using fewer triangles in flat areas.
"""

import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any, Set

# Set up logging
logger = logging.getLogger(__name__)

class AdaptiveTriangulator:
    """.

    Class for adaptively triangulating a heightmap.
    
    This triangulator creates a mesh with varying triangle density based on the
    local curvature of the heightmap.
    """
    
    def __init__(
        self,
        height_map: np.ndarray,
        max_triangles: int = 100000,
        error_threshold: float = 0.001,
        min_area_fraction: float = 0.0001,
        z_scale: float = 1.0
    ):
        """.

        Initialize the adaptive triangulator.
        
        Args:
            height_map: 2D array of height values
            max_triangles: Maximum number of triangles to generate
            error_threshold: Maximum allowed error for approximation
            min_area_fraction: Minimum allowed triangle area as fraction of total area
            z_scale: Z-scaling factor for height values
        """
        self.height_map = height_map
        self.max_triangles = max_triangles
        self.error_threshold = error_threshold
        self.min_area = min_area_fraction * height_map.shape[0] * height_map.shape[1]
        self.z_scale = z_scale
        
        # Initialize internal state
        self.vertices = []  # List of (x, y, z) vertex coordinates
        self.indices = []   # List of triangle vertex indices
        self.vertex_map = {}  # Maps (x, y) grid coordinates to vertex indices
    
    def run(self, max_error: Optional[float] = None, max_triangles: Optional[int] = None, **kwargs) -> Tuple[List[List[float]], List[List[int]]]:
        """.

        Run the adaptive triangulation algorithm.
        
        Args:
            max_error: Optional override for error threshold
            max_triangles: Optional override for maximum number of triangles
            **kwargs: Additional parameters for compatibility with test cases
            
        Returns:
            Tuple of (vertices, triangles) where vertices is a list of [x, y, z] coordinates
            and triangles is a list of [a, b, c] indices.
        """
        # Override parameters if provided
        if max_error is not None:
            self.error_threshold = max_error
        
        # Override max_triangles if provided
        if max_triangles is not None:
            self.max_triangles = max_triangles
            
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
        
        # Refine the mesh
        triangles_to_check = list(range(len(self.indices)))
        
        while triangles_to_check and len(self.indices) < self.max_triangles:
            triangle_idx = triangles_to_check.pop(0)
            
            # Check if this triangle needs subdivision
            if self._needs_subdivision(triangle_idx):
                # Subdivide the triangle only if we won't exceed max_triangles
                if len(self.indices) + 1 > self.max_triangles:
                    break
                    
                # Subdivide the triangle
                new_triangle_indices = self._subdivide_triangle(triangle_idx)
                triangles_to_check.extend(new_triangle_indices)
        
        logger.info(f"Adaptive triangulation complete. Generated {len(self.indices)} triangles.")
        
        # Convert vertices to list format
        vertices_list = [[float(v[0]), float(v[1]), float(v[2])] for v in self.vertices]
        
        # Ensure indices are properly formatted
        indices_list = [[int(i[0]), int(i[1]), int(i[2])] for i in self.indices]
        
        return vertices_list, indices_list
    
    def _add_vertex(self, row: int, col: int) -> int:
        """.

        Add a vertex at the specified grid position.
        
        Args:
            row: Row index in heightmap
            col: Column index in heightmap
            
        Returns:
            Index of the new or existing vertex
        """
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
        """.

        Determine if a triangle needs to be subdivided.
        
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
        
        # Check if the triangle approximation error is too high
        error = self._approximation_error(triangle_idx)
        return error > self.error_threshold
    
    def _approximation_error(self, triangle_idx: int) -> float:
        """.

        Calculate the approximation error for a triangle.
        
        Args:
            triangle_idx: Index of the triangle to check
            
        Returns:
            Maximum error between triangle and actual heightmap
        """
        # Get triangle vertices
        triangle = self.indices[triangle_idx]
        v1 = self.vertices[triangle[0]]
        v2 = self.vertices[triangle[1]]
        v3 = self.vertices[triangle[2]]
        
        # Get vertex coordinates
        x1, y1, z1 = v1
        x2, y2, z2 = v2
        x3, y3, z3 = v3
        
        # Convert to grid coordinates
        r1, c1 = int(y1), int(x1)
        r2, c2 = int(y2), int(x2)
        r3, c3 = int(y3), int(x3)
        
        # Find bounding box for the triangle
        min_row = max(0, min(r1, r2, r3))
        max_row = min(self.height_map.shape[0]-1, max(r1, r2, r3))
        min_col = max(0, min(c1, c2, c3))
        max_col = min(self.height_map.shape[1]-1, max(c1, c2, c3))
        
        # If the triangle is tiny, return a small error
        if min_row == max_row or min_col == max_col:
            return 0.0
        
        # Sample points in the triangle and find maximum error
        max_error = 0.0
        
        for r in range(min_row, max_row+1):
            for c in range(min_col, max_col+1):
                # Check if point is inside the triangle
                if not self._point_in_triangle((c, r), (c1, r1), (c2, r2), (c3, r3)):
                    continue
                
                # Interpolate the z value at this point
                interp_z = self._interpolate_z((c, r), (x1, y1, z1), (x2, y2, z2), (x3, y3, z3))
                
                # Calculate error
                actual_z = float(self.height_map[r, c]) * self.z_scale  # Apply z_scale
                error = abs(actual_z - interp_z)
                max_error = max(max_error, error)
        
        return max_error
    
    def _subdivide_triangle(self, triangle_idx: int) -> List[int]:
        """.

        Subdivide a triangle by adding vertices at edge midpoints.
        
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
        edges = [
            (0, 1, self._distance(v1, v2)),
            (1, 2, self._distance(v2, v3)),
            (2, 0, self._distance(v3, v1))
        ]
        edges.sort(key=lambda e: e[2], reverse=True)
        
        # Split the longest edge
        longest_edge = edges[0]
        i1 = triangle[longest_edge[0]]
        i2 = triangle[longest_edge[1]]
        
        # Get grid coordinates for the two endpoints
        x1, y1, _ = self.vertices[i1]
        x2, y2, _ = self.vertices[i2]
        
        # Find midpoint in grid coordinates
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        
        # Convert to integer grid coordinates
        mid_row = int(round(mid_y))
        mid_col = int(round(mid_x))
        
        # Add vertex at midpoint
        mid_idx = self._add_vertex(mid_row, mid_col)
        
        # Replace the original triangle with two new ones
        self.indices[triangle_idx] = [i1, mid_idx, triangle[3-longest_edge[0]-longest_edge[1]]]
        new_triangle_idx = len(self.indices)
        self.indices.append([i2, mid_idx, triangle[3-longest_edge[0]-longest_edge[1]]])
        
        return [triangle_idx, new_triangle_idx]
    
    def _triangle_area(self, v1: List[float], v2: List[float], v3: List[float]) -> float:
        """.

        Calculate the area of a triangle.
        
        Args:
            v1, v2, v3: Vertices of the triangle
            
        Returns:
            Area of the triangle
        """
        # Project to 2D for simplicity
        return 0.5 * abs((v1[0] * (v2[1] - v3[1]) + 
                          v2[0] * (v3[1] - v1[1]) + 
                          v3[0] * (v1[1] - v2[1])))
    
    def _distance(self, v1: List[float], v2: List[float]) -> float:
        """.

        Calculate the Euclidean distance between two vertices.
        
        Args:
            v1, v2: Vertices to calculate distance between
            
        Returns:
            Distance between vertices
        """
        return sum((a - b) ** 2 for a, b in zip(v1, v2)) ** 0.5
    
    def _point_in_triangle(
        self, 
        p: Tuple[float, float], 
        v1: Tuple[float, float], 
        v2: Tuple[float, float], 
        v3: Tuple[float, float]
    ) -> bool:
        """.

        Check if a point is inside a triangle using barycentric coordinates.
        
        Args:
            p: Point to check
            v1, v2, v3: Vertices of the triangle
            
        Returns:
            True if point is inside triangle, False otherwise
        """
        # Convert to barycentric coordinates
        def area(p1, p2, p3):
            return abs((p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1])) / 2.0)
        
        # Calculate areas
        A = area(v1, v2, v3)
        A1 = area(p, v2, v3)
        A2 = area(v1, p, v3)
        A3 = area(v1, v2, p)
        
        # Check if point is inside
        return abs(A - (A1 + A2 + A3)) < 1e-10
    
    def _interpolate_z(
        self, 
        p: Tuple[float, float], 
        v1: Tuple[float, float, float], 
        v2: Tuple[float, float, float], 
        v3: Tuple[float, float, float]
    ) -> float:
        """.

        Interpolate the z value at a point inside a triangle.
        
        Args:
            p: Point at which to interpolate (x, y)
            v1, v2, v3: Vertices of the triangle (x, y, z)
            
        Returns:
            Interpolated z value
        """
        # Extract coordinates
        px, py = p
        x1, y1, z1 = v1
        x2, y2, z2 = v2
        x3, y3, z3 = v3
        
        # Calculate barycentric coordinates
        denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
        if abs(denom) < 1e-10:
            return (z1 + z2 + z3) / 3.0  # If degenerate, return average
        
        a = ((y2 - y3) * (px - x3) + (x3 - x2) * (py - y3)) / denom
        b = ((y3 - y1) * (px - x3) + (x1 - x3) * (py - y3)) / denom
        c = 1.0 - a - b
        
        # Interpolate z
        return a * z1 + b * z2 + c * z3
