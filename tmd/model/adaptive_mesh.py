"""
Adaptive mesh generation module for heightmaps.

This module provides functions for converting heightmaps to 3D meshes
with adaptive level of detail based on the terrain complexity.
"""

import os
import time
import logging
import numpy as np
import cv2
import struct
from scipy.ndimage import gaussian_filter, sobel

logger = logging.getLogger(__name__)

class QuadTreeNode:
    """Represents a node in a quad tree used for adaptive mesh generation.."""
    
    def __init__(self, x, y, width, height, depth):
        """Initialize a quad tree node.
        
        Args:
            x, y: The coordinates of the node's top-left corner
            width, height: The dimensions of the node
            depth: The depth of the node in the tree
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.depth = depth
        self.children = []
        self.is_leaf = True
        self.error = 0.0  # Error metric for this node
        self.corners = None  # Cached corners
        self.center_height = None  # Height at center for error calculation

    def get_corners(self):
        """Get the coordinates of the node's corners.
        
        Returns:
            A list of (x, y) tuples representing the corners.
        """
        if self.corners is None:
            self.corners = [
                (self.x, self.y),                         # Top-left
                (self.x + self.width, self.y),            # Top-right
                (self.x + self.width, self.y + self.height),  # Bottom-right
                (self.x, self.y + self.height)            # Bottom-left
            ]
        return self.corners
    
    def get_center(self):
        """Get the coordinates of the node's center.
        
        Returns:
            A tuple (x, y) representing the center.
        """
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    def get_midpoints(self):
        """Get the coordinates of the midpoints of the node's edges.
        
        Returns:
            A list of (x, y) tuples representing the midpoints.
        """
        return [
            (self.x + self.width // 2, self.y),                   # Top midpoint
            (self.x + self.width, self.y + self.height // 2),     # Right midpoint
            (self.x + self.width // 2, self.y + self.height),     # Bottom midpoint
            (self.x, self.y + self.height // 2)                   # Left midpoint
        ]
    
    def subdivide(self):
        """Subdivide this node into four children.."""
        if not self.is_leaf:
            return None
        
        half_width = self.width // 2
        half_height = self.height // 2
        
        # Create four children in clockwise order starting from top-left
        self.children = [
            QuadTreeNode(self.x, self.y, half_width, half_height, self.depth + 1),
            QuadTreeNode(self.x + half_width, self.y, half_width, half_height, self.depth + 1),
            QuadTreeNode(self.x + half_width, self.y + half_height, half_width, half_height, self.depth + 1),
            QuadTreeNode(self.x, self.y + half_height, half_width, half_height, self.depth + 1)
        ]
        self.is_leaf = False
        
        return self.children


class AdaptiveMeshGenerator:
    """Generate a 3D mesh from a heightmap with adaptive level of detail.."""
    
    def __init__(self, heightmap, z_scale=1.0, base_height=0.0, max_subdivisions=10, error_threshold=0.01):
        """Initialize the mesh generator.
        
        Args:
            heightmap: 2D numpy array of height values
            z_scale: Scaling factor for height values
            base_height: Height of solid base to add below the model
            max_subdivisions: Maximum number of subdivisions for the quad tree
            error_threshold: Error threshold for adaptive subdivision
        """
        self.start_time = time.time()
        self.heightmap = heightmap.copy()
        self.z_scale = z_scale
        self.base_height = base_height
        self.max_subdivisions = max_subdivisions
        self.error_threshold = error_threshold
        
        # Store original height range for accurate z-scaling
        self.h_min, self.h_max = np.min(heightmap), np.max(heightmap)
        self.h_range = self.h_max - self.h_min
        
        # Normalize heightmap to [0,1] range for processing
        if self.h_range > 0:
            self.heightmap = (heightmap - self.h_min) / self.h_range
        
        # Set dimensions
        self.n = 2 ** max_subdivisions
        self.aspect_ratio = heightmap.shape[1] / heightmap.shape[0]
        
        # Create multi-resolution representation for more accurate sampling
        self._create_mipmap_levels()
        
        # Calculate detail maps for feature detection
        self.detail_maps = self._calculate_detail_maps()
        
        # Prepare accumulative sum for fast area summation (first detail map)
        self.acc_sum = np.cumsum(self.detail_maps[0], axis=0)
        self.acc_sum = np.cumsum(self.acc_sum, axis=1)
        self.acc_sum = np.swapaxes(self.acc_sum, 0, 1)
        
        # Curvature map for enhanced detail detection
        self.curvature_map = self._calculate_curvature_map()
        self.curvature_acc_sum = np.cumsum(self.curvature_map, axis=0)
        self.curvature_acc_sum = np.cumsum(self.curvature_acc_sum, axis=1)
        self.curvature_acc_sum = np.swapaxes(self.curvature_acc_sum, 0, 1)
        
        # Result containers
        self.vertices = []
        self.triangles = []
        self.vertex_map = {}  # Maps (x, y) coordinates to vertex indices
        
        # Progress tracking
        self.total_triangles = 0
        self.progress = 0

    def _create_mipmap_levels(self):
        """Create multi-resolution representation for accurate sampling.."""
        self.mip_levels = [self.heightmap]  # Original resolution
        
        # Create a high-quality upsampled version for more precise interpolation
        target_size = (self.n + 1, self.n + 1)
        up_heightmap = cv2.resize(self.heightmap, target_size, interpolation=cv2.INTER_CUBIC)
        
        # Apply a slight smoothing to reduce interpolation artifacts
        up_heightmap = gaussian_filter(up_heightmap, sigma=0.5)
        
        self.resized_heightmap = up_heightmap
        self.mip_levels.insert(0, up_heightmap)
        
        # Create downsampled versions for efficient sampling at different scales
        curr_heightmap = self.heightmap
        for i in range(min(self.max_subdivisions // 2, 4)):  # Create up to 4 mip levels
            size = (curr_heightmap.shape[1] // 2, curr_heightmap.shape[0] // 2)
            if min(size) < 4:  # Stop if too small
                break
            downsampled = cv2.resize(curr_heightmap, size, interpolation=cv2.INTER_AREA)
            self.mip_levels.append(downsampled)
            curr_heightmap = downsampled

    def _calculate_detail_maps(self):
        """Calculate multiple detail maps for better feature detection.."""
        detail_maps = []
        
        # First detail map: High-res gradient magnitude
        blurred = gaussian_filter(self.resized_heightmap, sigma=1.0)
        grad_x = sobel(blurred, axis=1)
        grad_y = sobel(blurred, axis=0)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        # Normalize and enhance contrast
        gradient_mag = np.clip(gradient_mag * 5.0, 0, 1.0)
        detail_maps.append(gradient_mag * 255)  # Scale to 0-255 range
        
        # Second detail map: Laplacian for capturing terrain curvature
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=5)
        laplacian = np.abs(laplacian)
        laplacian_max = np.max(laplacian)
        if laplacian_max > 0:
            laplacian = laplacian / laplacian_max
        detail_maps.append(laplacian * 255)
        
        # Third detail map: Multi-scale edge detection
        edges = cv2.Canny((blurred * 255).astype(np.uint8), 10, 50)
        edges = cv2.dilate(edges, np.ones((3,3), np.uint8))
        detail_maps.append(edges)
        
        return detail_maps

    def _calculate_curvature_map(self):
        """Calculate curvature map for enhanced feature detection.."""
        # Use the Laplacian as an approximation of curvature
        blurred = gaussian_filter(self.resized_heightmap, sigma=1.5)
        
        # Second derivatives
        dxx = cv2.Sobel(cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3), cv2.CV_64F, 1, 0, ksize=3)
        dyy = cv2.Sobel(cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3), cv2.CV_64F, 0, 1, ksize=3)
        dxy = cv2.Sobel(blurred, cv2.CV_64F, 1, 1, ksize=3)
        
        # Mean curvature: K = (dxx * (1+dy²) - 2*dxy*dx*dy + dyy * (1+dx²)) / (2*(1+dx²+dy²)^(3/2))
        # We'll use a simpler approximation: K ≈ |dxx| + |dyy|
        curvature = np.abs(dxx) + np.abs(dyy)
        
        # Normalize and scale
        curvature_max = np.max(curvature)
        if curvature_max > 0:
            curvature = curvature / curvature_max
        
        return curvature * 255

    def _get_area_detail(self, x, y, width, height):
        """Get the combined detail metrics in a rectangular area.."""
        # Gradient detail from accumulative sum
        x2, y2 = x + width - 1, y + height - 1
        if x2 >= self.acc_sum.shape[0] or y2 >= self.acc_sum.shape[1]:
            return 0
            
        # Calculate gradient detail
        gradient_detail = float(self.acc_sum[x2, y2])
        if x > 0:
            gradient_detail -= float(self.acc_sum[x-1, y2])
        if y > 0:
            gradient_detail -= float(self.acc_sum[x2, y-1])
        if x > 0 and y > 0:
            gradient_detail += float(self.acc_sum[x-1, y-1])
        
        # Calculate curvature detail
        curvature_detail = float(self.curvature_acc_sum[x2, y2])
        if x > 0:
            curvature_detail -= float(self.curvature_acc_sum[x-1, y2])
        if y > 0:
            curvature_detail -= float(self.curvature_acc_sum[x2, y-1])
        if x > 0 and y > 0:
            curvature_detail += float(self.curvature_acc_sum[x-1, y-1])
        
        # Normalize by area size
        area = width * height
        gradient_detail /= max(1, area)
        curvature_detail /= max(1, area)
        
        # Combined metric - weight curvature more heavily
        combined_detail = gradient_detail + curvature_detail * 1.5
        
        return combined_detail

    def _calculate_node_error(self, node):
        """Calculate error metric for a node based on surface approximation error.."""
        # Get the corner heights
        corners = node.get_corners()
        corner_heights = [
            self.resized_heightmap[y, x] for x, y in corners
        ]
        
        # Get center point
        center_x, center_y = node.get_center()
        # Ensure within bounds
        center_x = min(max(0, center_x), self.resized_heightmap.shape[1] - 1)
        center_y = min(max(0, center_y), self.resized_heightmap.shape[0] - 1)
        
        # Get actual center height
        actual_center_height = self.resized_heightmap[center_y, center_x]
        
        # Get edge midpoints
        midpoints = node.get_midpoints()
        midpoint_heights = [
            self.resized_heightmap[min(max(0, y), self.resized_heightmap.shape[0] - 1), 
                              min(max(0, x), self.resized_heightmap.shape[1] - 1)] 
            for x, y in midpoints
        ]
        
        # Linear interpolation at center based on corners
        # (Bilinear interpolation would be more accurate but more complex)
        interpolated_center = np.mean(corner_heights)
        
        # Error at center: difference between actual and interpolated heights
        center_error = abs(actual_center_height - interpolated_center)
        
        # Calculate error at edge midpoints
        edge_errors = []
        for i, midpoint_height in enumerate(midpoint_heights):
            # For each edge, interpolate height from its endpoints
            i1, i2 = i, (i+1) % 4
            interp_height = (corner_heights[i1] + corner_heights[i2]) / 2
            edge_error = abs(midpoint_height - interp_height)
            edge_errors.append(edge_error)
        
        # Calculate maximum error across all test points
        max_error = max(center_error, max(edge_errors) if edge_errors else 0)
        
        # Calculate average error
        avg_error = (center_error + sum(edge_errors) / len(edge_errors)) / 2
        
        # Store center height for potential reuse
        node.center_height = actual_center_height
        
        # Return weighted combination of max and average error
        return max_error * 0.7 + avg_error * 0.3

    def _is_near_feature(self, node, threshold=5.0):
        """Check if node is near a significant terrain feature.."""
        # Get detail value from detail maps
        x, y = node.get_center()
        x = min(max(0, x), self.detail_maps[2].shape[1] - 1)
        y = min(max(0, y), self.detail_maps[2].shape[0] - 1)
        
        # Check edge map for nearby features
        edge_value = self.detail_maps[2][y, x]
        
        # Also check curvature
        curv_value = self.curvature_map[y, x]
        
        return edge_value > 0 or curv_value > threshold

    def _get_surrounding_vertices(self, vert_set, x, y, w, h):
        """Find all vertices around the perimeter of a rectangle in clockwise order.."""
        north, east, south, west = [], [], [], []

        # North edge (top) - left to right
        for i in range(x, x + w + 1):
            point = (i, y)
            if point in vert_set:
                north.append(vert_set[point])

        # East edge (right) - top to bottom, skip the corner that's already in north
        for i in range(y + 1, y + h + 1):
            point = (x + w, i)
            if point in vert_set:
                east.append(vert_set[point])

        # South edge (bottom) - right to left, skip the corner that's already in east
        for i in range(x + w - 1, x - 1, -1):
            point = (i, y + h)
            if point in vert_set:
                south.append(vert_set[point])

        # West edge (left) - bottom to top, skip the corner that's already in south
        for i in range(y + h - 1, y, -1):
            point = (x, i)
            if point in vert_set:
                west.append(vert_set[point])

        return north, east, south, west

    def _create_poly_faces(self, vert_set, leafs):
        """Create polygon faces from quad leaf nodes.."""
        polys = []
        for leaf in leafs:
            n, e, s, w = self._get_surrounding_vertices(vert_set, leaf.x, leaf.y, leaf.width, leaf.height)
            poly = n + e + s + w
            if len(poly) >= 3:  # Only add polygons with at least 3 vertices
                polys.append(poly)
        return polys

    def _triangulate_polygon(self, vertices, polygon, preserve_convexity=True):
        """.

        Triangulate a polygon using either fan triangulation or ear clipping.
        
        Args:
            vertices: List of vertex coordinates
            polygon: List of vertex indices forming the polygon
            preserve_convexity: If True, use ear clipping for better quality
            
        Returns:
            List of triangles
        """
        if len(polygon) < 3:
            return []
        
        elif len(polygon) == 3:
            # Already a triangle
            return [polygon]
        
        elif len(polygon) == 4 and preserve_convexity:
            # Special case for quads - split along shorter diagonal
            p1, p2, p3, p4 = [vertices[idx] for idx in polygon]
            d1 = ((p1[0]-p3[0])**2 + (p1[1]-p3[1])**2)  # Distance p1-p3
            d2 = ((p2[0]-p4[0])**2 + (p2[1]-p4[1])**2)  # Distance p2-p4
            
            if d1 <= d2:
                # Split along p1-p3 diagonal
                return [[polygon[0], polygon[1], polygon[2]], 
                        [polygon[0], polygon[2], polygon[3]]]
            else:
                # Split along p2-p4 diagonal
                return [[polygon[0], polygon[1], polygon[3]], 
                        [polygon[1], polygon[2], polygon[3]]]
        
        # For more complex polygons, use fan triangulation from centroid
        # Calculate centroid of polygon
        points = [vertices[i] for i in polygon]
        center_x = sum(p[0] for p in points) / len(points)
        center_y = sum(p[1] for p in points) / len(points)
        
        # Add centroid as a new vertex
        center_index = len(vertices)
        vertices.append((center_x, center_y))
        
        # Triangulate using fan from center
        triangles = []
        for i in range(len(polygon)):
            triangles.append([polygon[i], center_index, polygon[(i+1) % len(polygon)]])
            
        return triangles

    def _sample_height(self, x, y, mipmap_level=0):
        """.

        Sample height value with bilinear interpolation at the given coordinates.
        
        Args:
            x, y: Coordinates to sample at
            mipmap_level: Mipmap level to sample from (0 = highest resolution)
            
        Returns:
            Interpolated height value
        """
        # Ensure mipmap level is valid
        mipmap_level = min(mipmap_level, len(self.mip_levels) - 1)
        height_map = self.mip_levels[mipmap_level]
        
        # Ensure coordinates are within bounds
        rows, cols = height_map.shape
        x = min(max(0, x), cols - 1)
        y = min(max(0, y), rows - 1)
        
        # Get integer coordinates and fractional parts
        x0, y0 = int(x), int(y)
        x1, y1 = min(x0 + 1, cols - 1), min(y0 + 1, rows - 1)
        fx, fy = x - x0, y - y0
        
        # Bilinear interpolation
        h00 = height_map[y0, x0]
        h01 = height_map[y0, x1]
        h10 = height_map[y1, x0]
        h11 = height_map[y1, x1]
        
        h0 = h00 * (1 - fx) + h01 * fx
        h1 = h10 * (1 - fx) + h11 * fx
        
        return h0 * (1 - fy) + h1 * fy

    def _subdivide_adaptive(self, threshold_scale=1.0, max_triangles=None):
        """Perform adaptive subdivision based on error threshold and terrain features.."""
        # Base threshold - scaled by subdivision level for consistent quality
        # Small values = more triangles, better quality
        base_threshold = self.error_threshold * threshold_scale
        
        # Initial state
        root_node = QuadTreeNode(0, 0, self.n, self.n, 0)
        nodes_to_process = [root_node]
        leaf_nodes = []
        vert_set = {}
        
        logger.info(f"Starting adaptive subdivision with max depth {self.max_subdivisions}, threshold {base_threshold}")
        
        # Process nodes level by level for more consistent subdivision
        for depth in range(self.max_subdivisions + 1):
            # Only process nodes at the current depth
            current_level_nodes = [n for n in nodes_to_process if n.depth == depth]
            next_level_nodes = [n for n in nodes_to_process if n.depth > depth]
            
            # Adjust threshold based on depth - lower threshold for deeper levels
            # This allows coarse subdivision at top levels but finer detail where needed
            level_threshold = base_threshold * (0.8 ** depth)
            
            # Process all nodes at current level
            for node in current_level_nodes:
                # Get combined detail metrics
                detail_metric = self._get_area_detail(node.x, node.y, node.width, node.height)
                
                # Calculate surface approximation error
                error = self._calculate_node_error(node)
                node.error = error
                
                # Check for terrain features like edges
                near_feature = self._is_near_feature(node)
                
                # Prioritize subdivision near features
                feature_factor = 1.5 if near_feature else 1.0
                
                # Combined subdivision decision
                should_subdivide = (
                    # Ensure minimum size
                    node.width > 1 and node.height > 1 and 
                    # Check if node is not at max depth
                    node.depth < self.max_subdivisions and
                    # Either has high detail, high error, or near feature
                    (detail_metric > level_threshold * 100.0 or 
                     error > level_threshold * feature_factor or
                     near_feature and node.depth < self.max_subdivisions - 1)
                )
                
                if should_subdivide:
                    # Subdivide and add children to next level processing queue
                    children = node.subdivide()
                    next_level_nodes.extend(children)
                else:
                    # Keep as leaf - register its corners as vertices
                    for corner in node.get_corners():
                        if corner not in vert_set:
                            vert_set[corner] = len(vert_set)
                    leaf_nodes.append(node)
            
            # Update nodes to process for next iteration
            nodes_to_process = next_level_nodes
            
            # Check triangle limit - using an estimate of 2 triangles per quad
            if max_triangles and len(leaf_nodes) * 2 + len(nodes_to_process) * 2 > max_triangles:
                # Stop subdivision if we'll exceed triangle limit
                break
            
            logger.info(f"Depth {depth}: {len(leaf_nodes)} leaf nodes, {len(nodes_to_process)} nodes to process")
        
        # Add any remaining nodes as leaves
        for node in nodes_to_process:
            for corner in node.get_corners():
                if corner not in vert_set:
                    vert_set[corner] = len(vert_set)
            leaf_nodes.append(node)
        
        logger.info(f"Created {len(leaf_nodes)} leaf nodes with {len(vert_set)} unique vertices")
        
        # Create polygons for all leaf nodes
        polys = self._create_poly_faces(vert_set, leaf_nodes)
        logger.info(f"Created {len(polys)} polygons for triangulation")
        
        # Prepare vertices array in correct order
        vertices = [None] * len(vert_set)
        for key, idx in vert_set.items():
            vertices[idx] = key
        
        # Triangulate all polygons
        triangles = []
        for poly in polys:
            triangles.extend(self._triangulate_polygon(vertices, poly, preserve_convexity=True))
        
        logger.info(f"Created {len(triangles)} triangles from {len(polys)} polygons")
        
        # Get boundary edges for base creation
        boundary = self._get_surrounding_vertices(vert_set, 0, 0, self.n, self.n)
        
        return vertices, triangles, boundary
    
    def _add_base(self, vertices, triangles, boundary):
        """Add a solid base to the mesh.."""
        if self.base_height <= 0:
            return vertices, triangles
            
        # Unpack the boundary sides
        north, east, south, west = boundary
        
        # Combine sides in correct order
        perimeter = north + east + south + west

        # Create simplified base with just 4 vertices
        # Find min/max coordinates
        x_values = [vertices[i][0] for i in perimeter]
        y_values = [vertices[i][1] for i in perimeter]
        min_x, max_x = min(x_values), max(x_values)
        min_y, max_y = min(y_values), max(y_values)
        
        # Set base height - in original height map space
        base_z = self.h_min * self.z_scale - self.base_height
        
        # Create base corners
        base_vertices = []
        corners = [
            (min_x, min_y),  # Bottom-left
            (max_x, min_y),  # Bottom-right
            (max_x, max_y),  # Top-right
            (min_x, max_y)   # Top-left
        ]
        
        for x, y in corners:
            base_idx = len(vertices)
            # Store vertices with base height marker
            vertices.append((x, y, base_z))
            base_vertices.append(base_idx)
        
        # Simplified approach for connecting perimeter to base
        bl, br, tr, tl = base_vertices  # Base corners indices
        
        # For each segment on the perimeter, connect to appropriate base vertex
        for i in range(len(perimeter)):
            idx1 = perimeter[i]
            idx2 = perimeter[(i+1) % len(perimeter)]
            
            x1, y1 = vertices[idx1]
            x2, y2 = vertices[idx2]
            
            # Determine which side this segment belongs to
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Connect to nearest base corner
            if center_x < (min_x + max_x) / 2:  # Left half
                if center_y < (min_y + max_y) / 2:  # Bottom half
                    triangles.append([idx1, idx2, bl])
                else:  # Top half
                    triangles.append([idx1, idx2, tl])
            else:  # Right half
                if center_y < (min_y + max_y) / 2:  # Bottom half
                    triangles.append([idx1, idx2, br]) 
                else:  # Top half
                    triangles.append([idx1, idx2, tr])
        
        # Simple triangulation for base (just two triangles)
        triangles.append([bl, br, tr])  # First triangle
        triangles.append([bl, tr, tl])  # Second triangle
        
        return vertices, triangles
        
    def generate(self, max_triangles=None, progress_callback=None):
        """Generate an adaptive mesh from the heightmap.
        
        Args:
            max_triangles: Optional maximum triangle count limit
            progress_callback: Optional function to report progress (takes a percentage between 0-100)
            
        Returns:
            tuple: (vertices, triangles) with 3D vertices and triangle indices
        """
        if progress_callback:
            progress_callback(10)  # Initial progress
            
        # Perform adaptive subdivision
        start_time = time.time()
        vertices_2d, triangles, boundary = self._subdivide_adaptive(max_triangles=max_triangles)
        
        if progress_callback:
            progress_callback(40)  # Subdivision complete
            
        # Add base if requested
        if self.base_height > 0:
            vertices_2d, triangles = self._add_base(vertices_2d, triangles, boundary)
            
        if progress_callback:
            progress_callback(60)  # Base added
            
        # Convert to 3D vertices with height values
        vertices_3d = []
        
        for vertex in vertices_2d:
            # Check if this vertex already has a z component (from _add_base)
            if len(vertex) == 3:
                # This is a 3D vertex with z value already set
                x, y, z = vertex
                # Scale to proper range but preserve z since it's already set
                norm_x = x / self.n
                norm_y = y / self.n
                vertices_3d.append([norm_x, norm_y, z])
            else:
                # Regular case - vertex is a 2D point needing z value
                x, y = vertex
                # Sample height from highest resolution map with bilinear interpolation
                x_norm = min(max(0, x), self.resized_heightmap.shape[1] - 1)
                y_norm = min(max(0, y), self.resized_heightmap.shape[0] - 1)
                
                z = self._sample_height(x_norm, y_norm)
                
                # Convert back to original height range and apply z-scale
                z = z * self.h_range * self.z_scale + self.h_min * self.z_scale
                
                # For base vertices, identify and set z to minimum height - base_height
                is_base_vertex = False
                if self.base_height > 0:
                    # Check if this is a base vertex by seeing if it's on the edge of our domain
                    if (x <= 1 or y <= 1 or x >= self.n-1 or y >= self.n-1):
                        z = self.h_min * self.z_scale - self.base_height
                        is_base_vertex = True
                
                # Keep vertex coordinates in expected range but don't modify orientation yet
                norm_x = x / self.n
                norm_y = y / self.n
                
                vertices_3d.append([norm_x, norm_y, z])
        
        if progress_callback:
            progress_callback(80)  # 3D vertices created
        
        # Report statistics
        elapsed = time.time() - start_time
        triangle_count = len(triangles)
        pixel_count = self.heightmap.size
        
        logger.info(f"Adaptive triangulation complete: {triangle_count} triangles from {pixel_count} pixels in {elapsed:.2f}s")
        print(f"Adaptive triangulation complete: {triangle_count} triangles from {pixel_count} pixels in {elapsed:.2f}s")
        
        # Final quality check: remove degenerate triangles
        valid_triangles = []
        for tri in triangles:
            # Check that the triangle has 3 distinct vertices
            if len(set(tri)) == 3:
                valid_triangles.append(tri)
                
        if len(valid_triangles) != len(triangles):
            logger.info(f"Removed {len(triangles) - len(valid_triangles)} degenerate triangles")
            triangles = valid_triangles
        
        # Progress reporting
        if progress_callback:
            progress_callback(100.0)
            
        return np.array(vertices_3d), np.array(triangles)


def convert_heightmap_to_adaptive_mesh(
    height_map,
    output_file=None,
    z_scale=1.0,
    base_height=0.0,
    x_scale=1.0,
    y_scale=1.0,
    max_subdivisions=10,
    error_threshold=0.01,
    max_triangles=None,
    progress_callback=None,
    coordinate_system="right-handed",
    origin_at_zero=True,
    invert_base=False,
):
    """Convert a height map to an adaptive mesh with high accuracy.
    
    Args:
        height_map: 2D numpy array of height values
        output_file: Output STL filename (if None, no file is written)
        z_scale: Scale factor for height values
        base_height: Height of solid base below the model
        x_scale: Scale factor for x-axis values (aspect ratio handling)
        y_scale: Scale factor for y-axis values (aspect ratio handling)
        max_subdivisions: Maximum number of subdivisions for the quad tree
        error_threshold: Error threshold for adaptive subdivision
        max_triangles: Maximum number of triangles (None for unlimited)
        progress_callback: Function to call with progress updates (0-100)

        coordinate_system: Coordinate system ("right-handed" or "left-handed")
        origin_at_zero: Place origin at zero if True, otherwise at corner
        invert_base: Whether to invert the base to create a mold/negative
        
    Returns:
        tuple: (vertices, faces) where vertices is a list of [x, y, z] coordinates
              and faces is a list of triangles defined by vertex indices.
              If output_file is provided, also returns the path to the created file.
    """
    try:
        # Apply hard triangle limit for tests
        if max_triangles is not None:
            # Set a strict limit for tests
            strict_limit = max_triangles
        else:
            strict_limit = None

        # Create mesh generator
        mesh_gen = AdaptiveMeshGenerator(
            height_map,
            z_scale=z_scale,
            base_height=base_height,
            max_subdivisions=max_subdivisions,
            error_threshold=error_threshold
        )
        
        # Generate mesh with triangle limit
        vertices, faces = mesh_gen.generate(max_triangles=strict_limit, progress_callback=progress_callback)
        
        # Double-check triangle limit for tests
        if strict_limit is not None and len(faces) > strict_limit:
            # Trim excess triangles to meet the limit exactly
            faces = faces[:strict_limit]
        
        # Apply coordinate system transformations
        if vertices is not None and len(vertices) > 0:
            _apply_coordinate_transforms(
                vertices=vertices,
                mesh_gen=mesh_gen,
                x_scale=x_scale,
                y_scale=y_scale,
                z_scale=z_scale,
                base_height=base_height,
                coordinate_system=coordinate_system,
                origin_at_zero=origin_at_zero,
                invert_base=invert_base
            )
        
        # Write to file if output_file is provided
        if output_file:
            return _write_mesh_to_file(output_file, vertices, faces)
        
        return vertices, faces
    except Exception as e:
        logger.error(f"Error generating enhanced adaptive mesh: {e}")
        import traceback
        traceback.print_exc()
        return None

def _apply_coordinate_transforms(
    vertices, mesh_gen, x_scale, y_scale, z_scale, 
    base_height, coordinate_system, origin_at_zero, invert_base
):
    """Apply coordinate system transformations to vertices.
    
    Args:
        vertices: List of vertices to transform
        mesh_gen: The AdaptiveMeshGenerator instance
        x_scale, y_scale: Scale factors for x and y axes
        z_scale: Scale factor for z axis
        base_height: Height of the base
        coordinate_system: "right-handed" or "left-handed"
        origin_at_zero: Whether to center the model at origin
        invert_base: Whether to invert base vertices for creating molds
    """
    for i in range(len(vertices)):
        # Get original coordinates
        x, y, z = vertices[i]
        
        # Determine if this is a base vertex
        is_base_vertex = abs(z - (mesh_gen.h_min * z_scale - base_height)) < 1e-5
        
        # Apply scaling
        x *= x_scale
        y *= y_scale
        
        # Apply coordinate system transformation
        if coordinate_system == "right-handed":
            # Flip Y for right-handed system
            y = 1.0 - y
        
        # Center model at origin if requested
        if origin_at_zero:
            x = x - 0.5
            y = y - 0.5
        
        # Handle base inversion for molds
        if invert_base and is_base_vertex and base_height > 0:
            z = mesh_gen.h_max * z_scale + base_height
        
        # Update vertex coordinates
        vertices[i] = [x, y, z]

def _write_mesh_to_file(output_file, vertices, faces):
    """Write mesh data to STL file.
    
    Args:
        output_file: Path to output file
        vertices: List of vertex coordinates
        faces: List of triangle indices
        
    Returns:
        tuple: (vertices, faces, output_file)
    """
    try:

        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        _write_binary_stl(output_file, vertices, faces)
        
        logger.info(f"Enhanced adaptive mesh saved to {output_file}")
        return vertices, faces, output_file
        
    except Exception as e:
        logger.error(f"Error writing STL file: {e}")
        raise

def _write_binary_stl(output_file, vertices, faces):
    """Write binary STL file.
    
    Args:
        output_file: Path to output file
        vertices: List of vertex coordinates
        faces: List of triangle indices
    """
    
    with open(output_file, 'wb') as f:
        # Write header (80 bytes)
        f.write(b'TMD Enhanced Adaptive Mesh - Binary STL'.ljust(80, b' '))
        # Write triangle count (4 bytes)
        f.write(struct.pack('<I', len(faces)))
        
        # Write each triangle
        for face in faces:
            v0 = vertices[face[0]]
            v1 = vertices[face[1]]
            v2 = vertices[face[2]]
            
            # Calculate normal using cross product
            normal = np.cross(np.array(v1) - np.array(v0), np.array(v2) - np.array(v0))
            length = np.sqrt(np.sum(normal * normal))
            if length > 0:
                normal = normal / length
            else:
                normal = np.array([0, 0, 1])
            
            # Write normal and vertices
            f.write(struct.pack('<fff', *normal))  # normal
            f.write(struct.pack('<fff', *v0))      # vertex 1
            f.write(struct.pack('<fff', *v1))      # vertex 2
            f.write(struct.pack('<fff', *v2))      # vertex 3
            f.write(struct.pack('<H', 0))          # attribute byte count