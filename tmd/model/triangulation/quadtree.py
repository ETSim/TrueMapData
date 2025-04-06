"""
QuadTree-based triangulation for heightmaps.

This module provides a quadtree-based adaptive triangulation algorithm for
heightmaps, which recursively subdivides regions based on terrain complexity.
"""

import numpy as np
import logging
import time
from typing import List, Tuple, Dict, Any, Optional, Set, Union, Callable

from .base import BaseTriangulator
from ..utils.heightmap import calculate_terrain_complexity, sample_heightmap

# Set up logging
logger = logging.getLogger(__name__)


class QuadTreeNode:
    """Represents a node in a quadtree used for adaptive mesh generation."""
    
    def __init__(self, x: int, y: int, width: int, height: int, depth: int):
        """
        Initialize a quadtree node.
        
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
    
    def get_corners(self) -> List[Tuple[int, int]]:
        """
        Get the coordinates of the node's corners.
        
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
    
    def get_center(self) -> Tuple[int, int]:
        """
        Get the coordinates of the node's center.
        
        Returns:
            A tuple (x, y) representing the center.
        """
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    def get_midpoints(self) -> List[Tuple[int, int]]:
        """
        Get the coordinates of the midpoints of the node's edges.
        
        Returns:
            A list of (x, y) tuples representing the midpoints.
        """
        return [
            (self.x + self.width // 2, self.y),                   # Top midpoint
            (self.x + self.width, self.y + self.height // 2),     # Right midpoint
            (self.x + self.width // 2, self.y + self.height),     # Bottom midpoint
            (self.x, self.y + self.height // 2)                   # Left midpoint
        ]
    
    def subdivide(self) -> List['QuadTreeNode']:
        """
        Subdivide this node into four children.
        
        Returns:
            List of child nodes if subdivision was successful, None otherwise
        """
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


class QuadTreeTriangulator(BaseTriangulator):
    """
    Triangulate a heightmap using quadtree-based adaptive subdivision.
    
    This triangulator recursively subdivides the heightmap using a quadtree structure,
    with subdivision decisions based on error metrics and terrain complexity.
    """
    
    def __init__(
        self, 
        height_map: np.ndarray, 
        z_scale: float = 1.0, 
        max_triangles: int = 100000, 
        error_threshold: float = 0.05,  # Increased default threshold
        max_subdivisions: int = 4,  # Reduced default max subdivisions
        detail_boost: float = 0.5,  # Reduced detail boost for faster processing
        preserve_boundaries: bool = True,
        progress_callback: Optional[Callable[[float], None]] = None
    ):
        """
        Initialize the quadtree triangulator.
        
        Args:
            height_map: 2D array of height values
            z_scale: Scaling factor for height values
            max_triangles: Maximum number of triangles to generate
            error_threshold: Error threshold for subdivision
            max_subdivisions: Maximum number of recursive subdivisions
            detail_boost: Factor to boost detail in high-complexity areas
            preserve_boundaries: Whether to preserve heightmap boundaries in the mesh
            progress_callback: Optional callback function for progress reporting
        """
        super().__init__(
            height_map=height_map,
            z_scale=z_scale,
            max_triangles=max_triangles,
            error_threshold=error_threshold,
            progress_callback=progress_callback
        )
        
        self.max_subdivisions = max_subdivisions
        self.detail_boost = detail_boost
        self.preserve_boundaries = preserve_boundaries
        
        # Calculate detail map for feature detection
        self.complexity_map = self.calculate_complexity_map()
        
        # Compute target size based on max_subdivisions
        self.target_size = 2 ** max_subdivisions
        
        # Create resized heightmap for consistent sampling
        from ..utils.heightmap import resample_heightmap
        target_shape = (self.target_size + 1, self.target_size + 1)
        self.resized_heightmap = resample_heightmap(height_map, target_shape, 'bilinear')
        
        # Result containers
        self.vertices = []
        self.triangles = []
        self.vertex_map = {}  # Maps (x, y) coordinates to vertex indices
        
        logger.debug(f"QuadTreeTriangulator initialized with shape {height_map.shape}")
        logger.info(f"Initializing QuadTreeTriangulator with {height_map.shape} heightmap")
        logger.info(f"Parameters: max_subdivisions={max_subdivisions}, error_threshold={error_threshold}")
    
    def calculate_complexity_map(self) -> np.ndarray:
        """
        Calculate terrain complexity map to guide triangulation.
        
        Returns:
            2D array representing local terrain complexity
        """
        return calculate_terrain_complexity(self.height_map, smoothing=1.0)
    
    def triangulate(self) -> Tuple[List[List[float]], List[List[int]]]:
        """
        Run the quadtree triangulation algorithm.
        
        Returns:
            Tuple of (vertices, triangles) where vertices is a list of [x, y, z] coordinates
            and triangles is a list of [a, b, c] indices.
        """
        logger.info("Starting quadtree triangulation")
        
        # Report initial progress
        if self.progress_callback:
            self.progress_callback(0.05)
        
        # Create the quadtree
        quadtree = self._build_quadtree()
        logger.debug("Quadtree built successfully")
        
        # Report progress
        if self.progress_callback:
            self.progress_callback(0.4)
        
        # Extract mesh from quadtree
        logger.info("Extracting mesh from quadtree")
        self._extract_mesh_from_quadtree(quadtree)
        
        # Report progress
        if self.progress_callback:
            self.progress_callback(0.8)
        
        # Finalize the mesh
        vertices, triangles = self._finalize_mesh()
        logger.info(f"Mesh extracted: {len(vertices)} vertices, {len(triangles)} triangles")
        logger.info(f"Processing time: {time.time() - self.start_time:.2f}s")
        
        # Update statistics
        self.finalize_stats(vertices, triangles)
        
        # Final progress
        if self.progress_callback:
            self.progress_callback(1.0)
        
        return vertices, triangles
    
    def _build_quadtree(self) -> QuadTreeNode:
        """
        Build a quadtree based on heightmap complexity.
        
        Returns:
            Root node of the quadtree
        """
        logger.info("Starting quadtree construction")
        # Create root node covering the entire heightmap
        root = QuadTreeNode(0, 0, self.target_size, self.target_size, 0)
        
        # Queue for breadth-first traversal
        queue = [root]
        leaf_nodes = []
        
        # Track triangle count (for max_triangles limit)
        triangle_count = 0
        
        # Process nodes level by level
        for depth in range(self.max_subdivisions + 1):
            logger.debug(f"Processing depth {depth}")
            # Filter nodes at current depth
            current_nodes = [n for n in queue if n.depth == depth]
            next_nodes = [n for n in queue if n.depth > depth]
            logger.debug(f"Found {len(current_nodes)} nodes at depth {depth}")
            
            # Adjust error threshold based on depth
            level_threshold = self.error_threshold * (0.8 ** depth)
            
            # Process nodes at current depth
            for node in current_nodes:
                # Calculate error for this node
                error = self._calculate_node_error(node)
                node.error = error
                
                # Get feature importance
                complexity = self._get_node_complexity(node)
                
                # Determine if node should be subdivided
                should_subdivide = (
                    # Node is not at maximum depth
                    node.depth < self.max_subdivisions and
                    # Node is not too small
                    node.width > 1 and node.height > 1 and
                    # Either error is too high or node contains important features
                    (error > level_threshold * (1.0 + complexity * self.detail_boost) or
                    complexity > 0.7)  # High complexity always subdivides
                )
                
                # Check if subdividing would exceed max_triangles
                if should_subdivide:
                    # Each subdivision potentially adds 2 triangles per new node
                    new_triangles = 8  # 4 nodes * 2 triangles
                    if triangle_count + new_triangles > self.max_triangles:
                        should_subdivide = False
                        logger.info(f"Stopping subdivision at depth {depth} to meet triangle limit")
                
                if should_subdivide:
                    # Subdivide node and add children to queue
                    children = node.subdivide()
                    next_nodes.extend(children)
                    
                    # Update triangle count
                    triangle_count += new_triangles
                else:
                    # Keep as leaf node
                    leaf_nodes.append(node)
                    
                    # Each leaf node represents 2 triangles
                    triangle_count += 2
            
            # Update queue for next level
            queue = next_nodes
            
            # Report progress
            if self.progress_callback:
                progress = min(0.5, depth / (self.max_subdivisions + 1))
                self.progress_callback(progress)
            
            # Subdivision stats for this level
            subdivided_count = len([n for n in current_nodes if not n.is_leaf])
            logger.debug(f"Subdivided {subdivided_count} nodes at depth {depth}")
        
        # Add any remaining nodes as leaves
        leaf_nodes.extend(queue)
        
        logger.info(f"Built quadtree with {len(leaf_nodes)} leaf nodes, estimated {triangle_count} triangles")
        
        return root
    
    def _calculate_node_error(self, node: QuadTreeNode) -> float:
        """
        Calculate error metric for a node based on surface approximation error.
        
        Args:
            node: QuadTreeNode to calculate error for
            
        Returns:
            Error value indicating how well the node approximates the terrain
        """
        # Get the corner heights
        corners = node.get_corners()
        corner_heights = []
        for x, y in corners:
            # Ensure within bounds
            x = min(max(0, x), self.resized_heightmap.shape[1] - 1)
            y = min(max(0, y), self.resized_heightmap.shape[0] - 1)
            height = self.resized_heightmap[y, x]
            corner_heights.append(height)
        
        # Get center point
        center_x, center_y = node.get_center()
        # Ensure within bounds
        center_x = min(max(0, center_x), self.resized_heightmap.shape[1] - 1)
        center_y = min(max(0, center_y), self.resized_heightmap.shape[0] - 1)
        
        # Get actual center height
        actual_center_height = self.resized_heightmap[center_y, center_x]
        
        # Get edge midpoints
        midpoints = node.get_midpoints()
        midpoint_heights = []
        for x, y in midpoints:
            # Ensure within bounds
            x = min(max(0, x), self.resized_heightmap.shape[1] - 1)
            y = min(max(0, y), self.resized_heightmap.shape[0] - 1)
            height = self.resized_heightmap[y, x]
            midpoint_heights.append(height)
        
        # Linear interpolation at center based on corners
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
    
    def _get_node_complexity(self, node: QuadTreeNode) -> float:
        """
        Calculate average complexity value for a node.
        
        Args:
            node: QuadTreeNode to calculate complexity for
            
        Returns:
            Average complexity value for the node
        """
        # Convert to original heightmap coordinates (approximate)
        orig_height, orig_width = self.height_map.shape
        target_size = self.target_size
        
        # Get ratio between original and target size
        x_ratio = orig_width / target_size
        y_ratio = orig_height / target_size
        
        # Convert node bounds to original heightmap coordinates
        x1 = int(node.x * x_ratio)
        y1 = int(node.y * y_ratio)
        x2 = int(min((node.x + node.width) * x_ratio, orig_width - 1))
        y2 = int(min((node.y + node.height) * y_ratio, orig_height - 1))
        
        # Ensure valid bounds
        x1 = max(0, min(x1, orig_width - 1))
        y1 = max(0, min(y1, orig_height - 1))
        x2 = max(x1, min(x2, orig_width - 1))
        y2 = max(y1, min(y2, orig_height - 1))
        
        # If region is a single point, return complexity at that point
        if x1 == x2 or y1 == y2:
            return self.complexity_map[y1, x1]
        
        # Get region of complexity map
        region = self.complexity_map[y1:y2+1, x1:x2+1]
        
        # Return average complexity
        return np.mean(region)
    
    def _extract_mesh_from_quadtree(self, root: QuadTreeNode) -> None:
        """
        Extract mesh vertices and triangles from the quadtree.
        
        Args:
            root: Root node of the quadtree
        """
        # Reset result containers
        self.vertices = []
        self.triangles = []
        self.vertex_map = {}
        
        # Collect all leaf nodes
        leaf_nodes = self._collect_leaf_nodes(root)
        
        # Create vertices at all the corners of leaf nodes
        for node in leaf_nodes:
            corners = node.get_corners()
            for x, y in corners:
                if (x, y) not in self.vertex_map:
                    # Ensure coordinates are within bounds
                    x_safe = min(max(0, x), self.resized_heightmap.shape[1] - 1)
                    y_safe = min(max(0, y), self.resized_heightmap.shape[0] - 1)
                    
                    # Get height value
                    z = self.resized_heightmap[y_safe, x_safe] * self.z_scale
                    
                    # Add vertex
                    self.vertex_map[(x, y)] = len(self.vertices)
                    self.vertices.append([float(x), float(y), float(z)])
        
        # Create triangles from leaf nodes
        for node in leaf_nodes:
            # Get corners
            corners = node.get_corners()
            
            # Each leaf node forms two triangles
            # Triangle 1: Top-left, Top-right, Bottom-left
            self.triangles.append([
                self.vertex_map[corners[0]],  # Top-left
                self.vertex_map[corners[1]],  # Top-right
                self.vertex_map[corners[3]]   # Bottom-left
            ])
            
            # Triangle 2: Top-right, Bottom-right, Bottom-left
            self.triangles.append([
                self.vertex_map[corners[1]],  # Top-right
                self.vertex_map[corners[2]],  # Bottom-right
                self.vertex_map[corners[3]]   # Bottom-left
            ])
        
        logger.info(f"Extracted mesh with {len(self.vertices)} vertices, {len(self.triangles)} triangles")
    
    def _collect_leaf_nodes(self, node: QuadTreeNode) -> List[QuadTreeNode]:
        """
        Recursively collect all leaf nodes in the quadtree.
        
        Args:
            node: QuadTreeNode to start from
            
        Returns:
            List of all leaf nodes
        """
        if node.is_leaf:
            return [node]
        
        leaves = []
        for child in node.children:
            leaves.extend(self._collect_leaf_nodes(child))
        
        return leaves
    
    def _finalize_mesh(self) -> Tuple[List[List[float]], List[List[int]]]:
        """Finalize the mesh by normalizing coordinates."""
        # Calculate scale factors based on original heightmap dimensions
        x_scale = self.height_map.shape[1] / self.target_size
        y_scale = self.height_map.shape[0] / self.target_size
        
        # Scale vertices to match original heightmap dimensions
        normalized_vertices = []
        for vertex in self.vertices:
            x, y, z = vertex
            
            # Scale coordinates
            scaled_x = x * x_scale
            scaled_y = y * y_scale
            
            # Ensure we keep the proper height value
            x_idx = min(max(0, int(x)), self.resized_heightmap.shape[1] - 1)
            y_idx = min(max(0, int(y)), self.resized_heightmap.shape[0] - 1)
            scaled_z = self.resized_heightmap[y_idx, x_idx] * self.z_scale
            
            normalized_vertices.append([scaled_x, scaled_y, scaled_z])
        
        # Convert to lists for return
        vertices_list = [[float(v[0]), float(v[1]), float(v[2])] for v in normalized_vertices]
        triangles_list = [[int(t[0]), int(t[1]), int(t[2])] for t in self.triangles]
        
        return vertices_list, triangles_list


def triangulate_heightmap_quadtree(
    height_map: np.ndarray,
    max_triangles: int = 100000,
    error_threshold: float = 0.001,
    z_scale: float = 1.0,
    max_subdivisions: int = 8,
    detail_boost: float = 1.0,
    progress_callback: Optional[Callable[[float], None]] = None
) -> Tuple[List[List[float]], List[List[int]], Dict[str, Any]]:
    """
    Triangulate a heightmap using quadtree-based adaptive subdivision.
    
    Args:
        height_map: 2D numpy array of height values
        max_triangles: Maximum number of triangles to generate
        error_threshold: Maximum allowed error for approximation
        z_scale: Z-scaling factor for height values
        max_subdivisions: Maximum depth of the quadtree
        detail_boost: Factor to boost detail in high-complexity areas
        progress_callback: Optional callback function for progress reporting
        
    Returns:
        Tuple of (vertices, faces, statistics)
    """
    triangulator = QuadTreeTriangulator(
        height_map=height_map,
        max_triangles=max_triangles,
        error_threshold=error_threshold,
        z_scale=z_scale,
        max_subdivisions=max_subdivisions,
        detail_boost=detail_boost,
        progress_callback=progress_callback
    )
    
    vertices, triangles = triangulator.triangulate()
    stats = triangulator.get_statistics()
    
    return vertices, triangles, stats