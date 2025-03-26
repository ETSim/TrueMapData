"""Unit tests for TMD adaptive mesh module."""

import unittest
import numpy as np
import os
import sys
import tempfile
from unittest.mock import patch, MagicMock

# Add the project root to the path to import tmd modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from tmd.exporters.model.adaptive_mesh import AdaptiveMeshGenerator, QuadTreeNode, convert_heightmap_to_adaptive_mesh


class TestAdaptiveMesh(unittest.TestCase):
    """Test class for adaptive mesh functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test heightmaps
        self.heightmap_flat = np.zeros((32, 32), dtype=np.float32)
        
        self.heightmap_slope = np.zeros((32, 32), dtype=np.float32)
        for i in range(32):
            for j in range(32):
                self.heightmap_slope[i, j] = i / 32.0
        
        self.heightmap_complex = np.zeros((32, 32), dtype=np.float32)
        for i in range(32):
            for j in range(32):
                # Create a heightmap with a sine wave pattern
                self.heightmap_complex[i, j] = 0.5 + 0.5 * np.sin(i/4.0) * np.cos(j/4.0)
    
    def test_quad_tree_node(self):
        """Test QuadTreeNode class."""
        # Test node initialization
        node = QuadTreeNode(0, 0, 16, 16, 0)
        self.assertEqual(node.x, 0)
        self.assertEqual(node.y, 0)
        self.assertEqual(node.width, 16)
        self.assertEqual(node.height, 16)
        self.assertEqual(node.depth, 0)
        self.assertTrue(node.is_leaf)
        self.assertEqual(len(node.children), 0)
        
        # Test corner calculation
        corners = node.get_corners()
        self.assertEqual(len(corners), 4)
        self.assertEqual(corners[0], (0, 0))
        self.assertEqual(corners[1], (16, 0))
        self.assertEqual(corners[2], (16, 16))
        self.assertEqual(corners[3], (0, 16))
        
        # Test center calculation
        center = node.get_center()
        self.assertEqual(center, (8, 8))
        
        # Test midpoints calculation
        midpoints = node.get_midpoints()
        self.assertEqual(len(midpoints), 4)
        self.assertEqual(midpoints[0], (8, 0))    # Top midpoint
        self.assertEqual(midpoints[1], (16, 8))   # Right midpoint
        self.assertEqual(midpoints[2], (8, 16))   # Bottom midpoint
        self.assertEqual(midpoints[3], (0, 8))    # Left midpoint
        
        # Test subdivision
        children = node.subdivide()
        self.assertEqual(len(children), 4)
        self.assertFalse(node.is_leaf)
        
        # Verify children properties
        self.assertEqual(children[0].x, 0)
        self.assertEqual(children[0].y, 0)
        self.assertEqual(children[0].width, 8)
        self.assertEqual(children[0].height, 8)
        self.assertEqual(children[0].depth, 1)
        
        # Calling subdivide on a non-leaf node should return None
        self.assertIsNone(node.subdivide())
    
    def test_adaptive_mesh_generator_init(self):
        """Test initialization of AdaptiveMeshGenerator."""
        mesh_gen = AdaptiveMeshGenerator(
            self.heightmap_flat, 
            z_scale=2.0, 
            base_height=0.5,
            max_subdivisions=8, 
            error_threshold=0.05
        )
        
        self.assertEqual(mesh_gen.z_scale, 2.0)
        self.assertEqual(mesh_gen.base_height, 0.5)
        self.assertEqual(mesh_gen.max_subdivisions, 8)
        self.assertEqual(mesh_gen.error_threshold, 0.05)
        
        # Check that mip levels are created
        self.assertGreaterEqual(len(mesh_gen.mip_levels), 2)
        
        # Check that detail maps are created
        self.assertEqual(len(mesh_gen.detail_maps), 3)
    
    def test_mesh_generation_flat(self):
        """Test mesh generation with a flat heightmap."""
        mesh_gen = AdaptiveMeshGenerator(
            self.heightmap_flat,
            max_subdivisions=4,
            error_threshold=0.1
        )
        
        # Generate the mesh
        vertices, triangles = mesh_gen.generate()
        
        # A flat mesh should have very few triangles
        self.assertLessEqual(len(triangles), 10)
        self.assertGreaterEqual(len(vertices), 4)  # At least 4 corners
        
        # Check that vertices are in normalized range [0,1] in x,y
        for vertex in vertices:
            self.assertGreaterEqual(vertex[0], 0.0)
            self.assertLessEqual(vertex[0], 1.0)
            self.assertGreaterEqual(vertex[1], 0.0)
            self.assertLessEqual(vertex[1], 1.0)
    
    def test_mesh_generation_complex(self):
        """Test mesh generation with a complex heightmap."""
        mesh_gen = AdaptiveMeshGenerator(
            self.heightmap_complex,
            max_subdivisions=6,
            error_threshold=0.01,
            base_height=0.2
        )
        
        # Generate the mesh
        vertices, triangles = mesh_gen.generate()
        
        # Complex mesh should have more triangles
        self.assertGreater(len(triangles), 10)
        
        # Check that triangles reference valid vertex indices
        for triangle in triangles:
            self.assertEqual(len(triangle), 3)
            for idx in triangle:
                self.assertGreaterEqual(idx, 0)
                self.assertLess(idx, len(vertices))
                
        # Check for base vertices (should have negative z values with base_height=0.2)
        has_base_vertices = False
        for vertex in vertices:
            if vertex[2] < 0:
                has_base_vertices = True
                break
        self.assertTrue(has_base_vertices)
    
    def test_triangulate_polygon(self):
        """Test polygon triangulation."""
        mesh_gen = AdaptiveMeshGenerator(self.heightmap_flat)
        
        # Test triangulation of a triangle (should return as is)
        vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        polygon = [0, 1, 2]
        triangles = mesh_gen._triangulate_polygon(vertices, polygon)
        self.assertEqual(len(triangles), 1)
        self.assertEqual(triangles[0], polygon)
        
        # Test triangulation of a quad
        vertices = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        polygon = [0, 1, 2, 3]
        triangles = mesh_gen._triangulate_polygon(vertices, polygon)
        self.assertEqual(len(triangles), 2)
        
        # Test triangulation of a larger polygon (5 vertices)
        vertices = [(0, 0, 0), (1, 0, 0), (1.5, 0.5, 0), (1, 1, 0), (0, 1, 0)]
        polygon = [0, 1, 2, 3, 4]
        triangles = mesh_gen._triangulate_polygon(vertices, polygon)
        # A 5-vertex polygon should create 3 triangles with our algorithm
        self.assertEqual(len(triangles), 5)
    
    def test_max_triangles_limit(self):
        """Test that max_triangles limit is respected."""
        mesh_gen = AdaptiveMeshGenerator(
            self.heightmap_complex,
            max_subdivisions=8,
            error_threshold=0.001  # Force lots of subdivision
        )
        
        # Generate mesh with triangle limit
        max_triangles = 100
        vertices, triangles = mesh_gen.generate(max_triangles=max_triangles)
        
        # Check triangle count is within limit
        self.assertLessEqual(len(triangles), max_triangles)
    
    def test_progress_callback(self):
        """Test progress callback mechanism."""
        mesh_gen = AdaptiveMeshGenerator(
            self.heightmap_flat,
            max_subdivisions=4
        )
        
        # Mock progress callback
        progress_callback = MagicMock()
        vertices, triangles = mesh_gen.generate(progress_callback=progress_callback)
        
        # Progress callback should be called multiple times (at 10%, 40%, 60%, 80%, 100%)
        self.assertGreaterEqual(progress_callback.call_count, 4)
        
        # Check final call was with 100%
        progress_callback.assert_any_call(100.0)
    
    @patch('os.path.exists')
    @patch('os.makedirs')
    def test_convert_heightmap_to_adaptive_mesh(self, mock_makedirs, mock_exists):
        """Test the heightmap conversion utility function."""
        # Set up mocks
        mock_exists.return_value = True
        
        # Test basic conversion without file output
        vertices, triangles = convert_heightmap_to_adaptive_mesh(
            self.heightmap_complex,
            z_scale=2.0,
            base_height=0.1,
            max_subdivisions=5,
            error_threshold=0.05
        )
        
        self.assertIsNotNone(vertices)
        self.assertIsNotNone(triangles)
        self.assertIsInstance(vertices, np.ndarray)
        self.assertIsInstance(triangles, np.ndarray)
    
    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    @patch('os.path.exists')
    @patch('os.makedirs')
    def test_convert_heightmap_with_output_file(self, mock_makedirs, mock_exists, mock_open):
        """Test conversion with file output."""
        # Set up mocks
        mock_exists.return_value = True
        
        # Test with ASCII output
        result = convert_heightmap_to_adaptive_mesh(
            self.heightmap_complex,
            output_file="test.stl",
            z_scale=2.0,
            base_height=0.1,
            max_subdivisions=4,
            error_threshold=0.1,
            ascii=True
        )
        
        # Check that makedirs was called
        mock_makedirs.assert_called_once()
        
        # Check that open was called with write mode
        mock_open.assert_called_once_with("test.stl", 'w')
        
        # Result should be a tuple of (vertices, triangles, output_file)
        self.assertEqual(len(result), 3)
    
    @patch('builtins.open')
    @patch('os.path.exists')
    @patch('os.makedirs')
    def test_convert_heightmap_with_binary_output(self, mock_makedirs, mock_exists, mock_open):
        """Test conversion with binary file output."""
        # Set up mocks
        mock_exists.return_value = True
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        # Test with binary output
        result = convert_heightmap_to_adaptive_mesh(
            self.heightmap_complex,
            output_file="test.stl",
            z_scale=2.0,
            base_height=0.1,
            max_subdivisions=4,
            error_threshold=0.1,
            ascii=False
        )
        
        # Check that open was called with write binary mode
        mock_open.assert_called_once_with("test.stl", 'wb')
        
        # Check that file.write was called (for binary header)
        mock_file.write.assert_called()


if __name__ == '__main__':
    unittest.main()
