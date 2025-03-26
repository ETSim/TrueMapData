"""Unit tests for TMD adaptive triangulator module."""

import unittest
import numpy as np
import os
import sys
from unittest.mock import patch, MagicMock

# Add the project root to the path to import tmd modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from tmd.exporters.model.adaptive_triangulator import AdaptiveTriangulator


class TestAdaptiveTriangulator(unittest.TestCase):
    """Test class for adaptive triangulator functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple heightmap for testing
        self.height_map_flat = np.zeros((10, 10), dtype=np.float32)
        self.height_map_slope = np.zeros((10, 10), dtype=np.float32)
        for i in range(10):
            for j in range(10):
                self.height_map_slope[i, j] = i / 10.0
        
        self.height_map_peak = np.zeros((10, 10), dtype=np.float32)
        for i in range(10):
            for j in range(10):
                self.height_map_peak[i, j] = 1.0 - ((i-5)**2 + (j-5)**2) / 50.0
                if self.height_map_peak[i, j] < 0:
                    self.height_map_peak[i, j] = 0
        
    def test_initialization(self):
        """Test initialization of AdaptiveTriangulator."""
        # Test with default parameters
        triangulator = AdaptiveTriangulator(self.height_map_flat)
        self.assertEqual(triangulator.height_map.shape, (10, 10))
        self.assertEqual(triangulator.max_triangles, 100000)
        self.assertEqual(triangulator.error_threshold, 0.001)
        self.assertEqual(triangulator.z_scale, 1.0)
        self.assertEqual(triangulator.min_area, 0.0001 * 10 * 10)
        
        # Test with custom parameters
        triangulator = AdaptiveTriangulator(
            self.height_map_flat, 
            max_triangles=5000, 
            error_threshold=0.01,
            min_area_fraction=0.001,
            z_scale=2.0
        )
        self.assertEqual(triangulator.max_triangles, 5000)
        self.assertEqual(triangulator.error_threshold, 0.01)
        self.assertEqual(triangulator.z_scale, 2.0)
        self.assertEqual(triangulator.min_area, 0.001 * 10 * 10)
    
    def test_run_flat(self):
        """Test triangulation on a flat heightmap."""
        triangulator = AdaptiveTriangulator(self.height_map_flat)
        vertices, triangles = triangulator.run()
        
        # A flat heightmap should result in minimum triangulation
        # Expect only 2 triangles for a flat surface
        self.assertEqual(len(triangles), 2)
        self.assertEqual(len(vertices), 4)  # 4 corners
        
        # Check that all z values are 0 since it's a flat heightmap
        for vertex in vertices:
            self.assertEqual(vertex[2], 0.0)
    
    def test_add_vertex(self):
        """Test adding vertices to the triangulator."""
        triangulator = AdaptiveTriangulator(self.height_map_slope)
        
        # Add a vertex
        idx1 = triangulator._add_vertex(0, 0)
        self.assertEqual(idx1, 0)  # First vertex should have index 0
        
        # Adding the same vertex should return the same index
        idx2 = triangulator._add_vertex(0, 0)
        self.assertEqual(idx1, idx2)  # Should return same index
        
        # Add another vertex and check z-scaling
        triangulator.z_scale = 2.0
        idx3 = triangulator._add_vertex(5, 5)
        self.assertEqual(idx3, 1)  # Second vertex should be index 1
        self.assertEqual(triangulator.vertices[idx3][2], 1.0)  # 5/10 * 2.0 = 1.0
    
    def test_run_slope(self):
        """Test triangulation on a sloped heightmap."""
        triangulator = AdaptiveTriangulator(self.height_map_slope, error_threshold=0.001)
        vertices, triangles = triangulator.run()
        
        # A sloped heightmap will likely need more triangles than a flat one
        # for accurate representation, depending on the error threshold
        self.assertGreaterEqual(len(triangles), 2)
        self.assertGreaterEqual(len(vertices), 4)  # At least 4 corners
        
        # Check that z values increase along the slope
        vertices_by_y = {}
        for vertex in vertices:
            y = vertex[1]
            if y not in vertices_by_y:
                vertices_by_y[y] = []
            vertices_by_y[y].append(vertex[2])
        
        # Get unique y coordinates sorted
        y_coords = sorted(vertices_by_y.keys())
        if len(y_coords) > 1:
            # Height values at higher y should be greater
            avg_height_at_min_y = sum(vertices_by_y[y_coords[0]]) / len(vertices_by_y[y_coords[0]])
            avg_height_at_max_y = sum(vertices_by_y[y_coords[-1]]) / len(vertices_by_y[y_coords[-1]])
            self.assertGreater(avg_height_at_max_y, avg_height_at_min_y)
    
    def test_run_peak(self):
        """Test triangulation on a heightmap with a central peak."""
        triangulator = AdaptiveTriangulator(self.height_map_peak, error_threshold=0.01)
        vertices, triangles = triangulator.run()
        
        # A heightmap with a peak should result in more triangles for accuracy
        self.assertGreater(len(triangles), 2)
        
        # The triangulation should produce valid triangle indices
        for triangle in triangles:
            self.assertEqual(len(triangle), 3)
            for idx in triangle:
                self.assertGreaterEqual(idx, 0)
                self.assertLess(idx, len(vertices))
        
        # Check that the highest point is near the center
        max_height = 0
        max_height_pos = (0, 0)
        for vertex in vertices:
            if vertex[2] > max_height:
                max_height = vertex[2]
                max_height_pos = (vertex[0], vertex[1])
        
        # Center of the peak should be around (5,5) ± 2 units
        self.assertLess(abs(max_height_pos[0] - 5), 2)
        self.assertLess(abs(max_height_pos[1] - 5), 2)
    
    def test_max_triangles_limit(self):
        """Test that max_triangles limit is respected."""
        max_triangles = 10
        triangulator = AdaptiveTriangulator(self.height_map_peak, 
                                          max_triangles=max_triangles,
                                          error_threshold=0.0001)  # Force subdivision
        vertices, triangles = triangulator.run()
        
        # Should not exceed max_triangles
        self.assertLessEqual(len(triangles), max_triangles)
    
    def test_override_parameters(self):
        """Test parameter overrides in run method."""
        triangulator = AdaptiveTriangulator(self.height_map_peak, 
                                          max_triangles=100,
                                          error_threshold=0.1)
        
        # Override with stricter parameters
        vertices, triangles = triangulator.run(max_error=0.01, max_triangles=10)
        
        # Should respect the override parameters
        self.assertLessEqual(len(triangles), 10)
    
    @patch('logging.getLogger')
    def test_logging(self, mock_get_logger):
        """Test logging during triangulation."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        triangulator = AdaptiveTriangulator(self.height_map_peak)
        vertices, triangles = triangulator.run()
        
        # Verify logging was called
        mock_logger.info.assert_called()
    
    def test_triangle_area(self):
        """Test triangle area calculation."""
        triangulator = AdaptiveTriangulator(self.height_map_flat)
        area = triangulator._triangle_area([0, 0, 0], [3, 0, 0], [0, 4, 0])
        self.assertEqual(area, 6.0)  # Triangle with base 3 and height 4 has area 6
    
    def test_point_in_triangle(self):
        """Test point in triangle detection."""
        triangulator = AdaptiveTriangulator(self.height_map_flat)
        
        # Define a triangle
        v1 = (0, 0)
        v2 = (10, 0)
        v3 = (5, 10)
        
        # Test points inside
        self.assertTrue(triangulator._point_in_triangle((5, 5), v1, v2, v3))
        self.assertTrue(triangulator._point_in_triangle((2, 2), v1, v2, v3))
        
        # Test points outside
        self.assertFalse(triangulator._point_in_triangle((20, 5), v1, v2, v3))
        self.assertFalse(triangulator._point_in_triangle((5, 15), v1, v2, v3))
        
        # Edge cases - points exactly on vertices or edges
        self.assertTrue(triangulator._point_in_triangle(v1, v1, v2, v3))
        self.assertTrue(triangulator._point_in_triangle((5, 0), v1, v2, v3))  # On edge


if __name__ == '__main__':
    unittest.main()
