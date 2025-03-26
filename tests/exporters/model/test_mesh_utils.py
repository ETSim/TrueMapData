"""Unit tests for TMD mesh utils module."""

import unittest
import numpy as np
import os
import sys
import tempfile
from unittest.mock import patch, MagicMock

# Add the project root to the path to import tmd modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from tmd.exporters.model.mesh_utils import (
    calculate_vertex_normals,
    calculate_face_normals,
    calculate_heightmap_normals,
    optimize_mesh,
    validate_heightmap,
    ensure_directory_exists,
    generate_uv_coordinates
)


class TestMeshUtils(unittest.TestCase):
    """Test class for mesh utils functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Simple test mesh - pyramid
        self.vertices = np.array([
            [0, 0, 0],   # base
            [1, 0, 0],   # base 
            [1, 1, 0],   # base
            [0, 1, 0],   # base
            [0.5, 0.5, 1]  # top
        ], dtype=np.float32)
        
        self.faces = np.array([
            [0, 1, 4],  # side
            [1, 2, 4],  # side
            [2, 3, 4],  # side
            [3, 0, 4],  # side
            [0, 3, 1],  # base
            [1, 3, 2]   # base
        ], dtype=np.int32)
        
        # Test heightmaps
        self.heightmap_flat = np.zeros((5, 5), dtype=np.float32)
        
        self.heightmap_slope = np.zeros((5, 5), dtype=np.float32)
        for i in range(5):
            for j in range(5):
                self.heightmap_slope[i, j] = i / 5.0
    
    def test_calculate_vertex_normals(self):
        """Test calculation of vertex normals."""
        normals = calculate_vertex_normals(self.vertices, self.faces)
        
        # Should return one normal per vertex
        self.assertEqual(normals.shape, (5, 3))
        
        # All normals should be unit length
        lengths = np.linalg.norm(normals, axis=1)
        for length in lengths:
            self.assertAlmostEqual(length, 1.0, places=5)
        
        # Top vertex normal should point upward
        self.assertGreater(normals[4, 2], 0.9)  # Z component close to 1
    
    def test_calculate_face_normals(self):
        """Test calculation of face normals."""
        normals = calculate_face_normals(self.vertices, self.faces)
        
        # Should return one normal per face
        self.assertEqual(normals.shape, (6, 3))
        
        # All normals should be unit length
        lengths = np.linalg.norm(normals, axis=1)
        for length in lengths:
            self.assertAlmostEqual(length, 1.0, places=5)
        
        # Base faces should have normals pointing down
        self.assertLess(normals[4, 2], -0.9)  # Z component close to -1
        self.assertLess(normals[5, 2], -0.9)  # Z component close to -1
    
    def test_calculate_heightmap_normals(self):
        """Test calculation of heightmap normals."""
        # Test with flat heightmap
        normals_flat = calculate_heightmap_normals(self.heightmap_flat)
        
        # Should return a normal for each heightmap point
        self.assertEqual(normals_flat.shape, (5, 5, 3))
        
        # All normals should point straight up for flat heightmap
        for i in range(5):
            for j in range(5):
                normal = normals_flat[i, j]
                self.assertAlmostEqual(normal[0], 0.0, places=5)
                self.assertAlmostEqual(normal[1], 0.0, places=5)
                self.assertAlmostEqual(normal[2], 1.0, places=5)
        
        # Test with sloped heightmap
        normals_slope = calculate_heightmap_normals(self.heightmap_slope)
        
        # Should have same shape as input
        self.assertEqual(normals_slope.shape, (5, 5, 3))
        
        # Slope normals should have negative Y component
        all_y_negative = np.all(normals_slope[:, :, 1] < 0)
        self.assertTrue(all_y_negative)
    
    def test_optimize_mesh(self):
        """Test mesh optimization by merging vertices."""
        # Create a mesh with duplicate vertices
        vertices_with_duplicates = np.array([
            [0, 0, 0],   # Original vertices
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 0],   # Duplicate of vertex 0
            [1, 0, 0.00000001]   # Very close to vertex 1
        ])
        
        faces_with_duplicates = np.array([
            [0, 1, 2],
            [3, 1, 2],  # Using duplicate vertex 3 instead of 0
            [4, 2, 0]   # Using near-duplicate vertex 4
        ])
        
        # Optimize the mesh
        optimized_vertices, optimized_faces = optimize_mesh(vertices_with_duplicates, faces_with_duplicates)
        
        # Should have only 3 unique vertices after optimization
        self.assertEqual(len(optimized_vertices), 3)
        
        # Should still have 3 faces (none are degenerate)
        self.assertEqual(len(optimized_faces), 3)
        
        # Face indices should be updated to use the merged vertices
        for face in optimized_faces:
            self.assertTrue(np.all(face < 3))  # All indices should be < 3
    
    def test_validate_heightmap(self):
        """Test heightmap validation."""
        # Valid heightmaps
        self.assertTrue(validate_heightmap(self.heightmap_flat))
        self.assertTrue(validate_heightmap(self.heightmap_slope))
        
        # Invalid heightmaps
        self.assertFalse(validate_heightmap(None))
        self.assertFalse(validate_heightmap(np.array([])))
        self.assertFalse(validate_heightmap(np.array([1])))  # 1D array
        self.assertFalse(validate_heightmap(np.array([[1]])))  # Too small (1x1)
        
        # 3D array is invalid
        self.assertFalse(validate_heightmap(np.zeros((2, 2, 2))))
    
    @patch('os.makedirs')
    def test_ensure_directory_exists(self, mock_makedirs):
        """Test directory creation."""
        # Test successful directory creation
        mock_makedirs.side_effect = None
        self.assertTrue(ensure_directory_exists("test/file.txt"))
        mock_makedirs.assert_called_once()
        
        # Test directory creation failure
        mock_makedirs.reset_mock()
        mock_makedirs.side_effect = PermissionError("Test error")
        with patch('builtins.print') as mock_print:
            self.assertFalse(ensure_directory_exists("test/file2.txt"))
            mock_print.assert_called_once()
    
    def test_generate_uv_coordinates(self):
        """Test generation of UV coordinates."""
        uvs = generate_uv_coordinates(self.vertices)
        
        # Should return one UV pair per vertex
        self.assertEqual(uvs.shape, (5, 2))
        
        # UVs should be in range [0,1]
        self.assertTrue(np.all(uvs >= 0))
        self.assertTrue(np.all(uvs <= 1))
        
        # Check specific values
        # Bottom corners should have predictable UVs
        self.assertAlmostEqual(uvs[0, 0], 0.0)  # bottom-left, u=0
        self.assertAlmostEqual(uvs[0, 1], 1.0)  # bottom-left, v=1 (flipped)
        self.assertAlmostEqual(uvs[2, 0], 1.0)  # top-right, u=1
        self.assertAlmostEqual(uvs[2, 1], 0.0)  # top-right, v=0 (flipped)


if __name__ == '__main__':
    unittest.main()
