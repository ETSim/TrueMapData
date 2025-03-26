"""Unit tests for TMD gltf module."""

import os
import sys
import json
import unittest
import tempfile
import base64
import struct
import numpy as np
from unittest.mock import patch, MagicMock

# Add the project root to the path to import tmd modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from tmd.exporters.model.gltf import (
    convert_heightmap_to_gltf,
    convert_heightmap_to_glb,
    heightmap_to_mesh,
    _create_gltf_structure,
    _add_material,
    _generate_texture_from_heightmap
)

# Check if trimesh is available for mesh generation tests
try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False


@unittest.skipIf(not TRIMESH_AVAILABLE, "trimesh is not available")
class TestGltf(unittest.TestCase):
    """Test class for gltf functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create simple test heightmaps
        self.heightmap_flat = np.zeros((10, 10), dtype=np.float32)
        
        self.heightmap_slope = np.zeros((10, 10), dtype=np.float32)
        for i in range(10):
            for j in range(10):
                self.heightmap_slope[i, j] = i / 10.0
        
        self.heightmap_peak = np.zeros((10, 10), dtype=np.float32)
        for i in range(10):
            for j in range(10):
                self.heightmap_peak[i, j] = 1.0 - ((i-5)**2 + (j-5)**2) / 50.0
                if self.heightmap_peak[i, j] < 0:
                    self.heightmap_peak[i, j] = 0
    
    def test_heightmap_to_mesh(self):
        """Test basic mesh creation from heightmap."""
        vertices, faces = heightmap_to_mesh(self.heightmap_flat)
        
        # Check that mesh was created
        self.assertIsNotNone(vertices)
        self.assertIsNotNone(faces)
        
        # Should have 10x10=100 vertices
        self.assertEqual(len(vertices), 100)
        
        # Should have (10-1)x(10-1)x2=162 triangles (9x9 grid, 2 triangles per quad)
        self.assertEqual(len(faces), 162)
        
        # Test with base height
        vertices, faces = heightmap_to_mesh(
            self.heightmap_flat,
            base_height=0.5
        )
        
        # Should have 2x original vertices (top and bottom) for the base
        self.assertEqual(len(vertices), 200)
        
        # Should have original triangles + base triangles + sides
        self.assertGreater(len(faces), 162)
        
        # Test with invalid input
        empty_map = np.array([])
        vertices, faces = heightmap_to_mesh(empty_map)
        self.assertIsNone(vertices)
        self.assertIsNone(faces)
    
    def test_generate_uv_coordinates(self):
        """Test UV coordinate generation."""
        # Create some test vertices
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0]
        ])
        
        # Generate UVs
        uvs = _generate_uv_coordinates(vertices)
        
        # Check shape
        self.assertEqual(uvs.shape, (4, 2))
        
        # Check values (should be normalized to 0-1 range)
        self.assertEqual(uvs[0, 0], 0.0)  # u for vertex 0
        self.assertEqual(uvs[0, 1], 0.0)  # v for vertex 0
        self.assertEqual(uvs[2, 0], 1.0)  # u for vertex 2
        self.assertEqual(uvs[2, 1], 1.0)  # v for vertex 2
    
    @unittest.skipUnless(TRIMESH_AVAILABLE, "trimesh is required")
    @patch('os.makedirs')
    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    @patch('trimesh.Trimesh')
    @patch('trimesh.Scene')
    def test_convert_heightmap_to_gltf(self, mock_scene, mock_trimesh, mock_open, mock_makedirs):
        """Test GLTF conversion with mocked file operations."""
        # Setup mocks
        mock_mesh = MagicMock()
        mock_trimesh.return_value = mock_mesh
        mock_scene_instance = MagicMock()
        mock_scene.return_value = mock_scene_instance
        mock_scene_instance.export.return_value = "test_gltf_content"
        
        # Test conversion
        result = convert_heightmap_to_gltf(
            self.heightmap_flat,
            filename="test.gltf",
            z_scale=1.0,
            base_height=0.0
        )
        
        # Check that file was created
        mock_makedirs.assert_called_once()
        mock_open.assert_called_once()
        self.assertEqual(result, "test.gltf")
        
        # Test with invalid input
        mock_scene.reset_mock()
        mock_open.reset_mock()
        
        result = convert_heightmap_to_gltf(
            np.array([]),  # Empty array
            filename="test.gltf"
        )
        
        # Should return None for invalid input
        self.assertIsNone(result)
        mock_open.assert_not_called()
    
    @unittest.skipUnless(TRIMESH_AVAILABLE, "trimesh is required")
    @patch('tmd.exporters.model.gltf.convert_heightmap_to_gltf')
    def test_convert_heightmap_to_glb(self, mock_convert):
        """Test GLB conversion."""
        # Setup mock
        mock_convert.return_value = "test.glb"
        
        # Test conversion
        result = convert_heightmap_to_glb(
            self.heightmap_flat,
            filename="test.glb",
            z_scale=1.0
        )
        
        # Check that the convert_heightmap_to_gltf function was called with binary=True
        mock_convert.assert_called_once()
        call_args = mock_convert.call_args[1]
        self.assertTrue(call_args["binary"])
        self.assertEqual(result, "test.glb")
        
        # Test with filename without extension
        mock_convert.reset_mock()
        mock_convert.return_value = "test.glb"
        
        result = convert_heightmap_to_glb(
            self.heightmap_flat,
            filename="test"  # No extension
        )
        
        # Should add .glb extension
        self.assertEqual(mock_convert.call_args[1]["filename"], "test.glb")
    
    @unittest.skipUnless(TRIMESH_AVAILABLE, "trimesh is required")
    @patch('tmd.exporters.model.gltf.convert_heightmap_to_gltf')
    def test_export_functions(self, mock_convert):
        """Test convenience export functions."""
        # Setup mock
        mock_convert.return_value = "test.gltf"
        
        # Test GLTF export
        result = export_gltf(
            self.heightmap_flat,
            output_file="test.gltf"
        )
        
        # Check function was called correctly
        mock_convert.assert_called_once()
        self.assertEqual(result, "test.gltf")
        
        # Test GLB export
        mock_convert.reset_mock()
        mock_convert.return_value = "test.glb"
        
        result = export_glb(
            self.heightmap_flat,
            output_file="test.glb"
        )
        
        # Check the binary parameter was passed
        mock_convert.assert_called_once()
        self.assertTrue(mock_convert.call_args[1]["binary"])
        self.assertEqual(result, "test.glb")


if __name__ == '__main__':
    unittest.main()
