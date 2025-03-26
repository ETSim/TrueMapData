"""Unit tests for TMD PLY module."""

import unittest
import numpy as np
import os
import sys
import struct
import tempfile
from unittest.mock import patch, MagicMock

# Add the project root to the path to import tmd modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from tmd.exporters.model.ply import convert_heightmap_to_ply, _write_binary_ply, _generate_vertex_colors


class TestPly(unittest.TestCase):
    """Test class for PLY export functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test heightmaps
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
    
    def test_input_validation(self):
        """Test input validation for PLY export."""
        # Test with None heightmap
        result = convert_heightmap_to_ply(None, "test.ply")
        self.assertIsNone(result)
        
        # Test with empty heightmap
        result = convert_heightmap_to_ply(np.array([]), "test.ply")
        self.assertIsNone(result)
        
        # Test with too small heightmap
        result = convert_heightmap_to_ply(np.array([[1]]), "test.ply")
        self.assertIsNone(result)
    
    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    @patch('tmd.exporters.model.ply.create_mesh_from_heightmap')
    @patch('tmd.exporters.model.ply.ensure_directory_exists')
    @patch('tmd.exporters.model.ply._write_binary_ply')
    def test_binary_ply_export(self, mock_write_binary, mock_ensure_dir, mock_create_mesh, mock_open):
        """Test binary PLY export functionality."""
        # Setup mocks
        mock_ensure_dir.return_value = True
        
        # Mock the mesh creation
        vertices = [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0]
        ]
        faces = [
            [0, 1, 2],
            [1, 3, 2]
        ]
        mock_create_mesh.return_value = (vertices, faces)
        
        # Test export with binary format
        result = convert_heightmap_to_ply(
            self.heightmap_flat,
            filename="test.ply",
            z_scale=1.0,
            calculate_normals=True,
            add_color=True
        )
        
        # Check that file was opened in binary mode
        mock_open.assert_called_once_with("test.ply", 'wb')
        
        # Check that binary writer was called
        mock_write_binary.assert_called_once()
        
        # Result should be the filename
        self.assertEqual(result, "test.ply")
    
    @patch('tmd.exporters.model.ply._generate_vertex_colors')
    @patch('tmd.exporters.model.ply.calculate_vertex_normals')
    @patch('tmd.exporters.model.ply.create_mesh_from_heightmap')
    @patch('tmd.exporters.model.ply.ensure_directory_exists')
    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    def test_ply_with_colors_and_normals(self, mock_open, mock_ensure_dir, mock_create_mesh, 
                                          mock_calc_normals, mock_gen_colors):
        """Test PLY export with colors and normals."""
        # Setup mocks
        mock_ensure_dir.return_value = True
        
        # Mock the mesh creation
        vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]
        faces = [[0, 1, 2], [1, 3, 2]]
        mock_create_mesh.return_value = (vertices, faces)
        
        # Mock normals and colors
        mock_normals = np.zeros((4, 3), dtype=np.float32)
        mock_normals[:, 2] = 1.0  # All normals point up
        mock_calc_normals.return_value = mock_normals
        
        mock_colors = np.zeros((4, 3), dtype=np.uint8)
        mock_colors[:, 0] = 255  # Red color
        mock_gen_colors.return_value = mock_colors
        
        # Create a spy for _write_binary_ply to check its arguments
        original_write_binary = _write_binary_ply
        
        with patch('tmd.exporters.model.ply._write_binary_ply') as mock_write_binary:
            def side_effect(file_obj, vertices_arg, faces_arg, normals_arg, colors_arg):
                # Check that normals and colors are passed correctly
                self.assertIsNotNone(normals_arg)
                self.assertIsNotNone(colors_arg)
                self.assertTrue(np.array_equal(normals_arg, mock_normals))
                self.assertTrue(np.array_equal(colors_arg, mock_colors))
                # Call the original implementation
                # original_write_binary(file_obj, vertices_arg, faces_arg, normals_arg, colors_arg)
                
            mock_write_binary.side_effect = side_effect
            
            # Test export with colors and normals
            result = convert_heightmap_to_ply(
                self.heightmap_flat,
                filename="test.ply",
                z_scale=1.0,
                calculate_normals=True,
                add_color=True
            )
            
            # Verify that the function was called
            mock_write_binary.assert_called_once()
            
            # Result should be the filename
            self.assertEqual(result, "test.ply")
    
    def test_generate_vertex_colors(self):
        """Test generation of vertex colors from height values."""
        # Create test vertices with varying Z values
        vertices = np.array([
            [0, 0, 0],
            [0, 0, 0.5],
            [0, 0, 1.0]
        ])
        
        # Test with matplotlib available
        with patch('matplotlib.cm.get_cmap') as mock_get_cmap:
            # Mock the colormap function to return predictable colors
            def mock_colormap(values):
                # Return RGB values proportional to input values
                result = np.zeros((len(values), 4))
                result[:, 0] = values * 0.8  # R increases with height
                result[:, 1] = 0.2  # G constant
                result[:, 2] = 1.0 - values * 0.8  # B decreases with height
                result[:, 3] = 1.0  # Alpha
                return result
                
            mock_cmap = MagicMock()
            mock_cmap.side_effect = mock_colormap
            mock_get_cmap.return_value = mock_cmap
            
            colors = _generate_vertex_colors(vertices, self.heightmap_flat, 'terrain')
            
            # Should return colors for each vertex
            self.assertEqual(colors.shape, (3, 3))
            
            # Colors should be in the correct range
            self.assertTrue(np.all(colors >= 0))
            self.assertTrue(np.all(colors <= 255))
            
            # Colors should vary with height (first vertex has lowest Z)
            self.assertTrue(np.all(colors[0, 0] < colors[1, 0]))  # Red increases with height
            self.assertTrue(np.all(colors[0, 2] > colors[1, 2]))  # Blue decreases with height
    
    @patch('tmd.exporters.model.ply.ensure_directory_exists')
    def test_directory_creation_failure(self, mock_ensure_dir):
        """Test handling of directory creation failure."""
        mock_ensure_dir.return_value = False
        
        result = convert_heightmap_to_ply(
            self.heightmap_flat,
            filename="test.ply"
        )
        
        # Should return None if directory creation fails
        self.assertIsNone(result)
        mock_ensure_dir.assert_called_once_with("test.ply")
    
    @patch('builtins.open')
    @patch('tmd.exporters.model.ply.ensure_directory_exists')
    def test_error_handling(self, mock_ensure_dir, mock_open):
        """Test error handling in PLY export."""
        mock_ensure_dir.return_value = True
        
        # Make open raise an exception
        mock_open.side_effect = IOError("Test error")
        
        # Test export with IO error
        result = convert_heightmap_to_ply(
            self.heightmap_flat,
            filename="test.ply"
        )
        
        # Should return None on error
        self.assertIsNone(result)
    
    @unittest.skipIf(not os.getenv('RUN_INTEGRATION_TESTS'), "Integration tests disabled")
    def test_integration_real_file(self):
        """Test actual file creation (only runs if RUN_INTEGRATION_TESTS is set)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_file = os.path.join(tmp_dir, "test_real_output.ply")
            
            # Test binary PLY output
            result = convert_heightmap_to_ply(
                self.heightmap_peak,
                filename=output_file,
                z_scale=1.0,
                add_color=True
            )
            
            # Check that file exists
            self.assertTrue(os.path.isfile(output_file))
            
            # Basic check of file content
            with open(output_file, 'rb') as f:
                header = f.read(100)  # Read part of the header
                header_str = header.decode('ascii', errors='ignore')
                
                # Should have PLY header
                self.assertTrue("ply" in header_str)
                self.assertTrue("format binary_little_endian" in header_str)
                
                # Should have vertex and face elements
                self.assertTrue("element vertex" in header_str)
                self.assertTrue("element face" in header_str)


if __name__ == '__main__':
    unittest.main()
