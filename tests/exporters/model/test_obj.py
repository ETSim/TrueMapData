"""Unit tests for TMD OBJ module."""

import unittest
import numpy as np
import os
import sys
import tempfile
from unittest.mock import patch, MagicMock

# Add the project root to the path to import tmd modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from tmd.exporters.model.obj import convert_heightmap_to_obj


class TestObj(unittest.TestCase):
    """Test class for OBJ export functionality."""
    
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
        """Test input validation for OBJ export."""
        # Test with None heightmap
        result = convert_heightmap_to_obj(None, "test.obj")
        self.assertIsNone(result)
        
        # Test with empty heightmap
        result = convert_heightmap_to_obj(np.array([]), "test.obj")
        self.assertIsNone(result)
        
        # Test with too small heightmap
        result = convert_heightmap_to_obj(np.array([[1]]), "test.obj")
        self.assertIsNone(result)
    
    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    @patch('tmd.exporters.model.obj.create_mesh_from_heightmap')
    @patch('tmd.exporters.model.obj.ensure_directory_exists')
    def test_obj_export_basic(self, mock_ensure_dir, mock_create_mesh, mock_open):
        """Test basic OBJ export functionality."""
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
        
        # Test export
        result = convert_heightmap_to_obj(
            self.heightmap_flat,
            filename="test.obj",
            z_scale=1.0,
            calculate_normals=True
        )
        
        # Check that file was opened correctly
        mock_open.assert_called()
        mock_open().__enter__().write.assert_called()
        
        # Result should be the filename
        self.assertEqual(result, "test.obj")
        
        # Check that OBJ contents were written
        write_calls = mock_open().__enter__().write.call_args_list
        
        # Check for vertex definitions
        vertices_written = sum(1 for call in write_calls if call[0][0].startswith('v '))
        self.assertEqual(vertices_written, len(vertices))
        
        # Check for normal definitions
        normals_written = sum(1 for call in write_calls if call[0][0].startswith('vn '))
        self.assertEqual(normals_written, len(vertices))
        
        # Check for face definitions
        faces_written = sum(1 for call in write_calls if call[0][0].startswith('f '))
        self.assertEqual(faces_written, len(faces))
    
    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    @patch('tmd.exporters.model.obj.create_mesh_from_heightmap')
    @patch('tmd.exporters.model.obj.ensure_directory_exists')
    def test_obj_with_materials(self, mock_ensure_dir, mock_create_mesh, mock_open):
        """Test OBJ export with materials."""
        # Setup mocks
        mock_ensure_dir.return_value = True
        
        # Mock the mesh creation
        vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]
        faces = [[0, 1, 2], [1, 3, 2]]
        mock_create_mesh.return_value = (vertices, faces)
        
        # Test export with materials
        result = convert_heightmap_to_obj(
            self.heightmap_flat,
            filename="test.obj",
            include_materials=True
        )
        
        # Should open two files (OBJ and MTL)
        self.assertEqual(mock_open.call_count, 2)
        
        # Check for MTL reference in OBJ file
        mtllib_written = False
        for call in mock_open().__enter__().write.call_args_list:
            if call[0][0].startswith('mtllib'):
                mtllib_written = True
                break
        self.assertTrue(mtllib_written)
        
        # Result should be the filename
        self.assertEqual(result, "test.obj")
    
    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    @patch('tmd.exporters.model.obj.create_mesh_from_heightmap')
    @patch('tmd.exporters.model.obj.ensure_directory_exists')
    def test_obj_with_base(self, mock_ensure_dir, mock_create_mesh, mock_open):
        """Test OBJ export with a base."""
        # Setup mocks
        mock_ensure_dir.return_value = True
        
        # Mock the mesh creation - vertices include base vertices
        vertices = [
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],  # Surface
            [0, 0, -0.5], [1, 0, -0.5], [0, 1, -0.5], [1, 1, -0.5]  # Base
        ]
        faces = [
            [0, 1, 2], [1, 3, 2],  # Surface
            [4, 5, 6], [5, 7, 6]   # Base
        ]
        mock_create_mesh.return_value = (vertices, faces)
        
        # Test export with base
        result = convert_heightmap_to_obj(
            self.heightmap_flat,
            filename="test.obj",
            base_height=0.5
        )
        
        # Check for group definitions
        groups_written = 0
        for call in mock_open().__enter__().write.call_args_list:
            if call[0][0].startswith('g '):
                groups_written += 1
        
        # Should have 2 groups (surface and base)
        self.assertEqual(groups_written, 2)
        
        # Result should be the filename
        self.assertEqual(result, "test.obj")
    
    @patch('builtins.open')
    @patch('tmd.exporters.model.obj.ensure_directory_exists')
    def test_error_handling(self, mock_ensure_dir, mock_open):
        """Test error handling in OBJ export."""
        # Setup directory exists
        mock_ensure_dir.return_value = True
        
        # Make open raise an exception
        mock_open.side_effect = IOError("Test error")
        
        # Test export with IO error
        result = convert_heightmap_to_obj(
            self.heightmap_flat,
            filename="test.obj"
        )
        
        # Should return None on error
        self.assertIsNone(result)
    
    @patch('tmd.exporters.model.obj.ensure_directory_exists')
    def test_directory_creation_failure(self, mock_ensure_dir):
        """Test handling of directory creation failure."""
        # Setup directory creation failure
        mock_ensure_dir.return_value = False
        
        # Test export with directory creation failure
        result = convert_heightmap_to_obj(
            self.heightmap_flat,
            filename="test.obj"
        )
        
        # Should return None if directory creation fails
        self.assertIsNone(result)
        mock_ensure_dir.assert_called_once_with("test.obj")
    
    @unittest.skipIf(not os.getenv('RUN_INTEGRATION_TESTS'), "Integration tests disabled")
    def test_integration_real_file(self):
        """Test actual file creation (only runs if RUN_INTEGRATION_TESTS is set)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_file = os.path.join(tmp_dir, "test_real_output.obj")
            mtl_file = os.path.join(tmp_dir, "test_real_output.mtl")
            
            # Test OBJ output with materials
            result = convert_heightmap_to_obj(
                self.heightmap_peak,
                filename=output_file,
                z_scale=1.0,
                include_materials=True
            )
            
            # Check that files exist
            self.assertTrue(os.path.isfile(output_file))
            self.assertTrue(os.path.isfile(mtl_file))
            
            # Basic check of OBJ file content
            with open(output_file, 'r') as f:
                content = f.read()
                self.assertIn("mtllib", content)
                self.assertIn("v ", content)  # Vertices
                self.assertIn("vt ", content)  # Texture coordinates
                self.assertIn("vn ", content)  # Normals
                self.assertIn("f ", content)  # Faces


if __name__ == '__main__':
    unittest.main()
