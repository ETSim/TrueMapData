"""Unit tests for TMD STL module."""

import unittest
import numpy as np
import os
import sys
import struct
import tempfile
from unittest.mock import patch, MagicMock, call

# Add the project root to the path to import tmd modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from tmd.exporters.model.stl import convert_heightmap_to_stl, _ensure_watertight_mesh


class TestStl(unittest.TestCase):
    """Test class for STL export functionality."""
    
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
        """Test input validation for STL export."""
        # Test with None heightmap
        result = convert_heightmap_to_stl(None, "test.stl")
        self.assertIsNone(result)
        
        # Test with empty heightmap
        result = convert_heightmap_to_stl(np.array([]), "test.stl")
        self.assertIsNone(result)
        
        # Test with too small heightmap
        result = convert_heightmap_to_stl(np.array([[1]]), "test.stl")
        self.assertIsNone(result)
    
    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    @patch('os.makedirs')
    @patch('tmd.exporters.model.stl.create_mesh_from_heightmap')
    def test_binary_export(self, mock_create_mesh, mock_makedirs, mock_open):
        """Test binary STL export functionality."""
        # Mock the create_mesh_from_heightmap function
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
        
        # Mock file operations
        mock_file = mock_open.return_value.__enter__.return_value
        
        # Test with binary format
        result = convert_heightmap_to_stl(
            self.heightmap_flat,
            filename="test.stl",
            z_scale=1.0
        )
        
        # Check that file was opened in binary write mode
        mock_open.assert_called_once_with("test.stl", 'wb')
        
        # Check header and triangle count writes
        header_write = mock_file.write.call_args_list[0]
        self.assertEqual(len(header_write[0][0]), 80)  # 80-byte header
        
        triangle_count_write = mock_file.write.call_args_list[1]
        self.assertEqual(len(triangle_count_write[0][0]), 4)  # 4-byte triangle count
        
        # Check that the result is the expected filename
        self.assertEqual(result, "test.stl")
    
    @patch('tmd.exporters.model.stl.convert_heightmap_to_adaptive_mesh')
    def test_adaptive_export(self, mock_adaptive):
        """Test adaptive STL export option."""
        # Setup mock for adaptive mesh generation
        mock_adaptive.return_value = ("vertices", "faces", "test_adaptive.stl")
        
        # Test with adaptive meshing enabled
        result = convert_heightmap_to_stl(
            self.heightmap_flat,
            filename="test.stl",
            adaptive=True,
            error_threshold=0.01
        )
        
        # Check that adaptive mesh function was called
        mock_adaptive.assert_called_once()
        
        # Check that error threshold was passed correctly
        self.assertEqual(mock_adaptive.call_args[1]["error_threshold"], 0.01)
        
        # Check that binary format was specified (no ASCII)
        self.assertEqual(mock_adaptive.call_args[1]["ascii"], False)
        
        # Result should be the filename returned by adaptive meshing
        self.assertEqual(result, "test_adaptive.stl")
    
    @patch('builtins.open')
    def test_error_handling(self, mock_open):
        """Test error handling in STL export."""
        # Make open raise an exception
        mock_open.side_effect = IOError("Test error")
        
        # Test export with IO error
        result = convert_heightmap_to_stl(
            self.heightmap_flat,
            filename="test.stl"
        )
        
        # Should return None on error
        self.assertIsNone(result)
    
    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    @patch('os.makedirs')
    @patch('tmd.exporters.model.stl.create_mesh_from_heightmap')
    def test_base_triangulation(self, mock_create_mesh, mock_makedirs, mock_open):
        """Test that the base uses minimal triangles."""
        # Create test mesh with base
        vertices = []
        # Add 3x3 grid of vertices
        for y in range(3):
            for x in range(3):
                vertices.append([float(x), float(y), 0.0])
        
        # Create faces for the top surface
        faces = [
            [0, 1, 3], [1, 4, 3],  # Top-left quad
            [1, 2, 4], [2, 5, 4],  # Top-right quad
            [3, 4, 6], [4, 7, 6],  # Bottom-left quad
            [4, 5, 7], [5, 8, 7]   # Bottom-right quad
        ]
        
        # Create 5 vertices for the base (center + 4 corners)
        base_vertices = vertices.copy()
        # Center of base
        base_vertices.append([1.0, 1.0, -0.5])  # at index 9
        # Corners of base
        base_vertices.append([0.0, 0.0, -0.5])  # at index 10
        base_vertices.append([2.0, 0.0, -0.5])  # at index 11
        base_vertices.append([2.0, 2.0, -0.5])  # at index 12
        base_vertices.append([0.0, 2.0, -0.5])  # at index 13
        
        # Add 4 triangles for the base (connecting center to corners)
        base_faces = faces.copy()
        base_faces.append([9, 10, 11])  # center to bottom edge
        base_faces.append([9, 11, 12])  # center to right edge
        base_faces.append([9, 12, 13])  # center to top edge
        base_faces.append([9, 13, 10])  # center to left edge
        
        # Set the mock to return our mesh with base
        mock_create_mesh.return_value = (base_vertices, base_faces)
        
        # Test STL export
        result = convert_heightmap_to_stl(
            self.heightmap_flat,
            filename="test.stl",
            base_height=0.5
        )
        
        # Inspect the binary data that was written
        write_calls = mock_open().write.call_args_list
        
        # The first two calls are header and triangle count
        # Then there should be 12 triangles (8 surface + 4 base)
        # Each triangle is 50 bytes (3 floats for normal, 9 floats for vertices, 2 bytes for attribute)
        self.assertEqual(len(write_calls), 2 + 12)
        
        # Check that we get the correct result
        self.assertEqual(result, "test.stl")
    
    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    @patch('os.makedirs')
    @patch('tmd.exporters.model.stl.create_mesh_from_heightmap')
    @patch('tmd.exporters.model.stl._ensure_watertight_mesh')
    def test_watertight_option(self, mock_ensure_watertight, mock_create_mesh, mock_makedirs, mock_open):
        """Test the watertight mesh option."""
        # Mock mesh creation to return a simple mesh without base
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
        
        # Mock the watertight function to return a modified mesh
        watertight_vertices = vertices + [[0.5, 0.5, -0.001]]  # Add center base vertex
        watertight_faces = faces + [[0, 2, 4], [1, 0, 4], [3, 1, 4], [2, 3, 4]]  # Add base triangles
        mock_ensure_watertight.return_value = (watertight_vertices, watertight_faces)
        
        # Test with watertight enabled
        result = convert_heightmap_to_stl(
            self.heightmap_flat,
            filename="test.stl",
            base_height=0.0,  # No explicit base
            ensure_watertight=True  # But ensure it's watertight
        )
        
        # Check that watertight function was called
        mock_ensure_watertight.assert_called_once()
        
        # Check that the result is the expected filename
        self.assertEqual(result, "test.stl")
    
    @unittest.skipIf(not os.getenv('RUN_INTEGRATION_TESTS'), "Integration tests disabled")
    def test_integration_real_file(self):
        """Test actual file creation (only runs if RUN_INTEGRATION_TESTS is set)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_file = os.path.join(tmp_dir, "test_real_output.stl")
            
            # Test binary STL output
            result = convert_heightmap_to_stl(
                self.heightmap_peak,
                filename=output_file,
                z_scale=1.0
            )
            
            # Check that file exists
            self.assertTrue(os.path.isfile(output_file))
            
            # Binary STL has 80-byte header, 4-byte triangle count
            with open(output_file, 'rb') as f:
                header = f.read(80)
                self.assertEqual(len(header), 80)
                
                triangle_count_bytes = f.read(4)
                triangle_count = struct.unpack("<I", triangle_count_bytes)[0]
                self.assertGreater(triangle_count, 0)
    
    @patch('tmd.exporters.model.base._add_base_to_mesh')
    def test_ensure_watertight_mesh(self, mock_add_base):
        """Test the _ensure_watertight_mesh function."""
        # Setup test data
        vertices = [
            [0, 0, 0],
            [1, 0, 0], 
            [0, 1, 0]
        ]
        faces = [
            [0, 1, 2]
        ]
        
        # Mock the base function to return a known result
        mock_add_base.return_value = (vertices + [[0.5, 0.5, -0.001]], faces + [[0, 1, 3]])
        
        # Call the function
        result_vertices, result_faces = _ensure_watertight_mesh(vertices, faces)
        
        # Check that add_base was called with correct parameters
        mock_add_base.assert_called_once()
        args = mock_add_base.call_args[0]
        self.assertIs(args[0], vertices)
        self.assertIs(args[1], faces)
        self.assertEqual(args[2], 0.001)  # Default min_base_height
        
        # Check that the result is what was returned by add_base
        self.assertEqual(len(result_vertices), 4)
        self.assertEqual(len(result_faces), 2)


if __name__ == '__main__':
    unittest.main()
