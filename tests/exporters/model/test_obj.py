""".

Unit tests for OBJ export functionality.

Tests the conversion of height maps to OBJ files with various options.
"""

import os
import tempfile
import unittest
import numpy as np

from tmd.exporters.model.obj import (
    convert_heightmap_to_obj,
    convert_heightmap_to_obj_meshio
)
from tmd.utils.utils import create_sample_height_map


class TestOBJExport(unittest.TestCase):
    """Test cases for OBJ export functionality.."""
    
    def setUp(self):
        """Set up test data and temporary directory.."""
        # Create a temporary directory for test outputs
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create sample data - small for faster tests
        self.height_map = create_sample_height_map(width=20, height=20, pattern="peak")
        
    def tearDown(self):
        """Clean up temporary directory.."""
        self.temp_dir.cleanup()
    
    def _validate_obj_file(self, file_path, expected_exists=True, min_size=100):
        """Helper to validate OBJ output files.."""
        if expected_exists:
            self.assertTrue(os.path.exists(file_path), f"File {file_path} does not exist")
            self.assertGreater(os.path.getsize(file_path), min_size, 
                              f"File {file_path} is too small ({os.path.getsize(file_path)} bytes)")
            
            # Basic validation of file format
            with open(file_path, 'r') as f:
                content = f.read(200)
                # OBJ should start with "v" (vertex) or # (comment)
                self.assertTrue(content.strip().startswith("v") or content.strip().startswith("#"), 
                               f"OBJ file should start with vertex or comment, got: {content[:20]}")
                # Check for vertices and faces
                self.assertIn("v ", content[:1000] if len(content) > 1000 else content)
                with open(file_path, 'r') as f2:
                    full_content = f2.read()
                    self.assertIn("f ", full_content)
        else:
            self.assertFalse(os.path.exists(file_path), f"File {file_path} exists but should not")
    
    def test_basic_obj_export(self):
        """Test basic OBJ export functionality.."""
        output_path = os.path.join(self.temp_dir.name, "test_basic.obj")
        result = convert_heightmap_to_obj(
            self.height_map, 
            filename=output_path,
            z_scale=2.0
        )
        self.assertEqual(result, output_path)
        self._validate_obj_file(output_path)
    
    def test_obj_with_base(self):
        """Test OBJ export with base.."""
        output_path = os.path.join(self.temp_dir.name, "test_base.obj")
        result = convert_heightmap_to_obj(
            self.height_map, 
            filename=output_path,
            z_scale=2.0,
            base_height=1.5
        )
        self.assertEqual(result, output_path)
        self._validate_obj_file(output_path)
        
        # Get file size and compare with base vs without base
        no_base_path = os.path.join(self.temp_dir.name, "test_no_base.obj")
        convert_heightmap_to_obj(
            self.height_map, 
            filename=no_base_path,
            z_scale=2.0
        )
        
        # Base version should be larger (contains more vertices and faces)
        self.assertGreater(os.path.getsize(output_path), os.path.getsize(no_base_path))
    
    def test_obj_with_custom_dimensions(self):
        """Test OBJ export with custom physical dimensions.."""
        output_path = os.path.join(self.temp_dir.name, "test_dimensions.obj")
        result = convert_heightmap_to_obj(
            self.height_map, 
            filename=output_path,
            x_offset=10.0,
            y_offset=5.0,
            x_length=100.0,
            y_length=50.0,
            z_scale=3.0
        )
        self.assertEqual(result, output_path)
        self._validate_obj_file(output_path)
        
        # Check if the custom dimensions are applied
        with open(output_path, 'r') as f:
            content = f.read()
            vertices = [line for line in content.splitlines() if line.startswith('v ')]
            
            # First vertex should be at (x_offset, y_offset, _)
            first_vertex = vertices[0].split()
            self.assertAlmostEqual(float(first_vertex[1]), 10.0)  # x_offset
            self.assertAlmostEqual(float(first_vertex[2]), 5.0)   # y_offset
    
    def test_meshio_obj_export(self):
        """Test OBJ export using meshio.."""
        output_path = os.path.join(self.temp_dir.name, "test_meshio.obj")
        result = convert_heightmap_to_obj_meshio(
            self.height_map, 
            filename=output_path,
            z_scale=2.0
        )
        self.assertEqual(result, output_path)
        self._validate_obj_file(output_path)
    
    def test_error_handling(self):
        """Test error handling for invalid inputs.."""
        # Test with height map that's too small
        small_map = np.zeros((1, 1))
        result = convert_heightmap_to_obj(small_map, os.path.join(self.temp_dir.name, "small.obj"))
        self.assertIsNone(result)
        
        # Test with invalid directory - use a temp path that we can create but won't have write access
        if os.name == 'nt':  # Windows
            invalid_dir = "C:\\Windows\\System32\\restricted_area\\file.obj"
        else:  # Unix/Linux
            invalid_dir = "/root/restricted_area/file.obj"
            
        try:
            result = convert_heightmap_to_obj(self.height_map, invalid_dir)
            self.assertIsNone(result)
        except (PermissionError, OSError):
            # This is also acceptable - either the converter catches the error and returns None
            # or the error is propagated up
            pass


if __name__ == "__main__":
    unittest.main()
