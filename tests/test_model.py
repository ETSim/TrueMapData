"""
Unit tests for model exporter module.

Tests the functionality for converting height maps to various 3D model formats.
"""
import os
import tempfile
import unittest
import numpy as np

from tmd.exporters.model import (
    convert_heightmap_to_stl,
    convert_heightmap_to_obj,
    convert_heightmap_to_ply,
    convert_heightmap_to_stl_meshio,
    convert_heightmap_to_obj_meshio,
    convert_heightmap_to_ply_meshio
)
from tmd.utils.utils import create_sample_height_map


class TestModelExporter(unittest.TestCase):
    """Test cases for model export functionality."""
    
    def setUp(self):
        """Set up test data and temporary directory."""
        # Create a temporary directory for test outputs
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create sample data
        self.height_map = create_sample_height_map(width=10, height=10, pattern="peak")
        
    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()
    
    def _validate_file(self, file_path, expected_exists=True, min_size=100):
        """Helper to validate output files."""
        if expected_exists:
            self.assertTrue(os.path.exists(file_path), f"File {file_path} does not exist")
            self.assertGreater(os.path.getsize(file_path), min_size, 
                              f"File {file_path} is too small ({os.path.getsize(file_path)} bytes)")
        else:
            self.assertFalse(os.path.exists(file_path), f"File {file_path} exists but should not")
    
    def test_stl_export(self):
        """Test STL export with various options."""
        # Test ASCII STL
        ascii_path = os.path.join(self.temp_dir.name, "test_ascii.stl")
        result = convert_heightmap_to_stl(
            self.height_map, 
            filename=ascii_path,
            z_scale=2.0,
            ascii=True
        )
        self.assertEqual(result, ascii_path)
        self._validate_file(ascii_path)
        
        # Test binary STL
        binary_path = os.path.join(self.temp_dir.name, "test_binary.stl")
        result = convert_heightmap_to_stl(
            self.height_map, 
            filename=binary_path,
            z_scale=2.0,
            ascii=False
        )
        self.assertEqual(result, binary_path)
        self._validate_file(binary_path)
        
        # Test with custom dimensions
        custom_path = os.path.join(self.temp_dir.name, "test_custom.stl")
        result = convert_heightmap_to_stl(
            self.height_map, 
            filename=custom_path,
            x_offset=5.0,
            y_offset=10.0,
            x_length=20.0,
            y_length=30.0,
            z_scale=5.0
        )
        self.assertEqual(result, custom_path)
        self._validate_file(custom_path)
        
        # Test with base
        base_path = os.path.join(self.temp_dir.name, "test_base.stl")
        result = convert_heightmap_to_stl(
            self.height_map, 
            filename=base_path,
            base_height=2.0
        )
        self.assertEqual(result, base_path)
        self._validate_file(base_path)
    
    def test_obj_export(self):
        """Test OBJ export with various options."""
        # Basic test
        obj_path = os.path.join(self.temp_dir.name, "test.obj")
        result = convert_heightmap_to_obj(
            self.height_map, 
            filename=obj_path,
            z_scale=2.0
        )
        self.assertEqual(result, obj_path)
        self._validate_file(obj_path)
        
        # Test with base
        base_path = os.path.join(self.temp_dir.name, "test_base.obj")
        result = convert_heightmap_to_obj(
            self.height_map, 
            filename=base_path,
            base_height=2.0
        )
        self.assertEqual(result, base_path)
        self._validate_file(base_path)
    
    def test_ply_export(self):
        """Test PLY export with various options."""
        # Basic test
        ply_path = os.path.join(self.temp_dir.name, "test.ply")
        result = convert_heightmap_to_ply(
            self.height_map, 
            filename=ply_path,
            z_scale=2.0
        )
        self.assertEqual(result, ply_path)
        self._validate_file(ply_path)
        
        # Test with base
        base_path = os.path.join(self.temp_dir.name, "test_base.ply")
        result = convert_heightmap_to_ply(
            self.height_map, 
            filename=base_path,
            base_height=2.0
        )
        self.assertEqual(result, base_path)
        self._validate_file(base_path)
    
    def test_meshio_exports(self):
        """Test exports using meshio."""
        # STL
        stl_path = os.path.join(self.temp_dir.name, "meshio_test.stl")
        result = convert_heightmap_to_stl_meshio(
            self.height_map, 
            filename=stl_path,
            z_scale=2.0
        )
        self.assertEqual(result, stl_path)
        self._validate_file(stl_path)
        
        # OBJ
        obj_path = os.path.join(self.temp_dir.name, "meshio_test.obj")
        result = convert_heightmap_to_obj_meshio(
            self.height_map, 
            filename=obj_path,
            z_scale=2.0
        )
        self.assertEqual(result, obj_path)
        self._validate_file(obj_path)
        
        # PLY
        ply_path = os.path.join(self.temp_dir.name, "meshio_test.ply")
        result = convert_heightmap_to_ply_meshio(
            self.height_map, 
            filename=ply_path,
            z_scale=2.0
        )
        self.assertEqual(result, ply_path)
        self._validate_file(ply_path)
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test with height map that's too small
        small_map = np.zeros((1, 1))
        result = convert_heightmap_to_stl(small_map, os.path.join(self.temp_dir.name, "small.stl"))
        self.assertIsNone(result)
        
        # Test with invalid directory
        invalid_dir = "/path/that/does/not/exist/file.stl"
        # This should not raise an exception but return None
        result = convert_heightmap_to_stl(self.height_map, invalid_dir)
        self.assertIsNone(result)

        # Test with path that causes an error during processing
        # Use a truly small map that's guaranteed to fail even after directory creation
        tiny_map = np.zeros((1, 1))
        nested_path = os.path.join(self.temp_dir.name, "nonexistent_subdir", "deeply", "nested", "file.stl")
        result = convert_heightmap_to_stl(tiny_map, nested_path)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
