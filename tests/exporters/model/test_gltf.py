""".

Unit tests for glTF/GLB export functionality.

Tests the conversion of height maps to glTF and GLB files with various options.
"""

import os
import tempfile
import unittest
import json
import numpy as np

from tmd.exporters.model.gltf import (
    convert_heightmap_to_gltf,
    convert_heightmap_to_glb
)
from tmd.utils.utils import create_sample_height_map


class TestGLTFExport(unittest.TestCase):
    """Test cases for glTF/GLB export functionality.."""
    
    def setUp(self):
        """Set up test data and temporary directory.."""
        # Create a temporary directory for test outputs
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create sample data - small for faster tests
        self.height_map = create_sample_height_map(width=20, height=20, pattern="peak")
        
    def tearDown(self):
        """Clean up temporary directory.."""
        self.temp_dir.cleanup()
    
    def _validate_gltf_file(self, file_path, expected_exists=True, min_size=100, is_glb=False):
        """Helper to validate glTF/GLB output files.."""
        if expected_exists:
            self.assertTrue(os.path.exists(file_path), f"File {file_path} does not exist")
            self.assertGreater(os.path.getsize(file_path), min_size, 
                              f"File {file_path} is too small ({os.path.getsize(file_path)} bytes)")
            
            # Basic validation of file format
            if is_glb:
                # Check GLB magic bytes
                with open(file_path, 'rb') as f:
                    header = f.read(4)
                    self.assertEqual(header, b'glTF', "GLB file should start with 'glTF' magic bytes")
            else:
                # Check glTF JSON structure
                with open(file_path, 'r') as f:
                    gltf_data = json.load(f)
                    
                    # Check for required elements in the glTF schema
                    self.assertIn("asset", gltf_data, "Missing 'asset' in glTF")
                    self.assertIn("version", gltf_data["asset"], "Missing asset.version in glTF")
                    self.assertIn("scenes", gltf_data, "Missing 'scenes' in glTF")
                    self.assertIn("nodes", gltf_data, "Missing 'nodes' in glTF")
                    self.assertIn("meshes", gltf_data, "Missing 'meshes' in glTF")
                    self.assertIn("buffers", gltf_data, "Missing 'buffers' in glTF")
                    self.assertIn("bufferViews", gltf_data, "Missing 'bufferViews' in glTF")
                    self.assertIn("accessors", gltf_data, "Missing 'accessors' in glTF")
        else:
            self.assertFalse(os.path.exists(file_path), f"File {file_path} exists but should not")
    
    def test_basic_gltf_export(self):
        """Test basic glTF export functionality.."""
        output_path = os.path.join(self.temp_dir.name, "test_basic.gltf")
        result = convert_heightmap_to_gltf(
            self.height_map, 
            filename=output_path,
            z_scale=2.0
        )
        self.assertEqual(result, output_path)
        self._validate_gltf_file(output_path)
    
    def test_basic_glb_export(self):
        """Test basic GLB export functionality.."""
        output_path = os.path.join(self.temp_dir.name, "test_basic.glb")
        result = convert_heightmap_to_glb(
            self.height_map, 
            filename=output_path,
            z_scale=2.0
        )
        self.assertEqual(result, output_path)
        self._validate_gltf_file(output_path, is_glb=True)
    
    def test_gltf_with_base(self):
        """Test glTF export with base.."""
        output_path = os.path.join(self.temp_dir.name, "test_base.gltf")
        result = convert_heightmap_to_gltf(
            self.height_map, 
            filename=output_path,
            z_scale=2.0,
            base_height=1.5
        )
        self.assertEqual(result, output_path)
        self._validate_gltf_file(output_path)
        
        # Get file size and compare with base vs without base
        no_base_path = os.path.join(self.temp_dir.name, "test_no_base.gltf")
        convert_heightmap_to_gltf(
            self.height_map, 
            filename=no_base_path,
            z_scale=2.0
        )
        
        # Base version should be larger (contains more vertices and faces)
        self.assertGreater(os.path.getsize(output_path), os.path.getsize(no_base_path))
    
    def test_gltf_with_texture(self):
        """Test glTF export with texture.."""
        output_path = os.path.join(self.temp_dir.name, "test_texture.gltf")
        result = convert_heightmap_to_gltf(
            self.height_map, 
            filename=output_path,
            z_scale=2.0,
            add_texture=True,
            texture_resolution=256
        )
        self.assertEqual(result, output_path)
        self._validate_gltf_file(output_path)
        
        # Verify texture file was created
        texture_path = os.path.splitext(output_path)[0] + "_texture.png"
        self.assertTrue(os.path.exists(texture_path), "Texture file was not created")
        
        # Check that texture is referenced in the glTF file
        with open(output_path, 'r') as f:
            gltf_data = json.load(f)
            self.assertIn("textures", gltf_data, "Missing 'textures' in glTF")
            self.assertIn("images", gltf_data, "Missing 'images' in glTF")
            self.assertIn("materials", gltf_data, "Missing 'materials' in glTF")
    
    def test_glb_with_texture(self):
        """Test GLB export with texture.."""
        output_path = os.path.join(self.temp_dir.name, "test_texture.glb")
        result = convert_heightmap_to_glb(
            self.height_map, 
            filename=output_path,
            z_scale=2.0,
            add_texture=True,
            texture_resolution=256
        )
        self.assertEqual(result, output_path)
        self._validate_gltf_file(output_path, is_glb=True)
        
        # Verify texture file was created
        texture_path = os.path.splitext(output_path)[0] + "_texture.png"
        self.assertTrue(os.path.exists(texture_path), "Texture file was not created")
    
    def test_gltf_extension_correction(self):
        """Test that GLB filenames are automatically corrected when using convert_heightmap_to_glb.."""
        # Don't include .glb extension in the filename
        output_path = os.path.join(self.temp_dir.name, "test_no_extension")
        result = convert_heightmap_to_glb(
            self.height_map, 
            filename=output_path
        )
        # Result should have .glb extension added
        self.assertEqual(result, output_path + ".glb")
        self._validate_gltf_file(output_path + ".glb", is_glb=True)
    
    def test_error_handling(self):
        """Test error handling for invalid inputs.."""
        # Test with height map that's too small
        small_map = np.zeros((1, 1))
        result = convert_heightmap_to_gltf(small_map, os.path.join(self.temp_dir.name, "small.gltf"))
        self.assertIsNone(result)
        
        # Test with invalid directory
        invalid_dir = "/path/that/does/not/exist/file.gltf"
        result = convert_heightmap_to_gltf(self.height_map, invalid_dir)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
