""".

Unit tests for Three.js export functionality.

Tests the conversion of height maps to Three.js compatible formats.
"""

import os
import tempfile
import unittest
import json
import numpy as np
import zlib
import base64
from unittest.mock import patch

from tmd.exporters.model import threejs
from tmd.exporters.model.threejs import (
    convert_heightmap_to_threejs
)
from tmd.utils.utils import create_sample_height_map

class TestThreeJSExport(unittest.TestCase):
    """Test cases for Three.js export functionality.."""
    
    def setUp(self):
        """Set up test fixtures.."""
        self.temp_dir = tempfile.mkdtemp(prefix="tmd_test_threejs_")
        self.height_map = create_sample_height_map(32, 32, 'dome')
    
    def tearDown(self):
        """Clean up test fixtures.."""
        # Recursively delete the temp directory and all files
        for root, dirs, files in os.walk(self.temp_dir, topdown=False):
            for file in files:
                os.unlink(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))
        os.rmdir(self.temp_dir)
    
    def _validate_threejs_json(self, file_path, expected_exists=True, min_size=100):
        """Helper to validate Three.js JSON output files.."""
        if expected_exists:
            self.assertTrue(os.path.exists(file_path), f"File {file_path} does not exist")
            self.assertGreater(os.path.getsize(file_path), min_size,
                              f"File {file_path} is too small ({os.path.getsize(file_path)} bytes)")

            # Basic validation of file format
            with open(file_path, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    # Check if it's a compressed file
                    f.seek(0)
                    content = f.read()
                    try:
                        # Try to decompress the content
                        decoded = base64.b64decode(content)
                        decompressed = zlib.decompress(decoded)
                        data = json.loads(decompressed)
                    except Exception:
                        self.fail(f"File {file_path} is not valid JSON or compressed JSON")
                
                # Validate required fields
                self.assertIn("metadata", data)
                self.assertIn("geometries", data)
                self.assertIn("materials", data)
                self.assertIn("object", data)
        else:
            self.assertFalse(os.path.exists(file_path), f"File {file_path} exists but should not")
    
    def test_basic_export(self):
        """Test basic Three.js export with default parameters.."""
        output_path = os.path.join(self.temp_dir, "test_default.json")
        result = convert_heightmap_to_threejs(
            self.height_map,
            filename=output_path
        )
        self.assertEqual(result, output_path)
        self._validate_threejs_json(output_path)
    
    def test_threejs_with_texture(self):
        """Test Three.js export with texture.."""
        output_path = os.path.join(self.temp_dir, "test_texture.json")

        # Patch the _generate_threejs_texture function to return correct tuple
        original_func = threejs._generate_threejs_texture
        try:
            def patched_func(*args, **kwargs):
                texture_data, image_uuid = original_func(*args, **kwargs)
                return texture_data, image_uuid

            threejs._generate_threejs_texture = patched_func

            result = convert_heightmap_to_threejs(
                self.height_map,
                filename=output_path,
                z_scale=2.0,
                add_texture=True,
                texture_resolution=(256, 256)
            )

            self.assertEqual(result, output_path)
            self._validate_threejs_json(output_path)
            
            # Check for texture in the data
            with open(output_path, 'r') as f:
                data = json.load(f)
                self.assertIn("textures", data)
        
        finally:
            # Restore original function
            threejs._generate_threejs_texture = original_func
    
    def test_threejs_with_wireframe(self):
        """Test Three.js export with wireframe.."""
        output_path = os.path.join(self.temp_dir, "test_wireframe.json")
        result = convert_heightmap_to_threejs(
            self.height_map,
            filename=output_path,
            z_scale=2.0,
            add_wireframe=True
        )
        self.assertEqual(result, output_path)
        self._validate_threejs_json(output_path)

        # Verify wireframe in the Three.js JSON
        with open(output_path, 'r') as f:
            data = json.load(f)
            # Should have at least two geometries (one for mesh, one for wireframe)
            self.assertGreaterEqual(len(data["geometries"]), 2)
    
    def test_threejs_compressed(self):
        """Test Three.js export with compression.."""
        output_path = os.path.join(self.temp_dir, "test_compressed.json")
        result = convert_heightmap_to_threejs(
            self.height_map,
            filename=output_path,
            z_scale=2.0,
            compress=True
        )
        self.assertEqual(result, output_path)
        
        # For compressed files, we need to manually decode and validate
        self.assertTrue(os.path.exists(output_path))
        self.assertGreater(os.path.getsize(output_path), 10)
        
        # Try to decompress and validate the content
        with open(output_path, 'r') as f:
            content = f.read()
            try:
                decoded = base64.b64decode(content)
                decompressed = zlib.decompress(decoded)
                data = json.loads(decompressed)
                
                # Validate required fields
                self.assertIn("metadata", data)
                self.assertIn("geometries", data)
                self.assertIn("materials", data)
                self.assertIn("object", data)
            except Exception as e:
                self.fail(f"Failed to decompress or validate JSON: {e}")
    
    @patch('tmd.exporters.model.gltf.convert_heightmap_to_gltf')
    def test_use_gltf(self, mock_convert_gltf):
        """Test Three.js export with glTF mode.."""
        # Mock the gltf conversion to return the expected path
        output_path = os.path.join(self.temp_dir, "test_gltf.gltf")
        mock_convert_gltf.return_value = output_path
        
        result = convert_heightmap_to_threejs(
            self.height_map,
            filename=output_path,
            use_gltf=True
        )
        
        # Verify that conversion was successful
        self.assertEqual(result, output_path)
        
        # Verify that the gltf converter was called with the right parameters
        mock_convert_gltf.assert_called_once()
        args, kwargs = mock_convert_gltf.call_args
        self.assertEqual(kwargs['filename'], output_path)
        self.assertIs(kwargs['height_map'], self.height_map)

if __name__ == "__main__":
    unittest.main()
