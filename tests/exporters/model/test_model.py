""".

Unit tests for model export functionality at the module level.

Tests the high-level import interface for model exporters.
"""

import os
import tempfile
import unittest

import numpy as np

# Import from the model module instead of individual modules
from tmd.exporters.model import (
    convert_heightmap_to_glb,
    convert_heightmap_to_gltf,
    convert_heightmap_to_obj,
    convert_heightmap_to_ply,
    convert_heightmap_to_stl,
    convert_heightmap_to_threejs,
    convert_heightmap_to_usdz,
)
from tmd.utils.utils import create_sample_height_map


class TestModelExportModule(unittest.TestCase):
    """Test cases for model export module-level interface.."""

    def setUp(self):
        """Set up test data and temporary directory.."""
        # Create a temporary directory for test outputs
        self.temp_dir = tempfile.TemporaryDirectory()

        # Create sample data - small for faster tests
        self.height_map = create_sample_height_map(width=20, height=20, pattern="peak")

    def tearDown(self):
        """Clean up temporary directory.."""
        self.temp_dir.cleanup()

    def test_stl_export(self):
        """Test STL export via module interface.."""
        output_path = os.path.join(self.temp_dir.name, "test_module.stl")
        result = convert_heightmap_to_stl(self.height_map, filename=output_path, z_scale=2.0)
        self.assertTrue(os.path.exists(output_path))
        self.assertEqual(result, output_path)

    def test_obj_export(self):
        """Test OBJ export via module interface.."""
        output_path = os.path.join(self.temp_dir.name, "test_module.obj")
        result = convert_heightmap_to_obj(self.height_map, filename=output_path, z_scale=2.0)
        self.assertTrue(os.path.exists(output_path))
        self.assertEqual(result, output_path)

    def test_ply_export(self):
        """Test PLY export via module interface.."""
        output_path = os.path.join(self.temp_dir.name, "test_module.ply")
        result = convert_heightmap_to_ply(self.height_map, filename=output_path, z_scale=2.0)
        self.assertTrue(os.path.exists(output_path))
        self.assertEqual(result, output_path)

    def test_gltf_export(self):
        """Test glTF export via module interface.."""
        output_path = os.path.join(self.temp_dir.name, "test_module.gltf")
        result = convert_heightmap_to_gltf(self.height_map, filename=output_path, z_scale=2.0)
        self.assertTrue(os.path.exists(output_path))
        self.assertEqual(result, output_path)

    def test_glb_export(self):
        """Test GLB export via module interface.."""
        output_path = os.path.join(self.temp_dir.name, "test_module.glb")
        result = convert_heightmap_to_glb(self.height_map, filename=output_path, z_scale=2.0)
        self.assertTrue(os.path.exists(output_path))
        self.assertEqual(result, output_path)

    def test_threejs_export(self):
        """Test Three.js export via module interface.."""
        output_path = os.path.join(self.temp_dir.name, "test_module.json")
        result = convert_heightmap_to_threejs(self.height_map, filename=output_path, z_scale=2.0)
        self.assertTrue(os.path.exists(output_path))
        self.assertEqual(result, output_path)

    def test_usdz_export(self):
        """Test USDZ export via module interface.."""
        output_path = os.path.join(self.temp_dir.name, "test_module.usdz")
        result = convert_heightmap_to_usdz(self.height_map, filename=output_path, z_scale=2.0)
        self.assertTrue(os.path.exists(output_path))
        self.assertEqual(result, output_path)


if __name__ == "__main__":
    unittest.main()
