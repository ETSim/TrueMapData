"""
Unit tests for the STL, OBJ, and PLY export modules.

These tests verify the functionality of exporting height maps using both
the custom implementations and the Meshio-based implementations.
"""

import os
import re
import struct
import tempfile
import unittest
import numpy as np

# Adjust the import path as needed.
from tmd.exporters.model import (
    convert_heightmap_to_stl,
    convert_heightmap_to_obj,
    convert_heightmap_to_ply,
    convert_heightmap_to_stl_meshio,
    convert_heightmap_to_obj_meshio,
    convert_heightmap_to_ply_meshio,
)

# -------------- Custom Exporter Tests --------------

class TestSTLExport(unittest.TestCase):
    """Test cases for custom STL export functionality."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name
        
        # Simple height map for testing (3x3)
        self.small_height_map = np.array([
            [0.0, 0.5, 1.0],
            [0.0, 0.5, 1.0],
            [0.0, 0.5, 1.0]
        ])
        
        # Gradient height map (10x10)
        size = 10
        x = np.linspace(0, 1, size)
        y = np.linspace(0, 1, size)
        X, Y = np.meshgrid(x, y)
        self.gradient_height_map = X + Y

        # Peak height map (20x20)
        self.peak_height_map = np.zeros((20, 20))
        for i in range(20):
            for j in range(20):
                dx = (i - 10) / 5
                dy = (j - 10) / 5
                self.peak_height_map[i, j] = np.exp(-(dx**2 + dy**2))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_convert_small_height_map_ascii(self):
        """Test converting a small height map to ASCII STL."""
        output_file = os.path.join(self.output_dir, "small_ascii.stl")
        convert_heightmap_to_stl(
            height_map=self.small_height_map,
            filename=output_file,
            x_length=1.0,
            y_length=1.0,
            z_scale=1.0,
            ascii=True
        )
        self.assertTrue(os.path.exists(output_file))
        # Check file size and ASCII markers.
        file_size = os.path.getsize(output_file)
        self.assertGreater(file_size, 100)
        with open(output_file, 'r') as f:
            content = f.read()
            self.assertTrue(content.startswith("solid displacement"))
            self.assertTrue(content.endswith("endsolid displacement\n"))
            # Verify expected number of triangles.
            rows, cols = self.small_height_map.shape
            expected_triangles = 2 * (rows - 1) * (cols - 1)
            facet_count = content.count("facet normal")
            self.assertEqual(facet_count, expected_triangles)

    def test_convert_small_height_map_binary(self):
        """Test converting a small height map to binary STL."""
        output_file = os.path.join(self.output_dir, "small_binary.stl")
        convert_heightmap_to_stl(
            height_map=self.small_height_map,
            filename=output_file,
            x_length=1.0,
            y_length=1.0,
            z_scale=1.0,
            ascii=False
        )
        self.assertTrue(os.path.exists(output_file))
        with open(output_file, 'rb') as f:
            header = f.read(80)
            self.assertEqual(header[:34], b"TMD Processor Generated Binary STL")
            triangle_count_bytes = f.read(4)
            triangle_count = struct.unpack("<I", triangle_count_bytes)[0]
            expected_triangles = 2 * (self.small_height_map.shape[0]-1) * (self.small_height_map.shape[1]-1)
            self.assertEqual(triangle_count, expected_triangles)
            expected_file_size = 80 + 4 + (50 * triangle_count)
            self.assertEqual(os.path.getsize(output_file), expected_file_size)

    def test_gradient_height_map(self):
        """Test converting a gradient height map to ASCII STL."""
        output_file = os.path.join(self.output_dir, "gradient.stl")
        convert_heightmap_to_stl(
            height_map=self.gradient_height_map,
            filename=output_file,
            ascii=True
        )
        self.assertTrue(os.path.exists(output_file))
        self.assertGreater(os.path.getsize(output_file), 1000)
        rows, cols = self.gradient_height_map.shape
        expected_triangles = 2 * (rows - 1) * (cols - 1)
        with open(output_file, 'r') as f:
            content = f.read()
            facet_count = content.count("facet normal")
            self.assertEqual(facet_count, expected_triangles)

    def test_heightmap_z_scaling(self):
        """Test that z_scale affects the output vertex Z-values."""
        output_file1 = os.path.join(self.output_dir, "peak_z1.stl")
        output_file2 = os.path.join(self.output_dir, "peak_z5.stl")
        convert_heightmap_to_stl(
            height_map=self.peak_height_map,
            filename=output_file1,
            z_scale=1.0,
            ascii=True
        )
        convert_heightmap_to_stl(
            height_map=self.peak_height_map,
            filename=output_file2,
            z_scale=5.0,
            ascii=True
        )
        self.assertTrue(os.path.exists(output_file1))
        self.assertTrue(os.path.exists(output_file2))
        with open(output_file1, 'r') as f1, open(output_file2, 'r') as f2:
            content1 = f1.read()
            content2 = f2.read()
            # Extract the first vertex z-value from each file.
            match1 = re.search(r'vertex\s+\S+\s+\S+\s+(\S+)', content1)
            match2 = re.search(r'vertex\s+\S+\s+\S+\s+(\S+)', content2)
            self.assertIsNotNone(match1)
            self.assertIsNotNone(match2)
            z1 = float(match1.group(1))
            z2 = float(match2.group(1))
            if z1 != 0:
                self.assertGreater(abs(z2), abs(z1))

    def test_custom_physical_dimensions(self):
        """Test that specifying custom physical dimensions scales vertices accordingly."""
        output_default = os.path.join(self.output_dir, "dim_default.stl")
        output_custom = os.path.join(self.output_dir, "dim_custom.stl")
        convert_heightmap_to_stl(
            height_map=self.small_height_map,
            filename=output_default,
            ascii=True
        )
        convert_heightmap_to_stl(
            height_map=self.small_height_map,
            filename=output_custom,
            x_length=10.0,
            y_length=10.0,
            ascii=True
        )
        self.assertTrue(os.path.exists(output_default))
        self.assertTrue(os.path.exists(output_custom))
        with open(output_default, 'r') as f1, open(output_custom, 'r') as f2:
            content1 = f1.read()
            content2 = f2.read()
            # Extract first non-zero vertex x coordinate.
            match1 = re.search(r'vertex\s+(\S+)\s+(\S+)', content1)
            match2 = re.search(r'vertex\s+(\S+)\s+(\S+)', content2)
            self.assertIsNotNone(match1)
            self.assertIsNotNone(match2)
            x1 = float(match1.group(1))
            x2 = float(match2.group(1))
            # Expect roughly 10x scaling in the x-direction.
            self.assertAlmostEqual(x2 / (x1 if x1 != 0 else 1), 10.0, delta=0.2)

    def test_offset_application(self):
        """Test that x/y offsets correctly shift the vertices."""
        file_no_offset = os.path.join(self.output_dir, "no_offset.stl")
        file_with_offset = os.path.join(self.output_dir, "with_offset.stl")
        convert_heightmap_to_stl(
            height_map=self.small_height_map,
            filename=file_no_offset,
            ascii=True
        )
        convert_heightmap_to_stl(
            height_map=self.small_height_map,
            filename=file_with_offset,
            x_offset=10.0,
            y_offset=20.0,
            ascii=True
        )
        self.assertTrue(os.path.exists(file_no_offset))
        self.assertTrue(os.path.exists(file_with_offset))
        with open(file_no_offset, 'r') as f1, open(file_with_offset, 'r') as f2:
            c1 = f1.read()
            c2 = f2.read()
            match1 = re.search(r'vertex\s+(\S+)\s+(\S+)', c1)
            match2 = re.search(r'vertex\s+(\S+)\s+(\S+)', c2)
            self.assertIsNotNone(match1)
            self.assertIsNotNone(match2)
            x1, y1 = float(match1.group(1)), float(match1.group(2))
            x2, y2 = float(match2.group(1)), float(match2.group(2))
            self.assertAlmostEqual(x2, x1 + 10.0, places=5)
            self.assertAlmostEqual(y2, y1 + 20.0, places=5)

    def test_tiny_heightmap(self):
        """Test that a 1x1 height map does not create an STL file."""
        tiny_map = np.array([[1.0]])
        output_file = os.path.join(self.output_dir, "tiny.stl")
        convert_heightmap_to_stl(
            height_map=tiny_map,
            filename=output_file,
            ascii=True
        )
        self.assertFalse(os.path.exists(output_file))

    def test_normal_vector_calculation(self):
        """Test that computed normals are normalized."""
        simple_map = np.array([
            [0.0, 0.0],
            [0.0, 1.0]
        ])
        output_file = os.path.join(self.output_dir, "simple_normals.stl")
        convert_heightmap_to_stl(
            height_map=simple_map,
            filename=output_file,
            ascii=True
        )
        self.assertTrue(os.path.exists(output_file))
        with open(output_file, 'r') as f:
            content = f.read()
            match = re.search(r'facet normal\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)', content)
            self.assertIsNotNone(match, "No normal vector found")
            nx, ny, nz = float(match.group(1)), float(match.group(2)), float(match.group(3))
            norm = np.sqrt(nx**2 + ny**2 + nz**2)
            self.assertAlmostEqual(norm, 1.0, places=5)

    def test_binary_file_structure(self):
        """Test detailed binary structure of the exported STL."""
        output_file = os.path.join(self.output_dir, "binary_structure.stl")
        convert_heightmap_to_stl(
            height_map=self.small_height_map,
            filename=output_file,
            ascii=False
        )
        self.assertTrue(os.path.exists(output_file))
        with open(output_file, 'rb') as f:
            header = f.read(80)
            triangle_count_bytes = f.read(4)
            triangle_count = struct.unpack("<I", triangle_count_bytes)[0]
            rows, cols = self.small_height_map.shape
            expected_triangles = 2 * (rows - 1) * (cols - 1)
            self.assertEqual(triangle_count, expected_triangles)
            # Read first triangle (50 bytes)
            normal_bytes = f.read(12)
            nx, ny, nz = struct.unpack("<fff", normal_bytes)
            norm = np.sqrt(nx**2 + ny**2 + nz**2)
            self.assertAlmostEqual(norm, 1.0, places=5)
            expected_size = 80 + 4 + (50 * triangle_count)
            self.assertEqual(os.path.getsize(output_file), expected_size)


# -------------- Meshio-based Exporter Tests --------------

class TestMeshioExports(unittest.TestCase):
    """Test cases for Meshio-based export functions."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name
        size = 10
        x = np.linspace(0, 1, size)
        y = np.linspace(0, 1, size)
        X, Y = np.meshgrid(x, y)
        self.gradient_height_map = X + Y

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_meshio_stl_export(self):
        output_file = os.path.join(self.output_dir, "meshio_ascii.stl")
        convert_heightmap_to_stl_meshio(
            height_map=self.gradient_height_map,
            filename=output_file,
            x_length=10,
            y_length=10,
            z_scale=2,
            ascii=True
        )
        self.assertTrue(os.path.exists(output_file))
        with open(output_file, 'r') as f:
            content = f.read()
            self.assertTrue(content.startswith("solid"))
            expected_triangles = 2 * (self.gradient_height_map.shape[0]-1) * (self.gradient_height_map.shape[1]-1)
            facet_count = content.count("facet normal")
            self.assertEqual(facet_count, expected_triangles)

    def test_meshio_obj_export(self):
        output_file = os.path.join(self.output_dir, "meshio_output.obj")
        convert_heightmap_to_obj_meshio(
            height_map=self.gradient_height_map,
            filename=output_file,
            x_length=10,
            y_length=10,
            z_scale=2
        )
        self.assertTrue(os.path.exists(output_file))
        with open(output_file, 'r') as f:
            content = f.read()
            self.assertTrue(content.startswith("v "))
            self.assertIn("f ", content)

    def test_meshio_ply_export(self):
        output_file = os.path.join(self.output_dir, "meshio_output.ply")
        convert_heightmap_to_ply_meshio(
            height_map=self.gradient_height_map,
            filename=output_file,
            x_length=10,
            y_length=10,
            z_scale=2
        )
        self.assertTrue(os.path.exists(output_file))
        
        # Try to read the file - if it's binary, open in binary mode
        try:
            with open(output_file, 'r') as f:
                content = f.read()
                self.assertTrue(content.startswith("ply"))
                self.assertIn("element vertex", content)
                self.assertIn("element face", content)
        except UnicodeDecodeError:
            # If a UnicodeDecodeError occurs, it's likely a binary PLY file
            with open(output_file, 'rb') as f:
                # Just check that we can read the file and it begins with 'ply'
                content = f.read(100)  # Read just the first part of the file
                self.assertTrue(content.startswith(b"ply"))
                # Binary PLY files should have "format binary" in the header
                self.assertTrue(b"format binary" in content or b"ascii" in content)


if __name__ == "__main__":
    unittest.main()
