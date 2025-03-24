""".

Unit tests for the SDF exporter module.

These tests verify the functionality of the export_heightmap_to_sdf function.
"""
import os
import struct
import tempfile
import unittest

import numpy as np

from tmd.exporters.model.sdf import export_heightmap_to_sdf


class TestSDFExport(unittest.TestCase):
    """Test cases for SDF export function.."""

    def setUp(self):
        """Create temporary directory and test data.."""
        self.temp_dir = tempfile.TemporaryDirectory()

        # Create a small test height map
        size = 50
        x = np.linspace(-5, 5, size)
        y = np.linspace(-5, 5, size)
        X, Y = np.meshgrid(x, y)
        self.test_heightmap = np.sin(X) * np.cos(Y)

    def tearDown(self):
        """Clean up temporary directory.."""
        self.temp_dir.cleanup()

    def test_basic_export(self):
        """Test basic export with default parameters.."""
        output_file = os.path.join(self.temp_dir.name, "test_default.sdf")

        result = export_heightmap_to_sdf(self.test_heightmap, output_file)

        self.assertTrue(result)
        self.assertTrue(os.path.exists(output_file))
        self.assertGreater(os.path.getsize(output_file), 0)

        # Verify the file has the correct format by reading the header
        with open(output_file, "rb") as f:
            header = f.read(16)  # 4 bytes magic + 3*4 bytes
            magic, version, width, height = struct.unpack("4siii", header)

            self.assertEqual(magic, b"SDF1")
            self.assertEqual(version, 1)
            self.assertEqual(width, self.test_heightmap.shape[1])
            self.assertEqual(height, self.test_heightmap.shape[0])

    def test_scale_and_offset(self):
        """Test export with custom scale and offset.."""
        output_file = os.path.join(self.temp_dir.name, "test_scale_offset.sdf")
        scale = 2.5
        offset = 1.0

        result = export_heightmap_to_sdf(
            self.test_heightmap, output_file, scale=scale, offset=offset
        )

        self.assertTrue(result)

        # Calculate expected file size: header (16 bytes) + data (width * height * 4 bytes)
        expected_size = 16 + (self.test_heightmap.shape[0] * self.test_heightmap.shape[1] * 4)
        self.assertEqual(os.path.getsize(output_file), expected_size)

        # Read back and verify a sample value is scaled correctly
        with open(output_file, "rb") as f:
            f.read(16)  # Skip header

            # Read and check a sample value
            # First get an original value
            orig_value = self.test_heightmap[10, 10]
            expected_value = orig_value * scale + offset

            # Calculate position in file
            pos = 16 + ((10 * self.test_heightmap.shape[1] + 10) * 4)
            f.seek(pos)

            # Read and check the value
            binary_value = f.read(4)
            value = struct.unpack("f", binary_value)[0]

            # Allow for small floating point differences
            self.assertAlmostEqual(value, expected_value, places=5)

    def test_nonexistent_directory(self):
        """Test export to non-existent directory.."""
        new_dir = os.path.join(self.temp_dir.name, "new_folder", "subfolder")
        output_file = os.path.join(new_dir, "test.sdf")

        result = export_heightmap_to_sdf(self.test_heightmap, output_file)

        self.assertTrue(result)
        self.assertTrue(os.path.exists(output_file))

    def test_error_handling(self):
        """Test error handling with invalid input.."""
        # Test with invalid file path (using a directory instead of a file)
        with self.assertRaises(Exception):
            # This should trigger an exception when opening the file
            export_heightmap_to_sdf(
                self.test_heightmap, self.temp_dir.name  # This is a directory, not a file
            )


if __name__ == "__main__":
    unittest.main()
