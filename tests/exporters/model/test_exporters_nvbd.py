""".

Unit tests for the NVBD exporter module.

These tests verify the functionality of the export_heightmap_to_nvbd function.
"""
import os
import struct
import tempfile
import unittest

import numpy as np

from tmd.exporters.model.nvbd import export_heightmap_to_nvbd


class TestNVBDExport(unittest.TestCase):
    """Test cases for NVBD export function.."""

    def setUp(self):
        """Create temporary directory and test data.."""
        self.temp_dir = tempfile.TemporaryDirectory()

        # Create a small test height map
        size = 32
        x = np.linspace(-5, 5, size)
        y = np.linspace(-5, 5, size)
        X, Y = np.meshgrid(x, y)
        self.test_heightmap = np.sin(X) * np.cos(Y)

    def tearDown(self):
        """Clean up temporary directory.."""
        self.temp_dir.cleanup()

    def test_basic_export(self):
        """Test basic export with default parameters.."""
        output_file = os.path.join(self.temp_dir.name, "test_default.nvbd")

        result = export_heightmap_to_nvbd(self.test_heightmap, output_file)

        self.assertTrue(result)
        self.assertTrue(os.path.exists(output_file))
        self.assertGreater(os.path.getsize(output_file), 0)

        # Verify the file has the correct format by reading the header
        with open(output_file, "rb") as f:
            header = f.read(20)  # 4 bytes + 4*4 bytes
            magic, version, width, height, chunk_size = struct.unpack("4siiii", header)

            self.assertEqual(magic, b"NVBD")
            self.assertEqual(version, 1)
            self.assertEqual(width, self.test_heightmap.shape[1])
            self.assertEqual(height, self.test_heightmap.shape[0])
            self.assertEqual(chunk_size, 16)  # Default chunk size

    def test_custom_chunk_size(self):
        """Test export with custom chunk size.."""
        output_file = os.path.join(self.temp_dir.name, "test_chunk_size.nvbd")
        custom_chunk_size = 8

        result = export_heightmap_to_nvbd(
            self.test_heightmap, output_file, chunk_size=custom_chunk_size
        )

        self.assertTrue(result)

        # Verify the chunk size was correctly set
        with open(output_file, "rb") as f:
            header = f.read(20)
            magic, version, width, height, chunk_size = struct.unpack("4siiii", header)
            self.assertEqual(chunk_size, custom_chunk_size)

    def test_scale_and_offset(self):
        """Test export with custom scale and offset.."""
        output_file = os.path.join(self.temp_dir.name, "test_scale_offset.nvbd")
        scale = 2.5
        offset = 1.0

        result = export_heightmap_to_nvbd(
            self.test_heightmap,
            output_file,
            scale=scale,
            offset=offset,
            chunk_size=4,  # Small for faster test
        )

        self.assertTrue(result)

        # Calculate expected file size
        height, width = self.test_heightmap.shape
        chunks_x = (width + 3) // 4  # ceiling division
        chunks_y = (height + 3) // 4

        # Header + chunk metadata + chunk data
        expected_min_size = 20 + (chunks_x * chunks_y * 16) + (width * height * 4)

        self.assertGreater(os.path.getsize(output_file), expected_min_size * 0.9)

    def test_nonexistent_directory(self):
        """Test export to non-existent directory.."""
        new_dir = os.path.join(self.temp_dir.name, "new_folder", "subfolder")
        output_file = os.path.join(new_dir, "test.nvbd")

        result = export_heightmap_to_nvbd(self.test_heightmap, output_file)

        self.assertTrue(result)
        self.assertTrue(os.path.exists(output_file))

    def test_error_handling(self):
        """Test error handling with invalid input.."""
        # Test with invalid file path (using a directory instead of a file)
        with self.assertRaises(Exception):
            # This should trigger an exception when opening the file
            export_heightmap_to_nvbd(
                self.test_heightmap, self.temp_dir.name  # This is a directory, not a file
            )


if __name__ == "__main__":
    unittest.main()
