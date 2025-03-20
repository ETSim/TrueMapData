"""
Unit tests for the TMD utility functions.

These tests verify the functionality of the core utility functions for
TMD file processing and manipulation.
"""
import os
import struct
import tempfile
import unittest
from io import BytesIO
from unittest.mock import MagicMock, mock_open, patch

import numpy as np

from tmd.utils.utils import (
    create_sample_height_map,
    detect_tmd_version,
    generate_synthetic_tmd,
    get_header_offset,
    hexdump,
    process_tmd_file,
    read_null_terminated_string,
    write_tmd_file,
)


class TestHexdump(unittest.TestCase):
    """Test hexdump function"""

    def test_basic_hexdump(self):
        """Test basic hexdump functionality"""
        data = b"Hello, World!"
        result = hexdump(data)
        # Verify it contains both hex and ASCII parts
        self.assertIn("48 65 6c 6c 6f", result)  # hex for "Hello"
        self.assertIn("|Hello, World!|", result)

    def test_hexdump_with_offset(self):
        """Test hexdump with start offset"""
        data = b"Hello, World!"
        result = hexdump(data, start=7)
        # Should only include "World!"
        self.assertIn("57 6f 72 6c 64", result)  # hex for "World"
        self.assertIn("|World!|", result)
        self.assertNotIn("48 65 6c 6c 6f", result)  # "Hello" should be skipped

    def test_hexdump_with_length(self):
        """Test hexdump with specific length"""
        data = b"Hello, World!"
        result = hexdump(data, length=5)
        # Should only include "Hello"
        self.assertIn("48 65 6c 6c 6f", result)  # hex for "Hello"
        # Check that "World" is not included
        self.assertNotIn("57 6f 72 6c 64 21", result)  # "World!" should not be included

    def test_hexdump_width(self):
        """Test hexdump with different width"""
        data = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        result_width8 = hexdump(data, width=8)
        result_width16 = hexdump(data, width=16)

        # Width 8 should have more lines than width 16
        self.assertGreater(result_width8.count("\n"), result_width16.count("\n"))

    def test_hexdump_no_ascii(self):
        """Test hexdump without ASCII representation"""
        data = b"Hello, World!"
        result = hexdump(data, show_ascii=False)
        # Should not contain the ASCII part
        self.assertNotIn("|Hello", result)
        # But should contain the hex part
        self.assertIn("48 65 6c 6c 6f", result)


class TestReadNullTerminatedString(unittest.TestCase):
    """Test read_null_terminated_string function"""

    def test_with_null_terminator(self):
        """Test reading string with null terminator"""
        test_data = b"Hello\0World"
        file_mock = BytesIO(test_data)
        result = read_null_terminated_string(file_mock)
        self.assertEqual(result, "Hello")
        # Verify seek position is after null terminator
        self.assertEqual(file_mock.tell(), 6)  # 5 chars + null byte

    def test_without_null_terminator(self):
        """Test reading string without null terminator"""
        test_data = b"Hello World"
        file_mock = BytesIO(test_data)
        result = read_null_terminated_string(file_mock)
        self.assertEqual(result, "Hello World")

    def test_empty_string(self):
        """Test reading empty string"""
        test_data = b"\0Rest of data"
        file_mock = BytesIO(test_data)
        result = read_null_terminated_string(file_mock)
        self.assertEqual(result, "")
        # Verify seek position is after null terminator
        self.assertEqual(file_mock.tell(), 1)

    def test_chunk_size_limit(self):
        """Test reading with chunk size limit"""
        test_data = b"A" * 300  # String larger than default chunk size
        file_mock = BytesIO(test_data)
        result = read_null_terminated_string(file_mock, chunk_size=100)
        # Should read only up to the chunk size
        self.assertEqual(result, "A" * 100)


class TestDetectTMDVersion(unittest.TestCase):
    """Test detect_tmd_version function"""

    @patch("os.path.exists", return_value=True)
    def test_detect_v2(self, mock_exists):
        """Test detecting version 2 TMD file"""
        with patch("builtins.open", mock_open(read_data=b"Binary TrueMap Data File v2.0\0")):
            version = detect_tmd_version("dummy.tmd")
            self.assertEqual(version, 2)

    @patch("os.path.exists", return_value=True)
    def test_detect_v1(self, mock_exists):
        """Test detecting version 1 TMD file"""
        with patch("builtins.open", mock_open(read_data=b"TrueMap Data File v1.0\0")):
            version = detect_tmd_version("dummy.tmd")
            self.assertEqual(version, 1)

    @patch("os.path.exists", return_value=True)
    def test_detect_generic_binary(self, mock_exists):
        """Test detecting generic Binary TrueMap file (should default to v2)"""
        with patch("builtins.open", mock_open(read_data=b"Binary TrueMap Data File\0")):
            version = detect_tmd_version("dummy.tmd")
            self.assertEqual(version, 2)

    @patch("os.path.exists", return_value=True)
    def test_detect_unknown(self, mock_exists):
        """Test detecting unknown format (should default to v1)"""
        with patch("builtins.open", mock_open(read_data=b"Unknown Format\0")):
            version = detect_tmd_version("dummy.tmd")
            self.assertEqual(version, 1)

    @patch("os.path.exists", return_value=False)
    def test_detect_nonexistent_file(self, mock_exists):
        """Test detecting version with non-existent file"""
        with self.assertRaises(FileNotFoundError):
            detect_tmd_version("nonexistent.tmd")


class TestGetHeaderOffset(unittest.TestCase):
    """Test get_header_offset function"""

    def test_v1_offset(self):
        """Test getting header offset for v1"""
        offset = get_header_offset(1)
        self.assertEqual(offset, 32)

    def test_v2_offset(self):
        """Test getting header offset for v2"""
        offset = get_header_offset(2)
        self.assertEqual(offset, 64)

    def test_invalid_version(self):
        """Test getting header offset with invalid version (should default to v2)"""
        offset = get_header_offset(999)
        self.assertEqual(offset, 64)


class TestProcessTMDFile(unittest.TestCase):
    """Test process_tmd_file function"""

    def setUp(self):
        """Set up temporary directory and sample data"""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = self.temp_dir.name

    def tearDown(self):
        """Clean up temporary directory"""
        self.temp_dir.cleanup()

    def create_test_tmd_file(self, version=2, width=10, height=8, data_size_factor=1):
        """Create a test TMD file with specified parameters

        Args:
            version: TMD version (1 or 2)
            width: Width of height map
            height: Height of height map
            data_size_factor: Factor to multiply expected data size by (for testing size mismatches)

        Returns:
            Path to the created file
        """
        test_file = os.path.join(self.test_dir, f"test_v{version}.tmd")

        with open(test_file, "wb") as f:
            # Write header
            if version == 2:
                # Write the exact header format shown in the example
                header = "Binary TrueMap Data File v2.0\n"
                header_comment = " Created by TrueMap v6 "

                # Write header and pad to 32 bytes with nulls if needed
                header_bytes = header.encode("ascii")
                remaining_header = 32 - len(header_bytes)
                if remaining_header > 0:
                    header_bytes += b"\0" * remaining_header
                f.write(header_bytes[:32])  # Truncate if too long

                # Write comment
                comment_str = "Created by TrueMap v6"
                comment_bytes = header_comment.encode("ascii")
                remaining_comment = 24 - len(comment_bytes)
                if remaining_comment > 0:
                    comment_bytes += b"\0" * remaining_comment
                f.write(comment_bytes[:24])  # Truncate if too long
            else:
                # v1 format has a simpler header
                header_str = "TrueMap Data File v1.0"
                header_bytes = header_str.encode("ascii") + b"\0"
                f.write(header_bytes.ljust(28, b"\0"))

            # Write dimensions
            f.write(struct.pack("<II", width, height))

            # Write spatial info
            f.write(struct.pack("<ffff", 10.0, 8.0, 0.0, 0.0))

            # Write height map data
            expected_size = width * height
            actual_size = int(expected_size * data_size_factor)
            data = np.zeros(actual_size, dtype=np.float32)
            # Add some pattern to the data
            for i in range(actual_size):
                data[i] = float(i % 256) / 255.0

            f.write(data.tobytes())

        return test_file

    def test_process_v2_file(self):
        """Test processing a v2 TMD file"""
        test_file = self.create_test_tmd_file(version=2, width=10, height=8)

        metadata, height_map = process_tmd_file(test_file, debug=True)

        # Check metadata
        self.assertEqual(metadata["version"], 2)
        self.assertEqual(metadata["width"], 10)
        self.assertEqual(metadata["height"], 8)
        self.assertEqual(metadata["x_length"], 10.0)
        self.assertEqual(metadata["y_length"], 8.0)
        self.assertEqual(metadata["x_offset"], 0.0)
        self.assertEqual(metadata["y_offset"], 0.0)
        self.assertEqual(metadata["comment"], "Created by TrueMap v6 \x00")

        # Check height map
        self.assertEqual(height_map.shape, (8, 10))
        self.assertEqual(height_map.dtype, np.float32)

    def test_process_v1_file(self):
        """Test processing a v1 TMD file"""
        test_file = self.create_test_tmd_file(version=1, width=10, height=8)

        metadata, height_map = process_tmd_file(test_file)

        # Check metadata
        self.assertEqual(metadata["version"], 1)
        self.assertEqual(metadata["width"], 10)
        self.assertEqual(metadata["height"], 8)
        # Check no comment for v1
        self.assertIsNone(metadata["comment"])

        # Check height map
        self.assertEqual(height_map.shape, (8, 10))

    def test_process_with_force_offset(self):
        """Test processing with forced offset"""
        test_file = self.create_test_tmd_file()

        force_offset = (5.0, 10.0)
        metadata, height_map = process_tmd_file(test_file, force_offset=force_offset)

        # Check forced offsets were applied
        self.assertEqual(metadata["x_offset"], 5.0)
        self.assertEqual(metadata["y_offset"], 10.0)

    def test_process_with_incomplete_data(self):
        """Test processing a file with incomplete data (should pad with zeros)"""
        test_file = self.create_test_tmd_file(width=10, height=8, data_size_factor=0.5)

        metadata, height_map = process_tmd_file(test_file, debug=True)

        # Check height map was padded to expected dimensions
        self.assertEqual(height_map.shape, (8, 10))
        # Check some values are zeros (padding)
        self.assertEqual(height_map[-1, -1], 0.0)

    def test_process_with_extra_data(self):
        """Test processing a file with extra data (should trim excess)"""
        test_file = self.create_test_tmd_file(width=10, height=8, data_size_factor=1.5)

        metadata, height_map = process_tmd_file(test_file, debug=True)

        # Check height map was trimmed to expected dimensions
        self.assertEqual(height_map.shape, (8, 10))

    def test_process_nonexistent_file(self):
        """Test processing a non-existent file (should raise FileNotFoundError)"""
        with self.assertRaises(FileNotFoundError):
            process_tmd_file(os.path.join(self.test_dir, "nonexistent.tmd"))


class TestWriteTMDFile(unittest.TestCase):
    """Test write_tmd_file function"""

    def setUp(self):
        """Set up temporary directory"""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = self.temp_dir.name

        # Create sample height map
        self.height_map = np.zeros((8, 10), dtype=np.float32)
        for i in range(8):
            for j in range(10):
                self.height_map[i, j] = float((i * 10 + j) % 256) / 255.0

    def tearDown(self):
        """Clean up temporary directory"""
        self.temp_dir.cleanup()

    def test_write_basic_tmd_file(self):
        """Test writing a basic TMD file"""
        output_path = os.path.join(self.test_dir, "output.tmd")

        result_path = write_tmd_file(
            height_map=self.height_map, output_path=output_path, version=2, debug=True
        )

        # Verify the file was created
        self.assertTrue(os.path.exists(result_path))
        self.assertEqual(result_path, output_path)

        # Verify the file has non-zero size
        self.assertGreater(os.path.getsize(output_path), 0)

        # Verify we can read it back
        metadata, height_map = process_tmd_file(output_path)
        self.assertEqual(metadata["width"], 10)
        self.assertEqual(metadata["height"], 8)
        self.assertEqual(height_map.shape, (8, 10))
        # Verify data integrity
        np.testing.assert_array_equal(height_map, self.height_map)

    def test_write_with_custom_parameters(self):
        """Test writing a TMD file with custom parameters"""
        output_path = os.path.join(self.test_dir, "custom.tmd")

        comment = "Created by TrueMap v6\n\x00\x00"
        x_length = 20.0
        y_length = 15.0
        x_offset = 5.0
        y_offset = 10.0

        write_tmd_file(
            height_map=self.height_map,
            output_path=output_path,
            comment=comment,
            x_length=x_length,
            y_length=y_length,
            x_offset=x_offset,
            y_offset=y_offset,
            version=2,
        )

        # Verify we can read it back with custom parameters
        metadata, _ = process_tmd_file(output_path)
        self.assertEqual(metadata["x_length"], x_length)
        self.assertEqual(metadata["y_length"], y_length)
        self.assertEqual(metadata["x_offset"], x_offset)
        self.assertEqual(metadata["y_offset"], y_offset)
        self.assertEqual(metadata["comment"].strip(), comment.strip())

    def test_write_to_new_directory(self):
        """Test writing to a directory that doesn't exist (should create it)"""
        new_dir = os.path.join(self.test_dir, "new_dir")
        output_path = os.path.join(new_dir, "output.tmd")

        # Directory shouldn't exist yet
        self.assertFalse(os.path.exists(new_dir))

        write_tmd_file(height_map=self.height_map, output_path=output_path, version=2)

        # Verify the directory was created
        self.assertTrue(os.path.exists(new_dir))
        self.assertTrue(os.path.exists(output_path))


class TestCreateSampleHeightMap(unittest.TestCase):
    """Test create_sample_height_map function"""

    def test_default_parameters(self):
        """Test creating a height map with default parameters"""
        height_map = create_sample_height_map()

        # Verify dimensions
        self.assertEqual(height_map.shape, (100, 100))
        self.assertEqual(height_map.dtype, np.float32)

        # Verify normalized range
        self.assertGreaterEqual(height_map.min(), 0.0)
        self.assertLessEqual(height_map.max(), 1.0)

    def test_custom_dimensions(self):
        """Test creating a height map with custom dimensions"""
        width = 200
        height = 150
        height_map = create_sample_height_map(width=width, height=height)

        self.assertEqual(height_map.shape, (height, width))

    def test_pattern_types(self):
        """Test creating height maps with different pattern types"""
        patterns = ["waves", "peak", "dome", "ramp", "combined"]

        for pattern in patterns:
            height_map = create_sample_height_map(pattern=pattern)

            # Verify basic properties
            self.assertEqual(height_map.shape, (100, 100))
            self.assertGreaterEqual(height_map.min(), 0.0)
            self.assertLessEqual(height_map.max(), 1.0)

            # Pattern-specific checks could be added here

    def test_noise_levels(self):
        """Test creating height maps with different noise levels"""
        # Use a constant pattern for maximum sensitivity to noise
        np.random.seed(42)  # Fix seed for both calls
        constant_pattern = create_sample_height_map(pattern="dome", noise_level=0.0)

        # The issue is probably that normalization eliminates noise effects
        # Let's examine the raw arrays before normalization
        np.random.seed(42)  # Reset seed to get same pattern

        # Create dummy height map just to get X,Y coordinates
        width, height = 100, 100
        x = np.linspace(-5, 5, width)
        y = np.linspace(-5, 5, height)
        X, Y = np.meshgrid(x, y)

        # Use constant pattern that won't be affected by normalization
        Z = np.ones((height, width))

        # Add extreme noise to one copy
        noise = np.random.normal(0, 0.5, Z.shape)
        noisy_Z = Z + noise

        # Check difference to verify test
        self.assertGreater(np.std(noisy_Z), np.std(Z))

        # Now test the actual function with extra safety to ensure normalization doesn't flatten noise
        np.random.seed(42)
        base_map = create_sample_height_map(pattern="dome", noise_level=0.0)
        np.random.seed(42)
        # Use high enough noise that it will definitely exceed pattern's inherent variation
        noisy_map = create_sample_height_map(pattern="dome", noise_level=2.0)

        # Calculate the average absolute difference between maps
        diff = np.abs(noisy_map - base_map)
        mean_diff = np.mean(diff)

        # There should be a significant difference if noise is properly applied
        self.assertGreater(mean_diff, 0.01)


class TestGenerateSyntheticTMD(unittest.TestCase):
    """Test generate_synthetic_tmd function"""

    def setUp(self):
        """Set up temporary directory"""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = self.temp_dir.name

    def tearDown(self):
        """Clean up temporary directory"""
        self.temp_dir.cleanup()

    @patch("tmd.utils.utils.create_sample_height_map")
    @patch("tmd.utils.utils.write_tmd_file")
    def test_default_parameters(self, mock_write, mock_create):
        """Test generating a synthetic TMD file with default parameters"""
        # Mocks for testing the function call signatures
        mock_height_map = np.zeros((100, 100), dtype=np.float32)
        mock_create.return_value = mock_height_map
        mock_write.return_value = "output/synthetic.tmd"

        result = generate_synthetic_tmd()

        # Check that the function was called (any call is fine)
        mock_create.assert_called()
        mock_write.assert_called()
        self.assertEqual(result, "output/synthetic.tmd")

    @patch("tmd.utils.utils.create_sample_height_map")
    @patch("tmd.utils.utils.write_tmd_file")
    def test_custom_parameters(self, mock_write, mock_create):
        """Test generating a synthetic TMD file with custom parameters"""
        mock_height_map = np.zeros((200, 200), dtype=np.float32)
        mock_create.return_value = mock_height_map
        mock_write.return_value = "custom/path.tmd"

        output_path = "custom/path.tmd"
        width = 200
        height = 200
        pattern = "waves"
        comment = "Created by TrueMap v6"
        version = 1

        result = generate_synthetic_tmd(
            output_path=output_path,
            width=width,
            height=height,
            pattern=pattern,
            comment=comment,
            version=version,
        )

        # Check that the functions were called
        mock_create.assert_called()
        mock_write.assert_called()

        # Check if comment and version were passed to write_tmd_file
        write_args = mock_write.call_args[1]
        self.assertEqual(write_args["comment"], comment)
        self.assertEqual(write_args["version"], version)
        self.assertEqual(result, output_path)

    def test_integration(self):
        """Integration test of generate_synthetic_tmd"""
        output_path = os.path.join(self.test_dir, "synthetic.tmd")

        result = generate_synthetic_tmd(
            output_path=output_path, width=50, height=40, pattern="waves"
        )

        # Verify file was created
        self.assertTrue(os.path.exists(result))

        # Verify we can read it back
        metadata, height_map = process_tmd_file(result)
        self.assertEqual(metadata["width"], 50)
        self.assertEqual(metadata["height"], 40)
        self.assertEqual(height_map.shape, (40, 50))


if __name__ == "__main__":
    unittest.main()
