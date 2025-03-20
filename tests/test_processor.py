"""
Unit tests for the TMDProcessor class.

These tests verify the functionality of the TMDProcessor class for reading
and parsing TMD files.
"""
import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from tmd.processor import TMDProcessor
from tmd.utils.utils import create_sample_height_map, generate_synthetic_tmd

# Disable excessive logging during tests
logging.basicConfig(level=logging.ERROR)


class TestTMDProcessor(unittest.TestCase):
    """Test cases for the TMDProcessor class."""

    def setUp(self):
        """Set up test environment before each test."""
        # Create a temporary directory to store test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = self.temp_dir.name

        # Generate a synthetic TMD file for testing
        self.test_tmd_path = os.path.join(self.test_dir, "test_sample.tmd")
        self.tmd_path = generate_synthetic_tmd(
            output_path=self.test_tmd_path,
            width=50,
            height=40,
            pattern="waves",
            comment="Test TMD File",
            version=2,
        )

        # Create an instance of TMDProcessor for testing
        self.processor = TMDProcessor(self.test_tmd_path)

    def tearDown(self):
        """Clean up test environment after each test."""
        # Clean up temporary directory
        self.temp_dir.cleanup()

    def test_init(self):
        """Test initialization of TMDProcessor."""
        # Test with valid file path
        processor = TMDProcessor(self.test_tmd_path)
        self.assertEqual(processor.file_path, self.test_tmd_path)
        self.assertEqual(processor.basename, os.path.basename(self.test_tmd_path))
        self.assertIsNone(processor.data)
        self.assertFalse(processor.debug)
        self.assertEqual(processor._stats_cache, {})

        # Test with non-existent file (should initialize without error)
        non_existent_path = os.path.join(self.test_dir, "non_existent.tmd")
        processor = TMDProcessor(non_existent_path)
        self.assertEqual(processor.file_path, non_existent_path)

    def test_set_debug(self):
        """Test setting debug mode."""
        # Test enabling debug mode
        result = self.processor.set_debug(True)
        self.assertTrue(self.processor.debug)
        self.assertIs(result, self.processor)  # Should return self for chaining

        # Test disabling debug mode
        result = self.processor.set_debug(False)
        self.assertFalse(self.processor.debug)
        self.assertIs(result, self.processor)

    @patch("tmd.processor.logger")
    def test_print_file_header(self, mock_logger):
        """Test printing file header."""
        # Test with valid file
        self.processor.print_file_header(num_bytes=32)
        self.assertEqual(mock_logger.info.call_count, 2)

        # Test with non-existent file
        non_existent_processor = TMDProcessor(os.path.join(self.test_dir, "non_existent.tmd"))
        with self.assertRaises(FileNotFoundError):
            non_existent_processor.print_file_header()

    def test_get_stats_without_processing(self):
        """Test get_stats before processing."""
        # Should raise ValueError if process() hasn't been called
        with self.assertRaises(ValueError):
            self.processor.get_stats()

    def test_process_valid_file(self):
        """Test processing a valid TMD file."""
        result = self.processor.process()

        # Verify the result is a dict
        self.assertIsInstance(result, dict)

        # Check for expected keys
        expected_keys = [
            "file_path",
            "version",
            "header",
            "comment",
            "width",
            "height",
            "x_length",
            "y_length",
            "x_offset",
            "y_offset",
            "height_map",
        ]
        for key in expected_keys:
            self.assertIn(key, result)

        # Check specific values
        self.assertEqual(result["file_path"], self.test_tmd_path)

        # Use correct dimensions
        self.assertEqual(result["width"], 50)
        self.assertEqual(result["height"], 40)
        self.assertIsInstance(result["height_map"], np.ndarray)
        self.assertEqual(result["height_map"].shape, (40, 50))

    def test_process_nonexistent_file(self):
        """Test processing a non-existent file."""
        non_existent_processor = TMDProcessor(os.path.join(self.test_dir, "non_existent.tmd"))
        result = non_existent_processor.process()
        self.assertIsNone(result)

    def test_process_tiny_file(self):
        """Test processing a file that's too small to be valid."""
        tiny_file_path = os.path.join(self.test_dir, "tiny.tmd")
        with open(tiny_file_path, "wb") as f:
            f.write(b"ABC")  # Only 3 bytes

        tiny_processor = TMDProcessor(tiny_file_path)
        result = tiny_processor.process()
        self.assertIsNone(result)

    def test_process_with_force_offset(self):
        """Test processing with forced offset values."""
        # Process with forced offset
        forced_offsets = (10.0, 20.0)
        result = self.processor.process(force_offset=forced_offsets)

        # Check if the offsets were applied
        self.assertEqual(result["x_offset"], 10.0)
        self.assertEqual(result["y_offset"], 20.0)

    def test_get_stats_after_processing(self):
        """Test get_stats after processing."""
        # Process the file first
        self.processor.process()

        # Now get_stats should work
        stats = self.processor.get_stats()
        self.assertIsInstance(stats, dict)

        # Check for expected keys in stats
        expected_keys = ["min", "max", "mean", "median", "std", "shape", "non_nan", "nan_count"]
        for key in expected_keys:
            self.assertIn(key, stats)

        # Basic validation of stats values
        self.assertIsInstance(stats["min"], float)
        self.assertIsInstance(stats["max"], float)
        self.assertGreaterEqual(stats["max"], stats["min"])
        self.assertIsInstance(stats["mean"], float)
        self.assertIsInstance(stats["shape"], tuple)

        # Test caching - modify the cache and check it's returned
        self.processor._stats_cache["test_key"] = "test_value"
        new_stats = self.processor.get_stats()
        self.assertIn("test_key", new_stats)
        self.assertEqual(new_stats["test_key"], "test_value")

    def test_get_height_map(self):
        """Test retrieving the height map."""
        # Before processing, should return None
        self.assertIsNone(self.processor.get_height_map())

        # After processing
        self.processor.process()
        height_map = self.processor.get_height_map()
        self.assertIsInstance(height_map, np.ndarray)

        # Use correct dimensions
        self.assertEqual(height_map.shape, (40, 50))

    def test_get_metadata(self):
        """Test retrieving metadata."""
        # Before processing, should raise ValueError
        with self.assertRaises(ValueError):
            self.processor.get_metadata()

        # After processing
        self.processor.process()
        metadata = self.processor.get_metadata()
        self.assertIsInstance(metadata, dict)

        # Check that all expected keys are there except height_map
        self.assertIn("version", metadata)
        self.assertIn("width", metadata)
        self.assertIn("height", metadata)
        self.assertNotIn("height_map", metadata)

    def test_export_metadata(self):
        """Test exporting metadata to a file."""
        # Before processing, should raise ValueError
        output_path = os.path.join(self.test_dir, "metadata.txt")
        with self.assertRaises(ValueError):
            self.processor.export_metadata(output_path)

        # After processing
        self.processor.process()
        result_path = self.processor.export_metadata(output_path)

        # Check that the file exists and is not empty
        self.assertTrue(os.path.exists(result_path))
        self.assertGreater(os.path.getsize(result_path), 0)

        # Check that the file contains expected content
        with open(result_path, "r") as f:
            content = f.read()
            self.assertIn("TMD File:", content)
            self.assertIn("version:", content)
            self.assertIn("width:", content)
            self.assertIn("Height Map Statistics", content)


class TestTMDProcessorMocked(unittest.TestCase):
    """Test cases for TMDProcessor using mocks to simulate errors."""

    def setUp(self):
        """Set up test environment before each test."""
        # Create a mock file path
        self.mock_file_path = "/path/to/mock.tmd"
        self.processor = TMDProcessor(self.mock_file_path)

    @patch("tmd.processor.process_tmd_file")
    @patch("tmd.processor.Path")
    def test_process_with_exception(self, mock_path, mock_process_tmd):
        """Test process method when an exception occurs."""
        # Configure mocks
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_instance.stat.return_value.st_size = 1000
        mock_path.return_value = mock_path_instance

        # Make process_tmd_file raise an exception
        mock_process_tmd.side_effect = ValueError("Test error")

        # Call process method
        result = self.processor.process()

        # Verify result is None due to exception
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
