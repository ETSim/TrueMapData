"""Unit tests for TMD npz compression module."""

import unittest
from unittest.mock import patch, mock_open, MagicMock
import os
import json
import numpy as np
import tempfile
import shutil

from tmd.exporters.compression.npz import export_to_npz, load_from_npz
from tests.resources import create_sample_height_map


class TestNPZCompression(unittest.TestCase):
    """Test class for NPZ compression functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample data
        self.sample_height_map = create_sample_height_map(pattern="with_nan")
        
        # Sample metadata for testing
        self.sample_metadata = {
            "version": "1.0",
            "comment": "Test file",
            "width": 5,
            "height": 5,
            "x_length": 10.0,
            "y_length": 10.0,
            "x_offset": 0.0,
            "y_offset": 0.0,
            "custom_object": {"nested": "value"},  # Complex object for JSON testing
        }
        
        # Full sample data
        self.sample_data = {
            "height_map": self.sample_height_map,
            **self.sample_metadata
        }
        
        # Test file path
        self.test_file_path = os.path.join(self.temp_dir, "test_data.npz")
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary directory and its contents
        shutil.rmtree(self.temp_dir)
    
    def test_export_with_compression(self):
        """Test exporting data with compression enabled."""
        output_path = export_to_npz(self.sample_data, self.test_file_path, compress=True)
        
        # Check if the file was created
        self.assertTrue(os.path.exists(output_path))
        self.assertEqual(output_path, self.test_file_path)
        
        # Verify we can load the file
        loaded_data = np.load(output_path)
        self.assertIn("height_map", loaded_data)
        self.assertIn("metadata", loaded_data)
    
    def test_export_without_compression(self):
        """Test exporting data with compression disabled."""
        output_path = export_to_npz(self.sample_data, self.test_file_path, compress=False)
        
        # Check if the file was created
        self.assertTrue(os.path.exists(output_path))
        
        # Verify we can load the file
        loaded_data = np.load(output_path)
        self.assertIn("height_map", loaded_data)
        self.assertIn("metadata", loaded_data)
    
    def test_export_creates_directory(self):
        """Test if export_to_npz creates the output directory if it doesn't exist."""
        nested_path = os.path.join(self.temp_dir, "nested", "dir", "test.npz")
        
        output_path = export_to_npz(self.sample_data, nested_path)
        
        # Check if the file and directories were created
        self.assertTrue(os.path.exists(output_path))
        self.assertTrue(os.path.isdir(os.path.dirname(nested_path)))
    
    def test_export_invalid_data_type(self):
        """Test exporting invalid data types."""
        invalid_data = "not a dictionary"
        
        with self.assertRaises(TypeError):
            export_to_npz(invalid_data, self.test_file_path)
    
    def test_export_missing_height_map(self):
        """Test exporting data without height_map."""
        invalid_data = {"version": "1.0"}  # No height_map
        
        with self.assertRaises(ValueError):
            export_to_npz(invalid_data, self.test_file_path)
    
    def test_export_invalid_height_map(self):
        """Test exporting data with invalid height_map type."""
        invalid_data = {"height_map": "not an array"}
        
        with self.assertRaises(TypeError):
            export_to_npz(invalid_data, self.test_file_path)
    
    def test_load_complete_data(self):
        """Test loading a valid NPZ file with complete data."""
        # First create a valid file to load
        export_to_npz(self.sample_data, self.test_file_path)
        
        # Load the file
        loaded_data = load_from_npz(self.test_file_path)
        
        # Check if the height map data matches
        np.testing.assert_array_equal(loaded_data["height_map"], self.sample_height_map)
        
        # Check if metadata was properly loaded
        self.assertEqual(loaded_data["version"], self.sample_metadata["version"])
        self.assertEqual(loaded_data["comment"], self.sample_metadata["comment"])
        # Complex objects should be string-converted but present
        self.assertIn("custom_object", loaded_data)
    
    def test_load_nonexistent_file(self):
        """Test loading a non-existent file."""
        nonexistent_path = os.path.join(self.temp_dir, "nonexistent.npz")
        
        with self.assertRaises(FileNotFoundError):
            load_from_npz(nonexistent_path)
    
    @patch('numpy.load')
    def test_load_invalid_npz(self, mock_np_load):
        """Test loading an invalid NPZ file."""
        # Make numpy.load raise an exception
        mock_np_load.side_effect = ValueError("Invalid file format")
        
        # Create an empty file
        with open(self.test_file_path, 'w') as f:
            f.write("Not a valid NPZ file")
        
        with self.assertRaises(ValueError):
            load_from_npz(self.test_file_path)
    
    @patch('numpy.load')
    @patch('pathlib.Path.exists')
    def test_load_missing_height_map(self, mock_exists, mock_np_load):
        """Test loading an NPZ file without height_map."""
        # Set up the mock to report the file exists
        mock_exists.return_value = True
        
        # Create a mock for the returned npz data that doesn't have height_map
        mock_data = MagicMock()
        mock_data.__getitem__.side_effect = lambda key: {"metadata": json.dumps({"version": "1.0"})}.get(key)
        mock_data.__contains__ = lambda self, key: key in ["metadata"]
        mock_np_load.return_value = mock_data
        
        with self.assertRaises(ValueError):
            load_from_npz(self.test_file_path)
    
    @patch('json.loads')
    def test_load_with_invalid_metadata(self, mock_json_loads):
        """Test loading an NPZ file with invalid metadata JSON."""
        # First create a valid file
        export_to_npz(self.sample_data, self.test_file_path)
        
        # Make json.loads raise an exception when parsing metadata
        mock_json_loads.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        
        # Load should still work but with just the height map
        loaded_data = load_from_npz(self.test_file_path)
        
        # Should have height map but maybe not all metadata
        self.assertIn("height_map", loaded_data)
        np.testing.assert_array_equal(loaded_data["height_map"], self.sample_height_map)
        
        # Metadata should not be present or should be empty
        for key in self.sample_metadata.keys():
            if key in loaded_data:
                self.assertNotEqual(loaded_data[key], self.sample_metadata[key])


if __name__ == '__main__':
    unittest.main()
