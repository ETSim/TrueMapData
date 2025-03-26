"""Unit tests for TMD npy compression module."""

import unittest
from unittest.mock import patch, mock_open, MagicMock
import os
import numpy as np
import tempfile
import shutil

from tmd.exporters.compression.npy import export_to_npy, load_from_npy

class TestNPYCompression(unittest.TestCase):
    """Test class for NPY compression functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample data
        self.sample_height_map = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, np.nan]
        ])
        
        self.sample_data_dict = {
            "height_map": self.sample_height_map,
            "metadata": {"key": "value"}
        }
        
        # Test file path
        self.test_file_path = os.path.join(self.temp_dir, "test_height_map.npy")
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary directory and its contents
        shutil.rmtree(self.temp_dir)
    
    def test_export_array(self):
        """Test exporting a NumPy array to NPY format."""
        output_path = export_to_npy(self.sample_height_map, self.test_file_path)
        
        # Check if the file was created
        self.assertTrue(os.path.exists(output_path))
        
        # Load the file and check if the data matches
        loaded_data = np.load(output_path)
        np.testing.assert_array_equal(loaded_data, self.sample_height_map)
    
    def test_export_dict(self):
        """Test exporting a dictionary with height_map to NPY format."""
        output_path = export_to_npy(self.sample_data_dict, self.test_file_path)
        
        # Check if the file was created
        self.assertTrue(os.path.exists(output_path))
        
        # Load the file and check if the data matches
        loaded_data = np.load(output_path)
        np.testing.assert_array_equal(loaded_data, self.sample_height_map)
    
    def test_export_creates_directory(self):
        """Test if export_to_npy creates the output directory if it doesn't exist."""
        nested_path = os.path.join(self.temp_dir, "nested", "dir", "test.npy")
        
        output_path = export_to_npy(self.sample_height_map, nested_path)
        
        # Check if the file and directories were created
        self.assertTrue(os.path.exists(output_path))
    
    def test_export_invalid_data(self):
        """Test exporting invalid data types."""
        invalid_data = "not a numpy array or dict"
        
        with self.assertRaises(TypeError):
            export_to_npy(invalid_data, self.test_file_path)
    
    def test_load_valid_file(self):
        """Test loading a valid NPY file."""
        # First create a file to load
        np.save(self.test_file_path, self.sample_height_map)
        
        loaded_data = load_from_npy(self.test_file_path)
        
        # Check if the loaded data matches the original
        np.testing.assert_array_equal(loaded_data, self.sample_height_map)
    
    def test_load_nonexistent_file(self):
        """Test loading a non-existent file."""
        nonexistent_path = os.path.join(self.temp_dir, "nonexistent.npy")
        
        with self.assertRaises(FileNotFoundError):
            load_from_npy(nonexistent_path)
    
    @patch('numpy.load')
    def test_load_invalid_file(self, mock_np_load):
        """Test loading an invalid NPY file."""
        # Make numpy.load raise an exception
        mock_np_load.side_effect = ValueError("Invalid file format")
        
        # Create an empty file
        with open(self.test_file_path, 'w') as f:
            f.write("Not a valid NPY file")
        
        with self.assertRaises(ValueError):
            load_from_npy(self.test_file_path)


if __name__ == '__main__':
    unittest.main()
