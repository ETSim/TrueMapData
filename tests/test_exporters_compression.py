"""
Unit tests for the compression exporter module.

These tests verify the functionality of the compression export functions.
"""
import os
import tempfile
import unittest

import numpy as np

from tmd.exporters.compression import (
    export_to_npy, 
    export_to_npz, 
    load_from_npy, 
    load_from_npz
)


class TestCompressionExporters(unittest.TestCase):
    """Test case for compression export functions"""
    
    def setUp(self):
        """Create test data and temporary directory"""
        # Create a test height map
        self.height_map = np.zeros((50, 50), dtype=np.float32)
        # Add a simple pattern
        x = np.linspace(-3, 3, 50)
        y = np.linspace(-3, 3, 50)
        X, Y = np.meshgrid(x, y)
        self.height_map = np.sin(X) * np.cos(Y)
        
        # Create test metadata
        self.metadata = {
            'width': 50,
            'height': 50,
            'x_length': 10.0,
            'y_length': 10.0,
            'x_offset': 0.0,
            'y_offset': 0.0,
            'comment': 'Test data',
            'version': 2
        }
        
        # Create test data dictionary
        self.data = self.metadata.copy()
        self.data['height_map'] = self.height_map
        
        # Create a temp directory for test outputs
        self.temp_dir = tempfile.TemporaryDirectory()
    
    def tearDown(self):
        """Clean up temporary directory"""
        self.temp_dir.cleanup()
    
    def test_export_to_npy(self):
        """Test exporting height map to .npy format"""
        # Test with height map only
        output_path = os.path.join(self.temp_dir.name, "height_map.npy")
        result_path = export_to_npy(self.height_map, output_path)
        
        self.assertEqual(result_path, output_path)
        self.assertTrue(os.path.exists(output_path))
        self.assertGreater(os.path.getsize(output_path), 0)
        
        # Test with data dictionary
        output_path = os.path.join(self.temp_dir.name, "height_map_from_dict.npy")
        result_path = export_to_npy(self.data, output_path)
        
        self.assertEqual(result_path, output_path)
        self.assertTrue(os.path.exists(output_path))
        
        # Test error handling for invalid data type
        with self.assertRaises(TypeError):
            export_to_npy("invalid_data", os.path.join(self.temp_dir.name, "invalid.npy"))
    
    def test_load_from_npy(self):
        """Test loading height map from .npy format"""
        # First export the data
        output_path = os.path.join(self.temp_dir.name, "load_test.npy")
        export_to_npy(self.height_map, output_path)
        
        # Then load it back
        loaded_data = load_from_npy(output_path)
        
        # Check that the loaded data matches the original
        self.assertEqual(loaded_data.shape, self.height_map.shape)
        np.testing.assert_array_almost_equal(loaded_data, self.height_map)
        
        # Test error handling for nonexistent file
        with self.assertRaises(FileNotFoundError):
            load_from_npy(os.path.join(self.temp_dir.name, "nonexistent.npy"))
    
    def test_export_to_npz(self):
        """Test exporting TMD data to .npz format"""
        # Test with compression
        output_path = os.path.join(self.temp_dir.name, "tmd_data_compressed.npz")
        result_path = export_to_npz(self.data, output_path, compress=True)
        
        self.assertEqual(result_path, output_path)
        self.assertTrue(os.path.exists(output_path))
        self.assertGreater(os.path.getsize(output_path), 0)
        
        # Test without compression
        output_path = os.path.join(self.temp_dir.name, "tmd_data_uncompressed.npz")
        result_path = export_to_npz(self.data, output_path, compress=False)
        
        self.assertEqual(result_path, output_path)
        self.assertTrue(os.path.exists(output_path))
        
        # Test error handling for invalid data
        with self.assertRaises(TypeError):
            export_to_npz("invalid_data", os.path.join(self.temp_dir.name, "invalid.npz"))
        
        # Test error handling for missing height map
        with self.assertRaises(ValueError):
            export_to_npz(self.metadata, os.path.join(self.temp_dir.name, "missing_height_map.npz"))
    
    def test_load_from_npz(self):
        """Test loading TMD data from .npz format"""
        # First export the data
        output_path = os.path.join(self.temp_dir.name, "load_npz_test.npz")
        export_to_npz(self.data, output_path)
        
        # Then load it back
        loaded_data = load_from_npz(output_path)
        
        # Check that the loaded data has the height map
        self.assertIn('height_map', loaded_data)
        self.assertEqual(loaded_data['height_map'].shape, self.height_map.shape)
        np.testing.assert_array_almost_equal(loaded_data['height_map'], self.height_map)
        
        # Test error handling for nonexistent file
        with self.assertRaises(FileNotFoundError):
            load_from_npz(os.path.join(self.temp_dir.name, "nonexistent.npz"))
    
    def test_data_roundtrip(self):
        """Test a full export-load roundtrip with metadata preservation"""
        # Export the data
        output_path = os.path.join(self.temp_dir.name, "roundtrip_test.npz")
        export_to_npz(self.data, output_path)
        
        # Load it back
        loaded_data = load_from_npz(output_path)
        
        # Check height map
        self.assertEqual(loaded_data['height_map'].shape, self.height_map.shape)
        np.testing.assert_array_almost_equal(loaded_data['height_map'], self.height_map)
        
        # Check metadata (some conversion issues may occur due to JSON serialization)
        for key in ['width', 'height']:
            self.assertEqual(loaded_data[key], self.metadata[key])


if __name__ == "__main__":
    unittest.main()
