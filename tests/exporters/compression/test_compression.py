""".

Unit tests for compression exporters.
"""

import os
import unittest
import tempfile
import numpy as np

from tmd.exporters.compression import (
    export_to_npy,
    load_from_npy,
    export_to_npz,
    load_from_npz
)

class TestCompressionExporters(unittest.TestCase):
    """Test cases for compression exporters.."""
    
    def setUp(self):
        """Set up test data.."""
        # Create a simple heightmap for testing
        self.height_map = np.zeros((50, 50), dtype=np.float32)
        
        # Add a simple pattern
        x, y = np.meshgrid(np.linspace(-3, 3, 50), np.linspace(-3, 3, 50))
        self.height_map = np.sin(x) * np.cos(y)
        
        # Create metadata
        self.metadata = {
            "units": "mm",
            "x_length": 10.0,
            "y_length": 10.0,
            "data_source": "test"
        }
        
        # Create complete data dictionary
        self.data = self.metadata.copy()
        self.data["height_map"] = self.height_map
        
        # Create a temporary directory for test outputs
        self.temp_dir = tempfile.TemporaryDirectory()
    
    def tearDown(self):
        """Clean up temporary files.."""
        self.temp_dir.cleanup()
    
    def test_export_to_npy(self):
        """Test exporting to NPY format.."""
        # Test with direct height map
        output_path = os.path.join(self.temp_dir.name, "heightmap.npy")
        result = export_to_npy(self.height_map, output_path)
        
        # Check that the file exists
        self.assertTrue(os.path.exists(result))
        
        # Test with dictionary
        output_path2 = os.path.join(self.temp_dir.name, "heightmap_from_dict.npy")
        result2 = export_to_npy(self.data, output_path2)
        
        # Check that the file exists
        self.assertTrue(os.path.exists(result2))
        
        # Test error handling
        with self.assertRaises(TypeError):
            export_to_npy("not_a_numpy_array", os.path.join(self.temp_dir.name, "error.npy"))
    
    def test_load_from_npy(self):
        """Test loading from NPY format.."""
        # Export first
        output_path = os.path.join(self.temp_dir.name, "heightmap_to_load.npy")
        export_to_npy(self.height_map, output_path)
        
        # Load and verify
        loaded_height_map = load_from_npy(output_path)
        
        # Check shape and values
        self.assertEqual(loaded_height_map.shape, self.height_map.shape)
        self.assertTrue(np.allclose(loaded_height_map, self.height_map))
        
        # Test error handling
        with self.assertRaises(FileNotFoundError):
            load_from_npy(os.path.join(self.temp_dir.name, "nonexistent.npy"))
    
    def test_export_to_npz(self):
        """Test exporting to NPZ format.."""
        # Test with compression
        output_path = os.path.join(self.temp_dir.name, "data_compressed.npz")
        result = export_to_npz(self.data, output_path, compress=True)
        
        # Check that the file exists
        self.assertTrue(os.path.exists(result))
        
        # Test without compression
        output_path2 = os.path.join(self.temp_dir.name, "data_uncompressed.npz")
        result2 = export_to_npz(self.data, output_path2, compress=False)
        
        # Check that the file exists
        self.assertTrue(os.path.exists(result2))
        
        # Test error handling
        with self.assertRaises(TypeError):
            export_to_npz("not_a_dict", os.path.join(self.temp_dir.name, "error.npz"))
        
        with self.assertRaises(ValueError):
            export_to_npz({"no_height_map": True}, os.path.join(self.temp_dir.name, "error.npz"))
    
    def test_load_from_npz(self):
        """Test loading from NPZ format.."""
        # Export first
        output_path = os.path.join(self.temp_dir.name, "data_to_load.npz")
        export_to_npz(self.data, output_path)
        
        # Load and verify
        loaded_data = load_from_npz(output_path)
        
        # Check that the height map is present and correct
        self.assertIn("height_map", loaded_data)
        self.assertEqual(loaded_data["height_map"].shape, self.height_map.shape)
        self.assertTrue(np.allclose(loaded_data["height_map"], self.height_map))
        
        # Check that some metadata was preserved
        self.assertIn("units", loaded_data)
        self.assertEqual(loaded_data["units"], self.metadata["units"])
        
        # Test error handling
        with self.assertRaises(FileNotFoundError):
            load_from_npz(os.path.join(self.temp_dir.name, "nonexistent.npz"))

if __name__ == "__main__":
    unittest.main()
