""".

Unit tests for metadata utilities.
"""

import os
import unittest
import tempfile
import numpy as np
import json

from tmd.utils.metadata import (
    compute_stats,
    export_metadata,
    export_metadata_txt,
    extract_metadata
    # Removed update_metadata since it doesn't exist
)

class TestMetadata(unittest.TestCase):
    """Test cases for metadata utilities.."""
    
    def setUp(self):
        """Set up test data.."""
        # Create a simple heightmap for testing
        self.height_map = np.zeros((50, 50), dtype=np.float32)
        for i in range(50):
            for j in range(50):
                self.height_map[i, j] = np.sin(i/5) * np.cos(j/5)
        
        # Create sample metadata
        self.metadata = {
            "file_path": "sample.tmd",
            "units": "mm",
            "x_length": 10.0,
            "y_length": 10.0,
            "data_type": "height_map"
        }
        
        # Create a temporary directory for test outputs
        self.temp_dir = tempfile.TemporaryDirectory()
    
    def tearDown(self):
        """Clean up temporary files.."""
        self.temp_dir.cleanup()
    
    def test_compute_stats(self):
        """Test computation of height map statistics.."""
        stats = compute_stats(self.height_map)
        
        # Check that all expected stats are present
        self.assertIn("min", stats)
        self.assertIn("max", stats)
        self.assertIn("mean", stats)
        self.assertIn("median", stats)
        self.assertIn("std", stats)
        self.assertIn("shape", stats)
        
        # Check that the values are reasonable
        self.assertEqual(stats["shape"], (50, 50))
        self.assertAlmostEqual(stats["min"], -1.0, places=1)
        self.assertAlmostEqual(stats["max"], 1.0, places=1)
    
    def test_export_metadata(self):
        """Test exporting metadata to a text file.."""
        output_path = os.path.join(self.temp_dir.name, "metadata.txt")
        stats = compute_stats(self.height_map)
        
        result = export_metadata(self.metadata, stats, output_path)
        
        # Check that the file exists and contains expected content
        self.assertTrue(os.path.exists(result))
        
        with open(result, "r") as f:
            content = f.read()
            self.assertIn("units: mm", content)
            self.assertIn("x_length: 10.0", content)
            self.assertIn("mean:", content)
    
    def test_export_metadata_txt(self):
        """Test exporting TMD metadata to a text file.."""
        output_path = os.path.join(self.temp_dir.name, "tmd_metadata.txt")
        
        # Prepare a data dictionary with height map
        data_dict = self.metadata.copy()
        data_dict["height_map"] = self.height_map
        
        result = export_metadata_txt(data_dict, output_path)
        
        # Check that the file exists and contains expected content
        self.assertTrue(os.path.exists(result))
        
        with open(result, "r") as f:
            content = f.read()
            self.assertIn("units: mm", content)
            self.assertIn("x_length: 10.0", content)
            self.assertIn("Height Map Statistics", content)
            self.assertIn("Shape:", content)

if __name__ == "__main__":
    unittest.main()
