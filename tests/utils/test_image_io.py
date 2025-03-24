""".

Unit tests for image I/O functions.
"""

import os
import unittest
import tempfile
import numpy as np

from tmd.exporters.image.image_io import load_heightmap, save_heightmap  # Updated import

class TestImageIO(unittest.TestCase):
    """Test cases for image I/O functions.."""
    
    def setUp(self):
        """Set up test data.."""
        # Create a simple heightmap for testing
        self.height_map = np.zeros((100, 100), dtype=np.float32)
        
        # Add some patterns
        for i in range(100):
            for j in range(100):
                self.height_map[i, j] = np.sin(i/10) * np.cos(j/10) + 1  # Range [0, 2]
        
        # Create a temporary directory for test outputs
        self.temp_dir = tempfile.TemporaryDirectory()
    
    def tearDown(self):
        """Clean up temporary files.."""
        self.temp_dir.cleanup()
    
    def test_save_and_load_npy(self):
        """Test saving and loading heightmap as NPY format.."""
        # Save heightmap as NPY
        output_path = os.path.join(self.temp_dir.name, "test_heightmap.npy")
        saved_path = save_heightmap(self.height_map, output_path, normalize=True)
        
        # Check if file exists
        self.assertTrue(os.path.exists(saved_path))
        
        # Load the saved heightmap
        loaded_heightmap = load_heightmap(saved_path, normalize=False)
        
        # Check if loaded heightmap has the same shape
        self.assertEqual(loaded_heightmap.shape, self.height_map.shape)
        
        # Check if values are similar (allowing for small differences due to normalization)
        self.assertTrue(np.allclose(
            loaded_heightmap / np.max(loaded_heightmap),
            self.height_map / np.max(self.height_map),
            atol=1e-5
        ))
    
    def test_save_and_load_npz(self):
        """Test saving and loading heightmap as NPZ format.."""
        # Save heightmap as NPZ
        output_path = os.path.join(self.temp_dir.name, "test_heightmap.npz")
        saved_path = save_heightmap(self.height_map, output_path, normalize=True)
        
        # Check if file exists
        self.assertTrue(os.path.exists(saved_path))
        
        # Load the saved heightmap
        loaded_heightmap = load_heightmap(saved_path, normalize=False)
        
        # Check if loaded heightmap has the same shape
        self.assertEqual(loaded_heightmap.shape, self.height_map.shape)
        
        # Check if values are similar (allowing for small differences due to normalization)
        self.assertTrue(np.allclose(
            loaded_heightmap / np.max(loaded_heightmap),
            self.height_map / np.max(self.height_map),
            atol=1e-5
        ))

    def test_save_and_load_png(self):
        """Test saving and loading heightmap as PNG format.."""
        try:
            # Try to save as PNG (requires opencv or PIL)
            output_path = os.path.join(self.temp_dir.name, "test_heightmap.png")
            saved_path = save_heightmap(self.height_map, output_path, normalize=True)
            
            # Check if file exists
            self.assertTrue(os.path.exists(saved_path))
            
            # Load the saved heightmap
            loaded_heightmap = load_heightmap(saved_path, normalize=False)
            
            # Check if loaded heightmap has the same shape
            self.assertEqual(loaded_heightmap.shape, self.height_map.shape)
            
            # PNG is lossy for float data (8-bit), so we use a higher tolerance
            # We're mainly checking that the pattern is preserved
            pattern_correlation = np.corrcoef(
                loaded_heightmap.flatten(),
                self.height_map.flatten()
            )[0, 1]
            
            self.assertGreater(pattern_correlation, 0.9)
        except ImportError:
            # Skip if opencv or PIL is not available
            self.skipTest("OpenCV or PIL is required for PNG testing")

if __name__ == "__main__":
    unittest.main()
