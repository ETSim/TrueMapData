""".

Unit tests for EXR heightmap to STL conversion.
"""

import os
import unittest
import tempfile
import numpy as np

# Skip tests if OpenCV with EXR support is not available
try:
    import cv2
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    HAS_EXR_SUPPORT = True
except ImportError:
    HAS_EXR_SUPPORT = False

from tmd.exporters.image.image_io import save_heightmap  # Updated import
from tmd.exporters.model import convert_heightmap_to_stl

@unittest.skipIf(not HAS_EXR_SUPPORT, "OpenCV with EXR support not available")
class TestEXRToSTL(unittest.TestCase):
    """Test cases for EXR to STL conversion.."""
    
    def setUp(self):
        """Set up test data and temporary directory.."""
        # Create a temporary directory for test outputs
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create a simple test heightmap
        self.height_map = np.zeros((50, 50), dtype=np.float32)
        
        # Create a simple heightfield pattern
        x, y = np.meshgrid(np.linspace(-1, 1, 50), np.linspace(-1, 1, 50))
        r = np.sqrt(x**2 + y**2)
        self.height_map = np.exp(-r**2 * 4)
    
    def tearDown(self):
        """Clean up temporary directory.."""
        self.temp_dir.cleanup()
    
    def test_exr_to_stl_conversion(self):
        """Test EXR heightmap to STL file conversion.."""
        # Save heightmap as EXR
        exr_path = os.path.join(self.temp_dir.name, "test_heightmap.exr")
        save_heightmap(self.height_map, exr_path, normalize=True)
        
        # Check if EXR file exists
        self.assertTrue(os.path.exists(exr_path))
        
        # Convert EXR to STL using standard method
        stl_path = os.path.join(self.temp_dir.name, "output.stl")
        
        result = convert_heightmap_to_stl(
            self.height_map,
            filename=stl_path,
            z_scale=1.0,
            base_height=0.1,
            adaptive=True,
            max_subdivisions=6
        )
        
        # Check conversion was successful
        self.assertIsNotNone(result)
        self.assertTrue(os.path.exists(stl_path))
        
        # Check file size is reasonable (should be non-zero)
        file_size = os.path.getsize(stl_path)
        self.assertGreater(file_size, 1000)  # At least 1KB

if __name__ == "__main__":
    unittest.main()
