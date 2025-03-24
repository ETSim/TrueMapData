""".

Unit tests for STL model export functionality.
"""

import os
import unittest
import tempfile
import numpy as np

from tmd.exporters.model.stl import (
    convert_heightmap_to_stl,
    convert_heightmap_to_stl_streamed,
    convert_heightmap_to_stl_threaded
)

class TestSTLExporter(unittest.TestCase):
    """Test cases for STL export functionality.."""
    
    def setUp(self):
        """Set up test data.."""
        # Create a simple heightmap for testing
        self.height_map = np.zeros((50, 50), dtype=np.float32)
        
        # Create a simple heightfield pattern
        x, y = np.meshgrid(np.linspace(-1, 1, 50), np.linspace(-1, 1, 50))
        r = np.sqrt(x**2 + y**2)
        self.height_map = np.exp(-r**2 * 4)
        
        # Create a temporary directory for test outputs
        self.temp_dir = tempfile.TemporaryDirectory()
    
    def tearDown(self):
        """Clean up temporary files.."""
        self.temp_dir.cleanup()
    
    def test_convert_heightmap_to_stl_binary(self):
        """Test converting heightmap to binary STL.."""
        output_file = os.path.join(self.temp_dir.name, "binary.stl")
        result = convert_heightmap_to_stl(
            self.height_map,
            filename=output_file,
            z_scale=1.0,
            base_height=0.1,
            ascii=False  # Binary output
        )
        
        # Check that the file was created
        self.assertIsNotNone(result)
        self.assertTrue(os.path.exists(output_file))
        
        # Check file size is reasonable
        file_size = os.path.getsize(output_file)
        self.assertGreater(file_size, 1000)  # Should be at least 1KB
    
    def test_convert_heightmap_to_stl_ascii(self):
        """Test converting heightmap to ASCII STL.."""
        output_file = os.path.join(self.temp_dir.name, "ascii.stl")
        result = convert_heightmap_to_stl(
            self.height_map,
            filename=output_file,
            z_scale=1.0,
            base_height=0.1,
            ascii=True  # ASCII output
        )
        
        # Check that the file was created
        self.assertIsNotNone(result)
        self.assertTrue(os.path.exists(output_file))
        
        # Check file size is reasonable
        file_size = os.path.getsize(output_file)
        self.assertGreater(file_size, 1000)  # Should be at least 1KB
        
        # Check if the file starts with "solid" (ASCII STL header)
        with open(output_file, 'r') as f:
            first_line = f.readline().strip()
            self.assertTrue(first_line.startswith("solid"))
    
    def test_convert_heightmap_to_stl_with_base(self):
        """Test converting heightmap to STL with a base.."""
        output_file = os.path.join(self.temp_dir.name, "with_base.stl")
        result = convert_heightmap_to_stl(
            self.height_map,
            filename=output_file,
            z_scale=1.0,
            base_height=0.5  # Add a substantial base
        )
        
        # Check that the file was created
        self.assertIsNotNone(result)
        self.assertTrue(os.path.exists(output_file))
        
        # Check file size - with base should be larger than without
        file_size_with_base = os.path.getsize(output_file)
        
        # Create a version without base for comparison
        output_file_no_base = os.path.join(self.temp_dir.name, "no_base.stl")
        convert_heightmap_to_stl(
            self.height_map,
            filename=output_file_no_base,
            z_scale=1.0,
            base_height=0.0  # No base
        )
        file_size_no_base = os.path.getsize(output_file_no_base)
        
        # The version with base should be larger
        self.assertGreater(file_size_with_base, file_size_no_base)
    
    def test_convert_heightmap_to_stl_adaptive(self):
        """Test converting heightmap to STL with adaptive meshing.."""
        output_file = os.path.join(self.temp_dir.name, "adaptive.stl")
        result = convert_heightmap_to_stl(
            self.height_map,
            filename=output_file,
            z_scale=1.0,
            adaptive=True,
            max_subdivisions=6,
            error_threshold=0.01
        )
        
        # Check that the file was created
        self.assertIsNotNone(result)
        self.assertTrue(os.path.exists(output_file))
        
        # File size check - adaptive should typically be smaller than standard for simple geometry
        file_size_adaptive = os.path.getsize(output_file)
        self.assertGreater(file_size_adaptive, 1000)  # Should be at least 1KB
    
    def test_convert_heightmap_to_stl_streamed(self):
        """Test converting heightmap to STL using streamed output.."""
        # Skip if function is not implemented
        try:
            output_file = os.path.join(self.temp_dir.name, "streamed.stl")
            result = convert_heightmap_to_stl_streamed(
                self.height_map,
                filename=output_file,
                z_scale=1.0,
                chunk_size=10  # Small chunks for testing
            )
            
            # Check that the file was created
            self.assertIsNotNone(result)
            self.assertTrue(os.path.exists(output_file))
            
            # Check file size is reasonable
            file_size = os.path.getsize(output_file)
            self.assertGreater(file_size, 1000)  # Should be at least 1KB
        except (NotImplementedError, AttributeError):
            self.skipTest("Streamed STL conversion not implemented")

if __name__ == "__main__":
    unittest.main()
