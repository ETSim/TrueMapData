"""Unit tests for TMD bump map module."""

import unittest
from unittest.mock import patch, MagicMock
import os
import numpy as np
import tempfile
import shutil
from PIL import Image

from tmd.exporters.image.bump_map import convert_heightmap_to_bump_map
from tests.resources import create_sample_height_map


class TestBumpMap(unittest.TestCase):
    """Test class for bump map functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample height maps for testing
        self.sample_height_map = create_sample_height_map(pattern="peak")
        self.sample_height_map_with_nan = create_sample_height_map(pattern="with_nan")
        
        # Test file path
        self.output_file = os.path.join(self.temp_dir, "bump_map.png")
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary directory and its contents
        shutil.rmtree(self.temp_dir)
    
    def test_convert_heightmap_to_bump_map_basic(self):
        """Test basic conversion of height map to bump map."""
        # Convert height map to bump map without saving
        result = convert_heightmap_to_bump_map(
            self.sample_height_map,
            filename=None,
            strength=1.0,
            blur_radius=1.0
        )
        
        # Verify the result is a PIL Image
        self.assertIsInstance(result, Image.Image)
        
        # Verify the image dimensions
        self.assertEqual(result.size, (5, 5))
    
    def test_convert_heightmap_to_bump_map_with_file(self):
        """Test conversion of height map to bump map with file output."""
        # Convert height map to bump map and save to file
        result = convert_heightmap_to_bump_map(
            self.sample_height_map,
            filename=self.output_file,
            strength=1.0,
            blur_radius=1.0
        )
        
        # Verify the file was created
        self.assertTrue(os.path.exists(self.output_file))
        
        # Verify the result is a PIL Image
        self.assertIsInstance(result, Image.Image)
    
    def test_convert_heightmap_to_bump_map_with_nan(self):
        """Test handling of NaN values in height map."""
        # Convert height map with NaN values to bump map
        result = convert_heightmap_to_bump_map(
            self.sample_height_map_with_nan,
            filename=None,
            strength=1.0,
            blur_radius=1.0
        )
        
        # Verify the result is a PIL Image
        self.assertIsInstance(result, Image.Image)
        
        # Convert to array to verify values
        bump_array = np.array(result)
        
        # Verify no invalid values in the output
        self.assertFalse(np.any(np.isnan(bump_array)))
    
    def test_convert_heightmap_to_bump_map_strength(self):
        """Test the effect of strength parameter."""
        # Create bump maps with different strength values
        result_low = convert_heightmap_to_bump_map(
            self.sample_height_map,
            filename=None,
            strength=0.1,  # Much lower strength value for clearer difference
            blur_radius=0.0  # No blur for cleaner comparison
        )
        
        result_high = convert_heightmap_to_bump_map(
            self.sample_height_map,
            filename=None,
            strength=5.0,  # Much higher strength value for clearer difference
            blur_radius=0.0  # No blur for cleaner comparison
        )
        
        # Convert to arrays for comparison
        low_array = np.array(result_low)
        high_array = np.array(result_high)
        
        # Because normalization might make means similar, check specific pixels or patterns
        # Use standard deviation instead, which should be affected by strength
        self.assertNotEqual(
            np.std(low_array), 
            np.std(high_array), 
            "Different strength values should produce different results"
        )
    
    def test_convert_heightmap_to_bump_map_blur(self):
        """Test the effect of blur radius parameter."""
        # Create bump maps with different blur values
        result_no_blur = convert_heightmap_to_bump_map(
            self.sample_height_map,
            filename=None,
            strength=1.0,
            blur_radius=0.0
        )
        
        result_with_blur = convert_heightmap_to_bump_map(
            self.sample_height_map,
            filename=None,
            strength=1.0,
            blur_radius=2.0
        )
        
        # Convert to arrays for comparison
        no_blur_array = np.array(result_no_blur)
        with_blur_array = np.array(result_with_blur)
        
        # The arrays should be different due to blurring
        # Use standard deviation which should be lower in the blurred image
        self.assertNotEqual(np.std(no_blur_array), np.std(with_blur_array))
    
    def test_directory_creation(self):
        """Test that the directory is created if it doesn't exist."""
        # Set up a path that would require directory creation
        nested_path = os.path.join(self.temp_dir, "nested", "dir", "bump_map.png")
        
        try:
            # Convert height map to bump map and save to file
            convert_heightmap_to_bump_map(
                self.sample_height_map,
                filename=nested_path,
                strength=1.0,
                blur_radius=1.0
            )
            
            # Verify directory and file exist
            self.assertTrue(os.path.exists(os.path.dirname(nested_path)))
            self.assertTrue(os.path.exists(nested_path))
        except Exception as e:
            self.fail(f"Failed to create directory or save file: {e}")


if __name__ == '__main__':
    unittest.main()
