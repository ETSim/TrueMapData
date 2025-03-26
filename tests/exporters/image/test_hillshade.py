"""Unit tests for TMD hillshade module."""

import unittest
from unittest.mock import patch, MagicMock
import os
import numpy as np
import tempfile
import shutil
from PIL import Image

from tmd.exporters.image.hillshade import generate_hillshade, export_hillshade
from tests.resources import create_sample_height_map


class TestHillshade(unittest.TestCase):
    """Test class for hillshade functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample height maps for testing
        self.sample_height_map = create_sample_height_map(pattern="peak")
        self.sample_height_map_with_nan = create_sample_height_map(pattern="with_nan")
        
        # Test file path
        self.output_file = os.path.join(self.temp_dir, "hillshade.png")
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary directory and its contents
        shutil.rmtree(self.temp_dir)
    
    def test_generate_hillshade_basic(self):
        """Test basic generation of hillshade from heightmap."""
        # Generate hillshade without saving
        result = generate_hillshade(
            self.sample_height_map,
            filename=None,
            altitude=45,
            azimuth=315,
            z_factor=1.0
        )
        
        # Verify the result is a PIL Image
        self.assertIsInstance(result, Image.Image)
        
        # Verify the image dimensions
        self.assertEqual(result.size, (5, 5))
    
    def test_generate_hillshade_with_file(self):
        """Test generation of hillshade with file output."""
        result = generate_hillshade(
            self.sample_height_map,
            filename=self.output_file
        )
        
        # Verify the file was created
        self.assertTrue(os.path.exists(self.output_file))
        
        # Verify the result is a PIL Image
        self.assertIsInstance(result, Image.Image)
    
    def test_generate_hillshade_with_nan(self):
        """Test handling of NaN values in heightmap."""
        # Generate hillshade from heightmap with NaN values
        result = generate_hillshade(
            self.sample_height_map_with_nan,
            filename=None
        )
        
        # Verify the result is a PIL Image
        self.assertIsInstance(result, Image.Image)
        
        # Convert to array to verify values
        hillshade_array = np.array(result)
        
        # Verify no invalid values in the output
        self.assertFalse(np.any(np.isnan(hillshade_array)))
    
    def test_generate_hillshade_parameters(self):
        """Test the effect of different parameters on hillshade."""
        # Generate hillshades with different parameters
        result1 = generate_hillshade(
            self.sample_height_map,
            filename=None,
            altitude=30,
            azimuth=45,
            z_factor=1.0
        )
        
        result2 = generate_hillshade(
            self.sample_height_map,
            filename=None,
            altitude=60,
            azimuth=225,
            z_factor=2.0
        )
        
        # Convert to arrays for comparison
        array1 = np.array(result1)
        array2 = np.array(result2)
        
        # The arrays should be different due to different parameters
        self.assertFalse(np.array_equal(array1, array2))
    
    @patch('matplotlib.pyplot.imsave')
    def test_export_hillshade(self, mock_imsave):
        """Test exporting hillshade with matplotlib."""
        # Mock imsave to avoid actual file creation
        mock_imsave.return_value = None
        
        # Call export function
        result = export_hillshade(
            self.sample_height_map,
            self.output_file,
            altitude=45,
            azimuth=315,
            z_factor=1.0,
            cmap='terrain'
        )
        
        # Check if imsave was called with right parameters
        mock_imsave.assert_called_once()
        args, kwargs = mock_imsave.call_args
        self.assertEqual(args[0], self.output_file)
        self.assertEqual(kwargs.get('cmap'), 'terrain')
        
        # Check return value
        self.assertEqual(result, self.output_file)
    
    def test_directory_creation(self):
        """Test creating output directory if it doesn't exist."""
        # Set up nested path
        nested_path = os.path.join(self.temp_dir, "nested", "dir", "hillshade.png")
        
        # Generate hillshade
        result = generate_hillshade(
            self.sample_height_map,
            filename=nested_path
        )
        
        # Verify directory was created
        self.assertTrue(os.path.exists(os.path.dirname(nested_path)))
        
        # Verify file was created
        self.assertTrue(os.path.exists(nested_path))


if __name__ == '__main__':
    unittest.main()
