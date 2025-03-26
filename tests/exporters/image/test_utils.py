"""Unit tests for TMD image utils module."""

import unittest
from unittest.mock import patch, MagicMock
import os
import numpy as np
import tempfile
import shutil

from tmd.exporters.image.utils import (
    ensure_directory_exists,
    normalize_heightmap,
    handle_nan_values,
    array_to_image,
    save_image,
    apply_colormap,
    normalize_height_map
)


class TestImageUtils(unittest.TestCase):
    """Test class for image utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample arrays
        self.sample_array_float = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]
        ])
        
        self.sample_array_with_nan = np.array([
            [0.1, 0.2, np.nan],
            [0.4, 0.5, 0.6],
            [np.nan, 0.8, 0.9]
        ])
        
        # Test output file path
        self.output_file = os.path.join(self.temp_dir, "test_image.png")
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary directory and its contents
        shutil.rmtree(self.temp_dir)
    
    def test_ensure_directory_exists(self):
        """Test creating directory for a file."""
        # Create a path with nested directories
        nested_path = os.path.join(self.temp_dir, "a", "b", "c", "test.png")
        
        # Ensure directory exists
        ensure_directory_exists(nested_path)
        
        # Check that directory was created
        self.assertTrue(os.path.exists(os.path.dirname(nested_path)))
        
        # Test with existing directory (should not raise)
        ensure_directory_exists(nested_path)
    
    def test_normalize_heightmap(self):
        """Test normalizing height maps."""
        # Basic normalization
        normalized = normalize_heightmap(self.sample_array_float)
        
        # Check range
        self.assertAlmostEqual(np.min(normalized), 0.0)
        self.assertAlmostEqual(np.max(normalized), 1.0)
        
        # Test with custom range
        normalized_custom = normalize_heightmap(self.sample_array_float, vmin=0.2, vmax=0.8)
        
        # Values below vmin should be 0, above vmax should be 1
        self.assertAlmostEqual(normalized_custom[0, 0], 0.0)  # 0.1 is below vmin
        self.assertAlmostEqual(normalized_custom[2, 2], 1.0)  # 0.9 is above vmax
        
        # Test with all identical values
        flat_array = np.ones((3, 3))
        normalized_flat = normalize_heightmap(flat_array)
        
        # Should still produce valid output
        self.assertTrue(np.all(normalized_flat == 0) or np.all(normalized_flat == 1))
    
    def test_handle_nan_values(self):
        """Test handling NaN values in arrays."""
        # Replace NaNs with mean of non-NaN values
        fixed_array = handle_nan_values(self.sample_array_with_nan)
        
        # Check that no NaNs remain
        self.assertFalse(np.any(np.isnan(fixed_array)))
        
        # Values that were NaN should now be the mean of non-NaN values
        mean_value = np.nanmean(self.sample_array_with_nan)
        self.assertAlmostEqual(fixed_array[0, 2], mean_value)
        self.assertAlmostEqual(fixed_array[2, 0], mean_value)
        
        # Test with all NaN array
        all_nan = np.full((3, 3), np.nan)
        fixed_all_nan = handle_nan_values(all_nan)
        
        # Should fill with zeros
        self.assertTrue(np.all(fixed_all_nan == 0))
    
    def test_array_to_image(self):
        """Test converting arrays to image format."""
        # 8-bit conversion
        img_8bit = array_to_image(self.sample_array_float, bit_depth=8)
        
        # Check type and range
        self.assertEqual(img_8bit.dtype, np.uint8)
        self.assertTrue(np.all(img_8bit >= 0) and np.all(img_8bit <= 255))
        
        # Check specific values
        self.assertEqual(img_8bit[0, 0], 25)  # 0.1 * 255 = 25.5, rounded to 25
        
        # 16-bit conversion
        img_16bit = array_to_image(self.sample_array_float, bit_depth=16)
        
        # Check type and range
        self.assertEqual(img_16bit.dtype, np.uint16)
        self.assertTrue(np.all(img_16bit >= 0) and np.all(img_16bit <= 65535))
        
        # Check specific values
        self.assertEqual(img_16bit[0, 0], 6553)  # 0.1 * 65535 = 6553.5, rounded to 6553
    
    @patch('tmd.exporters.image.utils.HAS_OPENCV', False)
    @patch('tmd.exporters.image.utils.HAS_MATPLOTLIB', True)
    @patch('matplotlib.pyplot.imsave')
    def test_save_image_with_matplotlib(self, mock_imsave):
        """Test saving image with matplotlib."""
        # Save using matplotlib
        result = save_image(
            self.sample_array_float,
            self.output_file,
            cmap='viridis'
        )
        
        # Check that imsave was called correctly
        mock_imsave.assert_called_once()
        args, kwargs = mock_imsave.call_args
        self.assertEqual(args[0], self.output_file)
        self.assertEqual(kwargs.get('cmap'), 'viridis')
        
        # Check return value
        self.assertEqual(result, self.output_file)
    
    @patch('tmd.exporters.image.utils.HAS_OPENCV', True)
    @patch('cv2.imwrite')
    def test_save_image_with_opencv(self, mock_imwrite):
        """Test saving image with OpenCV."""
        # Save 16-bit image using OpenCV
        save_image(
            self.sample_array_float,
            self.output_file,
            bit_depth=16
        )
        
        # Check that imwrite was called
        mock_imwrite.assert_called_once()
        args, _ = mock_imwrite.call_args
        self.assertEqual(args[0], self.output_file)
        self.assertEqual(args[1].dtype, np.uint16)
    
    @patch('tmd.exporters.image.utils.HAS_MATPLOTLIB', True)
    @patch('matplotlib.pyplot.get_cmap')
    def test_apply_colormap(self, mock_get_cmap):
        """Test applying colormap to grayscale image."""
        # Set up mock colormap
        mock_cmap = MagicMock()
        mock_get_cmap.return_value = mock_cmap
        mock_cmap.return_value = np.zeros((3, 3, 4))  # Return a dummy RGBA image
        
        # Apply colormap
        result = apply_colormap(
            self.sample_array_float,
            colormap='viridis'
        )
        
        # Check that cmap was used
        mock_get_cmap.assert_called_once_with('viridis')
        
        # Check output shape and type
        self.assertEqual(result.shape, (3, 3, 3))  # RGB (alpha removed)
        self.assertEqual(result.dtype, np.uint8)
    
    @patch('tmd.exporters.image.utils.HAS_MATPLOTLIB', False)
    def test_apply_colormap_no_matplotlib(self):
        """Test apply_colormap with matplotlib unavailable."""
        with self.assertRaises(ImportError):
            apply_colormap(self.sample_array_float)
    
    def test_normalize_height_map(self):
        """Test normalize_height_map function."""
        # Basic normalization to default range (0-1)
        result = normalize_height_map(self.sample_array_float)
        self.assertAlmostEqual(np.min(result), 0.0)
        self.assertAlmostEqual(np.max(result), 1.0)
        
        # Custom range
        result = normalize_height_map(self.sample_array_float, min_val=-1.0, max_val=1.0)
        self.assertAlmostEqual(np.min(result), -1.0)
        self.assertAlmostEqual(np.max(result), 1.0)
        
        # Test with flat array
        flat_array = np.full((3, 3), 5.0)
        result = normalize_height_map(flat_array)
        self.assertEqual(np.mean(result), 0.5)  # Should be midpoint of range


if __name__ == '__main__':
    unittest.main()
