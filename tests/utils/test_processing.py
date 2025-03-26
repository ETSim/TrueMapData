"""Unit tests for TMD processing utility module."""

import unittest
import numpy as np
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock

from tmd.utils.processing import (
    crop_height_map,
    flip_height_map,
    rotate_height_map,
    threshold_height_map,
    extract_cross_section,
    extract_profile_at_percentage
)


class TestProcessingUtility(unittest.TestCase):
    """Test class for processing utility functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test height map data
        self.height_map = np.zeros((10, 10), dtype=np.float32)
        # Create a simple pattern: increasing values from top-left to bottom-right
        for i in range(10):
            for j in range(10):
                self.height_map[i, j] = (i + j) / 18.0  # Normalize to [0,1] range
        
        # Create a more complex test height map with a central peak
        x = np.linspace(-3, 3, 10)
        y = np.linspace(-3, 3, 10)
        X, Y = np.meshgrid(x, y)
        self.peak_map = np.exp(-(X**2 + Y**2)/3)
        
        # Set up temporary directory for file output tests
        self.temp_dir = tempfile.mkdtemp()
        
        # Create metadata dictionary for testing
        self.metadata = {
            'width': 10,
            'height': 10,
            'x_length': 10.0,
            'y_length': 10.0,
            'x_offset': 0.0,
            'y_offset': 0.0
        }
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_crop_height_map(self):
        """Test cropping a height map."""
        # Test valid cropping
        cropped = crop_height_map(self.height_map, (2, 7, 3, 8))
        
        # Check dimensions
        self.assertEqual(cropped.shape, (5, 5))
        
        # Check content (first and last values should match original at crop points)
        self.assertEqual(cropped[0, 0], self.height_map[2, 3])
        self.assertEqual(cropped[4, 4], self.height_map[6, 7])
        
        # Test error cases
        with self.assertRaises(ValueError):
            # Invalid region with negative coordinates
            crop_height_map(self.height_map, (-1, 5, 2, 7))
            
        with self.assertRaises(ValueError):
            # End coordinates less than start
            crop_height_map(self.height_map, (5, 2, 3, 8))
            
        with self.assertRaises(ValueError):
            # End coordinates exceed array dimensions
            crop_height_map(self.height_map, (2, 12, 3, 8))
    
    def test_flip_height_map(self):
        """Test flipping a height map along different axes."""
        # Test horizontal flip (axis=0)
        flipped_h = flip_height_map(self.height_map, 0)
        
        # In a horizontal flip, rows are reversed but columns stay the same
        for i in range(10):
            for j in range(10):
                self.assertEqual(flipped_h[i, j], self.height_map[9-i, j])
        
        # Test vertical flip (axis=1)
        flipped_v = flip_height_map(self.height_map, 1)
        
        # In a vertical flip, columns are reversed but rows stay the same
        for i in range(10):
            for j in range(10):
                self.assertEqual(flipped_v[i, j], self.height_map[i, 9-j])
        
        # Test invalid axis
        with self.assertRaises(ValueError):
            flip_height_map(self.height_map, 2)  # Only 0 and 1 are valid
    
    def test_rotate_height_map(self):
        """Test rotating a height map by different angles."""
        # Test 90 degree rotation
        rotated_90 = rotate_height_map(self.height_map, 90.0)
        
        # Create a small test pattern that's easy to verify when rotated
        test_map = np.zeros((5, 5), dtype=np.float32)
        test_map[0, 2] = 1.0  # Top-middle value
        
        # Rotate test_map 90 degrees
        rotated_test = rotate_height_map(test_map, 90.0)
        
        # The value should now be in middle-left position (approximately)
        # Allow small decimal differences due to interpolation
        max_pos = np.unravel_index(np.argmax(rotated_test), rotated_test.shape)
        self.assertEqual(max_pos[1], 0)  # Left edge
        self.assertIn(max_pos[0], [2, 3])  # Middle row approximately
        
        # Test 180 degree rotation
        rotated_180 = rotate_height_map(self.height_map, 180.0)
        
        # Value at [0,0] should now be at [9,9], etc.
        # Allow small decimal differences due to interpolation
        self.assertAlmostEqual(rotated_180[9, 9], self.height_map[0, 0], places=5)
        self.assertAlmostEqual(rotated_180[0, 0], self.height_map[9, 9], places=5)
        
        # Test with reshape=False (output keeps same dimensions)
        rotated_45_no_reshape = rotate_height_map(self.height_map, 45.0, reshape=False)
        self.assertEqual(rotated_45_no_reshape.shape, self.height_map.shape)
    
    def test_threshold_height_map(self):
        """Test applying thresholds to height map values."""
        # Test minimum threshold with default clipping
        min_threshold = 0.3
        thresholded_min = threshold_height_map(self.height_map, min_height=min_threshold)
        
        # All values should be >= min_threshold
        self.assertTrue(np.all(thresholded_min >= min_threshold))
        
        # Values that were already above threshold should remain unchanged
        original_above_threshold = self.height_map[self.height_map >= min_threshold]
        thresholded_above = thresholded_min[self.height_map >= min_threshold]
        np.testing.assert_array_equal(original_above_threshold, thresholded_above)
        
        # Test maximum threshold with replacement
        max_threshold = 0.7
        replacement = -1.0
        thresholded_max_replace = threshold_height_map(
            self.height_map, max_height=max_threshold, replacement=replacement
        )
        
        # Values above max_threshold should be replaced
        self.assertTrue(np.all(thresholded_max_replace[self.height_map > max_threshold] == replacement))
        
        # Values below max_threshold should remain unchanged
        original_below_threshold = self.height_map[self.height_map <= max_threshold]
        thresholded_below = thresholded_max_replace[self.height_map <= max_threshold]
        np.testing.assert_array_equal(original_below_threshold, thresholded_below)
        
        # Test combined min and max thresholds
        thresholded_both = threshold_height_map(
            self.height_map, min_height=0.2, max_height=0.8
        )
        self.assertTrue(np.all(thresholded_both >= 0.2))
        self.assertTrue(np.all(thresholded_both <= 0.8))
    
    def test_extract_cross_section_x(self):
        """Test extracting horizontal cross-sections."""
        # Extract middle row
        positions, heights = extract_cross_section(
            self.height_map, self.metadata, axis='x', position=5
        )
        
        # Check that heights match the 5th row of the height map
        np.testing.assert_array_equal(heights, self.height_map[5, :])
        
        # Check positions are generated correctly
        self.assertEqual(len(positions), self.height_map.shape[1])
        self.assertEqual(positions[0], self.metadata['x_offset'])
        self.assertEqual(positions[-1], self.metadata['x_offset'] + self.metadata['x_length'])
    
    def test_extract_cross_section_y(self):
        """Test extracting vertical cross-sections."""
        # Extract middle column
        positions, heights = extract_cross_section(
            self.height_map, self.metadata, axis='y', position=5
        )
        
        # Check that heights match the 5th column of the height map
        np.testing.assert_array_equal(heights, self.height_map[:, 5])
        
        # Check positions are generated correctly
        self.assertEqual(len(positions), self.height_map.shape[0])
        self.assertEqual(positions[0], self.metadata['y_offset'])
        self.assertEqual(positions[-1], self.metadata['y_offset'] + self.metadata['y_length'])
    
    def test_extract_cross_section_custom(self):
        """Test extracting custom cross-sections."""
        # Extract diagonal cross-section
        start_point = (2, 3)
        end_point = (7, 8)
        positions, heights = extract_cross_section(
            self.height_map, self.metadata, axis='custom', 
            start_point=start_point, end_point=end_point
        )
        
        # Check that we have the expected number of points
        self.assertGreaterEqual(len(positions), 
                              max(abs(end_point[0] - start_point[0]), 
                                  abs(end_point[1] - start_point[1])))
        
        # Check position range
        self.assertEqual(positions[0], 0.0)  # Should start at 0
        
        # Test errors for invalid parameters
        with self.assertRaises(ValueError):
            # Invalid axis
            extract_cross_section(self.height_map, self.metadata, axis='invalid')
            
        with self.assertRaises(ValueError):
            # Missing start/end points for custom axis
            extract_cross_section(self.height_map, self.metadata, axis='custom')
    
    def test_extract_profile_at_percentage(self):
        """Test extracting profiles at different percentage positions."""
        # Test horizontal profile at 25%
        profile_x_25 = extract_profile_at_percentage(
            self.height_map, self.metadata, axis='x', percentage=25
        )
        
        # Should match row at 25% of height
        row_idx = int(0.25 * (self.height_map.shape[0] - 1) + 0.5)
        np.testing.assert_array_equal(profile_x_25, self.height_map[row_idx, :])
        
        # Test vertical profile at 75%
        profile_y_75 = extract_profile_at_percentage(
            self.height_map, self.metadata, axis='y', percentage=75
        )
        
        # Should match column at 75% of width
        col_idx = int(0.75 * (self.height_map.shape[1] - 1) + 0.5)
        np.testing.assert_array_equal(profile_y_75, self.height_map[:, col_idx])
        
        # Test saving to file
        output_path = os.path.join(self.temp_dir, 'profile.npy')
        with patch('numpy.save') as mock_save:
            extract_profile_at_percentage(
                self.height_map, self.metadata, axis='x', percentage=50, 
                save_path=output_path
            )
            mock_save.assert_called_once()
        
        # Test invalid axis
        with self.assertRaises(ValueError):
            extract_profile_at_percentage(
                self.height_map, self.metadata, axis='invalid', percentage=50
            )


if __name__ == '__main__':
    unittest.main()
