#!/usr/bin/env python3
"""
Tests for TMD processing functions.

This module contains unit tests for the surface processing utilities
that handle height map manipulations.
"""

import os
import numpy as np
import pytest
from unittest import mock

# Import the modules to test - adjust these imports to match your project structure
# The current error suggests these functions may be in 'processing.py' not 'transformations.py'
from tmd.surface.processing import (
    crop_height_map,
    flip_height_map,
    rotate_height_map,
    threshold_height_map,
    extract_cross_section,
    extract_profile_at_percentage
)


class TestHeightMapProcessing:
    """Test suite for height map processing utilities."""

    def setup_method(self):
        """Set up test data for each test."""
        # Create a sample height map with a gradient pattern
        rows, cols = 5, 5
        self.height_map = np.zeros((rows, cols), dtype=np.float32)
        for r in range(rows):
            for c in range(cols):
                self.height_map[r, c] = r * 0.1 + c * 0.01
        
        # Add a recognizable max value at a specific location for testing
        self.height_map[0, 4] = 1.0  # max value at top-right
        
        # Create sample metadata dict
        self.data_dict = {
            'width': cols,
            'height': rows,
            'x_length': 10.0,
            'y_length': 10.0,
            'x_offset': 0.0,
            'y_offset': 0.0
        }

    def test_crop_height_map(self):
        """Test cropping a height map."""
        # Test normal cropping
        region = (1, 3, 1, 4)  # (min_row, max_row, min_col, max_col)
        cropped = crop_height_map(self.height_map, region)
        
        # Check dimensions
        assert cropped.shape == (2, 3)
        
        # Check that values match the original region
        np.testing.assert_array_equal(cropped, self.height_map[1:3, 1:4])
        
        # Test that a copy was made (not a view)
        cropped[0, 0] = 99.9
        assert self.height_map[1, 1] != 99.9
        
        # Test error cases
        # Negative start index
        with pytest.raises(ValueError):
            crop_height_map(self.height_map, (-1, 3, 1, 4))
        
        # End index smaller than start index
        with pytest.raises(ValueError):
            crop_height_map(self.height_map, (3, 1, 1, 4))
        
        # End index out of bounds
        with pytest.raises(ValueError):
            crop_height_map(self.height_map, (1, 10, 1, 4))

    def test_flip_height_map(self):
        """Test flipping a height map."""
        # Test horizontal flip (axis=0)
        flipped_h = flip_height_map(self.height_map, 0)
        
        # Check that dimensions match
        assert flipped_h.shape == self.height_map.shape
        
        # Check that max value moved from top-right to top-left
        max_position = np.unravel_index(np.argmax(flipped_h), flipped_h.shape)
        assert max_position[0] == 0  # Still in top row
        assert max_position[1] == 0  # Now in leftmost column
        
        # Test vertical flip (axis=1)
        flipped_v = flip_height_map(self.height_map, 1)
        
        # Check that dimensions match
        assert flipped_v.shape == self.height_map.shape
        
        # Check that max value moved from top-right to bottom-right
        max_position = np.unravel_index(np.argmax(flipped_v), flipped_v.shape)
        assert max_position[0] == 4  # Now in bottom row
        assert max_position[1] == 4  # Still in rightmost column
        
        # Test invalid axis
        with pytest.raises(ValueError):
            flip_height_map(self.height_map, 2)

    def test_rotate_height_map(self):
        """Test rotating a height map."""
        # Test 90-degree rotation
        rotated_90 = rotate_height_map(self.height_map, 90)
        
        # Check shape (should be preserved if reshape=True)
        assert rotated_90.shape == self.height_map.shape
        
        # Check that max value position changed correctly
        max_position = np.unravel_index(np.argmax(rotated_90), rotated_90.shape)
        # After 90° rotation, max should move toward the bottom-left region
        # Exact position may vary due to interpolation, so we check a range
        assert max_position[0] >= 2  # Should be in lower half
        
        # Test 180-degree rotation
        rotated_180 = rotate_height_map(self.height_map, 180)
        
        # Check shape
        assert rotated_180.shape == self.height_map.shape
        
        # Max value should move approximately from top-right to bottom-left
        max_position = np.unravel_index(np.argmax(rotated_180), rotated_180.shape)
        assert max_position[0] >= 3  # Should be in bottom rows
        assert max_position[1] <= 1  # Should be in left columns
        
        # Test rotation without reshaping
        rotated_no_reshape = rotate_height_map(self.height_map, 45, reshape=False)
        assert rotated_no_reshape.shape == self.height_map.shape

    def test_threshold_height_map(self):
        """Test thresholding a height map."""
        # Test lower threshold with clipping
        lower_threshold = 0.3
        thresholded_lower = threshold_height_map(self.height_map, min_height=lower_threshold)
        
        # Check that values below threshold were clipped
        assert np.all(thresholded_lower >= lower_threshold)
        
        # Check that values above threshold were unchanged
        mask_above = self.height_map >= lower_threshold
        np.testing.assert_array_equal(thresholded_lower[mask_above], self.height_map[mask_above])
        
        # Test upper threshold with clipping
        upper_threshold = 0.5
        thresholded_upper = threshold_height_map(self.height_map, max_height=upper_threshold)
        
        # Check that values above threshold were clipped
        assert np.all(thresholded_upper <= upper_threshold)
        
        # Test replacement value
        replacement = -999.0
        thresholded_replace = threshold_height_map(
            self.height_map, min_height=0.3, max_height=0.6, replacement=replacement
        )
        
        # Check that values outside the range were replaced
        mask_outside = (self.height_map < 0.3) | (self.height_map > 0.6)
        assert np.all(thresholded_replace[mask_outside] == replacement)

    def test_extract_cross_section(self):
        """Test extracting cross-sections from a height map."""
        # Test horizontal cross-section (X-axis)
        positions_x, heights_x = extract_cross_section(
            self.height_map, self.data_dict, axis="x", position=2
        )
        
        # Check lengths
        assert len(positions_x) == self.height_map.shape[1]
        assert len(heights_x) == self.height_map.shape[1]
        
        # Check that heights match the expected row
        np.testing.assert_array_equal(heights_x, self.height_map[2, :])
        
        # Test vertical cross-section (Y-axis)
        positions_y, heights_y = extract_cross_section(
            self.height_map, self.data_dict, axis="y", position=3
        )
        
        # Check lengths
        assert len(positions_y) == self.height_map.shape[0]
        assert len(heights_y) == self.height_map.shape[0]
        
        # Check that heights match the expected column
        np.testing.assert_array_equal(heights_y, self.height_map[:, 3])
        
        # Test custom cross-section
        start_point = (1, 1)
        end_point = (3, 3)
        positions_custom, heights_custom = extract_cross_section(
            self.height_map, 
            self.data_dict, 
            axis="custom", 
            start_point=start_point, 
            end_point=end_point
        )
        
        # Check that we got some values
        assert len(positions_custom) > 0
        assert len(heights_custom) > 0
        
        # Test error cases
        # Invalid axis
        with pytest.raises(ValueError):
            extract_cross_section(self.height_map, self.data_dict, axis="z")

    def test_extract_profile_at_percentage(self):
        """Test extracting profiles at specified percentages."""
        # Test X-axis profile at 50%
        profile_x_50 = extract_profile_at_percentage(
            self.height_map, self.data_dict, axis="x", percentage=50.0
        )
        
        # Check length
        assert len(profile_x_50) == self.height_map.shape[1]
        
        # Check that it matches the middle row
        middle_row = self.height_map.shape[0] // 2
        np.testing.assert_array_equal(profile_x_50, self.height_map[middle_row, :])
        
        # Test Y-axis profile at 25%
        profile_y_25 = extract_profile_at_percentage(
            self.height_map, self.data_dict, axis="y", percentage=25.0
        )
        
        # Check length
        assert len(profile_y_25) == self.height_map.shape[0]
        
        # Mock np.save to avoid creating files during test
        with mock.patch("numpy.save") as mock_save:
            with mock.patch("builtins.print") as mock_print:
                profile_save = extract_profile_at_percentage(
                    self.height_map, 
                    self.data_dict, 
                    axis="x", 
                    percentage=75.0,
                    save_path="test_profile.npy"
                )
                
                # Check that save was attempted
                mock_save.assert_called_once()


if __name__ == "__main__":
    pytest.main(["-v", __file__])