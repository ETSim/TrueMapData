"""
Unit tests for the height map processing functions.

These tests verify the functionality of processing operations like
crop, flip, rotate, threshold, and cross-section extraction.
"""
import os
import tempfile
import unittest

import numpy as np
from scipy import ndimage

from tmd.utils.processing import (
    crop_height_map,
    extract_cross_section,
    extract_profile_at_percentage,
    flip_height_map,
    rotate_height_map,
    threshold_height_map,
)


class TestCropHeightMap(unittest.TestCase):
    """Test the crop_height_map function."""

    def setUp(self):
        """Create a sample height map for testing."""
        self.height_map = np.ones((10, 10)) * 0.5
        # Set the outer border to 0 to verify correct cropping
        self.height_map[0, :] = 0
        self.height_map[-1, :] = 0
        self.height_map[:, 0] = 0
        self.height_map[:, -1] = 0
        # Set the inner region to a unique value
        self.height_map[2:8, 2:8] = 1.0

    def test_valid_crop(self):
        """Test cropping with valid parameters."""
        region = (2, 8, 2, 8)  # (row_start, row_end, col_start, col_end)
        cropped = crop_height_map(self.height_map, region)

        # Verify shape
        self.assertEqual(cropped.shape, (6, 6))
        # Verify the cropped region has the expected value
        self.assertTrue(np.all(cropped == 1.0))

    def test_invalid_crop_region(self):
        """Test that invalid crop regions raise ValueErrors."""
        # Starting coordinate negative
        with self.assertRaises(ValueError):
            crop_height_map(self.height_map, (-1, 5, 0, 5))

        # End coordinate beyond boundaries
        with self.assertRaises(ValueError):
            crop_height_map(self.height_map, (0, 11, 0, 5))

        # End before start
        with self.assertRaises(ValueError):
            crop_height_map(self.height_map, (5, 3, 0, 5))

    def test_edge_crop(self):
        """Test cropping to the very edges."""
        # Crop a thin slice from the top-left corner
        region = (0, 1, 0, 5)
        cropped = crop_height_map(self.height_map, region)
        self.assertEqual(cropped.shape, (1, 5))
        self.assertTrue(np.all(cropped == 0))  # Border value

    def test_single_pixel_crop(self):
        """Test cropping a single pixel."""
        region = (5, 6, 5, 6)
        cropped = crop_height_map(self.height_map, region)
        self.assertEqual(cropped.shape, (1, 1))
        self.assertEqual(cropped[0, 0], 1.0)  # Inner region value

    def test_copy_not_view(self):
        """Test that the result is a copy, not a view."""
        region = (2, 8, 2, 8)
        cropped = crop_height_map(self.height_map, region)

        # Modify the cropped array
        cropped[0, 0] = 9.9

        # Original should remain unchanged
        self.assertEqual(self.height_map[2, 2], 1.0)


class TestFlipHeightMap(unittest.TestCase):
    """Test the flip_height_map function."""

    def setUp(self):
        """Create a sample height map for testing."""
        # Create a gradient pattern to easily verify flips
        self.height_map = np.zeros((5, 10))
        for i in range(5):
            for j in range(10):
                self.height_map[i, j] = i * 10 + j

    def test_horizontal_flip(self):
        """Test horizontal flip (axis=0)."""
        flipped = flip_height_map(self.height_map, axis=0)

        # Verify shape remains the same
        self.assertEqual(flipped.shape, self.height_map.shape)

        # Check the flip worked - compare first row to last row flipped
        np.testing.assert_array_equal(flipped[0], self.height_map[4])
        np.testing.assert_array_equal(flipped[4], self.height_map[0])

    def test_vertical_flip(self):
        """Test vertical flip (axis=1)."""
        flipped = flip_height_map(self.height_map, axis=1)

        # Verify flipping along the columns
        for i in range(5):
            for j in range(10):
                self.assertEqual(flipped[i, j], self.height_map[i, 9 - j])

    def test_invalid_axis(self):
        """Test that invalid axis values raise ValueError."""
        with self.assertRaises(ValueError):
            flip_height_map(self.height_map, axis=2)

        with self.assertRaises(ValueError):
            flip_height_map(self.height_map, axis=-1)

    def test_copy_not_view(self):
        """Test that the result is a copy, not a view."""
        flipped = flip_height_map(self.height_map, axis=0)

        # Modify the flipped array
        original_value = self.height_map[0, 0]
        flipped[0, 0] = 999

        # Original should remain unchanged
        self.assertEqual(self.height_map[0, 0], original_value)


class TestRotateHeightMap(unittest.TestCase):
    """Test the rotate_height_map function."""

    def setUp(self):
        """Create a sample height map for testing."""
        # Create a simple pattern that's easy to verify after rotation
        self.height_map = np.zeros((5, 5), dtype=np.float32)
        self.height_map[2, 2] = 1.0  # Center point
        self.height_map[0, 2] = 0.5  # Top center

    def test_90_degree_rotation(self):
        """Test rotating by 90 degrees."""
        rotated = rotate_height_map(self.height_map, angle=90, reshape=False)

        # Center should stay the same
        self.assertEqual(rotated[2, 2], 1.0)

        # Top center should now be at right center (approximately)
        self.assertAlmostEqual(rotated[2, 4], 0.5, delta=0.1)

        # Check shape remains the same when reshape=False
        self.assertEqual(rotated.shape, self.height_map.shape)

    def test_45_degree_rotation_with_reshape(self):
        """Test rotating by 45 degrees with reshape=True."""
        rotated = rotate_height_map(self.height_map, angle=45, reshape=True)

        # Shape should be different when reshape=True
        self.assertNotEqual(rotated.shape, self.height_map.shape)

        # Check the rotated array contains the center value
        # Note: position changes due to reshape
        self.assertTrue(np.max(rotated) == 1.0)  # Center value should exist

    def test_rotation_interpolation_orders(self):
        """Test rotation with different interpolation orders."""
        # Create a test pattern
        test_map = np.zeros((10, 10))
        test_map[4:7, 4:7] = 1.0  # Small square in center

        # Rotate with nearest-neighbor interpolation
        rotated_nearest = rotate_height_map(test_map, angle=30, interpolation_order=0)

        # Rotate with bilinear interpolation
        rotated_bilinear = rotate_height_map(test_map, angle=30, interpolation_order=1)

        # The bilinear interpolation should have more intermediate values
        # than nearest neighbor
        unique_nearest = len(np.unique(rotated_nearest))
        unique_bilinear = len(np.unique(rotated_bilinear))
        self.assertGreater(unique_bilinear, unique_nearest)


class TestThresholdHeightMap(unittest.TestCase):
    """Test the threshold_height_map function."""

    def setUp(self):
        """Create a sample height map for testing."""
        # Create a gradient from 0 to 1
        self.height_map = np.linspace(0, 1, 100).reshape((10, 10))

    def test_min_threshold_only(self):
        """Test thresholding with only minimum value."""
        min_threshold = 0.5
        thresholded = threshold_height_map(self.height_map, min_height=min_threshold)

        # All values should be >= min_threshold
        self.assertTrue(np.all(thresholded >= min_threshold))

        # Values that were already >= min_threshold should remain unchanged
        original_above_threshold = self.height_map >= min_threshold
        np.testing.assert_array_equal(
            thresholded[original_above_threshold], self.height_map[original_above_threshold]
        )

    def test_max_threshold_only(self):
        """Test thresholding with only maximum value."""
        max_threshold = 0.7
        thresholded = threshold_height_map(self.height_map, max_height=max_threshold)

        # All values should be <= max_threshold
        self.assertTrue(np.all(thresholded <= max_threshold))

    def test_min_and_max_thresholds(self):
        """Test thresholding with both min and max values."""
        min_threshold = 0.3
        max_threshold = 0.7
        thresholded = threshold_height_map(
            self.height_map, min_height=min_threshold, max_height=max_threshold
        )

        # All values should be between min and max thresholds
        self.assertTrue(np.all(thresholded >= min_threshold))
        self.assertTrue(np.all(thresholded <= max_threshold))

    def test_threshold_with_replacement(self):
        """Test thresholding with replacement value."""
        min_threshold = 0.3
        max_threshold = 0.7
        replacement = -1.0

        thresholded = threshold_height_map(
            self.height_map,
            min_height=min_threshold,
            max_height=max_threshold,
            replacement=replacement,
        )

        # Values outside thresholds should be replaced
        original_below_min = self.height_map < min_threshold
        original_above_max = self.height_map > max_threshold

        self.assertTrue(np.all(thresholded[original_below_min] == replacement))
        self.assertTrue(np.all(thresholded[original_above_max] == replacement))

        # Values within thresholds should remain unchanged
        original_within_range = (self.height_map >= min_threshold) & (
            self.height_map <= max_threshold
        )
        np.testing.assert_array_equal(
            thresholded[original_within_range], self.height_map[original_within_range]
        )

    def test_copy_not_view(self):
        """Test that the result is a copy, not a view."""
        thresholded = threshold_height_map(self.height_map, min_height=0.5)

        # Modify the thresholded array
        original_value = self.height_map[0, 0]
        thresholded[0, 0] = 999

        # Original should remain unchanged
        self.assertEqual(self.height_map[0, 0], original_value)


class TestExtractCrossSection(unittest.TestCase):
    """Test the extract_cross_section function."""

    def setUp(self):
        """Create a sample height map and data dictionary for testing."""
        # Create a 10x10 sample height map with a recognizable pattern
        self.height_map = np.zeros((10, 10))

        # Create a diagonal ridge
        for i in range(10):
            self.height_map[i, i] = 1.0

        # Create a data dictionary with physical dimensions
        self.data_dict = {
            "width": 10,
            "height": 10,
            "x_offset": 0.0,
            "y_offset": 0.0,
            "x_length": 5.0,  # 5mm wide
            "y_length": 10.0,  # 10mm tall
        }

    def test_x_cross_section(self):
        """Test horizontal (X-axis) cross-section."""
        # Extract cross section at row 5
        positions, heights = extract_cross_section(
            self.height_map, self.data_dict, axis="x", position=5
        )

        # Check the number of points
        self.assertEqual(len(positions), 10)
        self.assertEqual(len(heights), 10)

        # Check that positions span from x_offset to x_offset + x_length
        self.assertAlmostEqual(positions[0], 0.0)
        self.assertAlmostEqual(positions[-1], 5.0)

        # Check that the height at diagonal position is 1.0
        self.assertEqual(heights[5], 1.0)

    def test_y_cross_section(self):
        """Test vertical (Y-axis) cross-section."""
        # Extract cross section at column 3
        positions, heights = extract_cross_section(
            self.height_map, self.data_dict, axis="y", position=3
        )

        # Check the number of points
        self.assertEqual(len(positions), 10)
        self.assertEqual(len(heights), 10)

        # Check that positions span from y_offset to y_offset + y_length
        self.assertAlmostEqual(positions[0], 0.0)
        self.assertAlmostEqual(positions[-1], 10.0)

        # Check that the height at diagonal position is 1.0
        self.assertEqual(heights[3], 1.0)

    def test_custom_cross_section(self):
        """Test custom cross-section along arbitrary line."""
        # Extract cross section from top-left to bottom-right
        positions, heights = extract_cross_section(
            self.height_map, self.data_dict, axis="custom", start_point=(0, 0), end_point=(9, 9)
        )

        # We should get more points than just the diagonal due to oversampling
        self.assertGreater(len(positions), 10)
        self.assertEqual(len(positions), len(heights))

        # Check that the heights contain values close to 1.0 (the diagonal)
        self.assertTrue(np.any(heights > 0.9))

    def test_invalid_cross_section(self):
        """Test that invalid parameters raise appropriate errors."""
        # Invalid axis
        with self.assertRaises(ValueError):
            extract_cross_section(self.height_map, self.data_dict, axis="z")

        # Invalid position
        with self.assertRaises(ValueError):
            extract_cross_section(self.height_map, self.data_dict, axis="x", position=20)

        # Missing custom points
        with self.assertRaises(ValueError):
            extract_cross_section(self.height_map, self.data_dict, axis="custom")

        # Out of bounds custom points
        with self.assertRaises(ValueError):
            extract_cross_section(
                self.height_map,
                self.data_dict,
                axis="custom",
                start_point=(0, 0),
                end_point=(20, 20),
            )

    def test_no_physical_dimensions(self):
        """Test extraction when physical dimensions aren't available."""
        # Create a data dict without physical dimensions
        simple_dict = {"width": 10, "height": 10}

        # Extract X cross section
        positions_x, heights_x = extract_cross_section(
            self.height_map, simple_dict, axis="x", position=5
        )

        # Positions should just be array indices
        np.testing.assert_array_equal(positions_x, np.arange(10))

        # Extract Y cross section
        positions_y, heights_y = extract_cross_section(
            self.height_map, simple_dict, axis="y", position=5
        )

        # Positions should just be array indices
        np.testing.assert_array_equal(positions_y, np.arange(10))

        # Extract custom cross section
        positions_custom, _ = extract_cross_section(
            self.height_map, simple_dict, axis="custom", start_point=(0, 0), end_point=(9, 9)
        )

        # First and last position should match the input distance
        self.assertAlmostEqual(positions_custom[0], 0.0)
        self.assertAlmostEqual(positions_custom[-1], 9.0 * np.sqrt(2))


class TestExtractProfileAtPercentage(unittest.TestCase):
    """Test the extract_profile_at_percentage function."""

    def setUp(self):
        """Create a sample height map and data dictionary for testing."""
        # Create a 10x10 sample height map
        self.height_map = np.zeros((10, 10))

        # Set values uniquely based on row and column
        for i in range(10):
            for j in range(10):
                self.height_map[i, j] = i * 10 + j

        # Create a data dictionary with physical dimensions
        self.data_dict = {
            "width": 10,
            "height": 10,
            "x_offset": 0.0,
            "y_offset": 0.0,
            "x_length": 5.0,
            "y_length": 10.0,
        }

        # Create a temporary directory for file output tests
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()

    def test_x_profile_middle(self):
        """Test extracting a horizontal profile at 50% (middle)."""
        profile = extract_profile_at_percentage(
            self.height_map, self.data_dict, axis="x", percentage=50.0
        )

        # Profile should have width elements
        self.assertEqual(len(profile), 10)

        # Check values match the expected row (50% = row 5 at index 4)
        expected_row = 5  # 50% of the 0-indexed 10-row array is row at index 4
        for j in range(10):
            self.assertAlmostEqual(profile[j], expected_row * 10 + j)

    def test_y_profile(self):
        """Test extracting a vertical profile."""
        profile = extract_profile_at_percentage(
            self.height_map, self.data_dict, axis="y", percentage=25.0
        )

        # Profile should have height elements
        self.assertEqual(len(profile), 10)

        # Check values match the expected column (25% = column 2 at index 2)
        expected_col = 2  # 25% of the 0-indexed 10-column array is column at index 2
        for i in range(10):
            self.assertAlmostEqual(profile[i], i * 10 + expected_col)

    def test_save_profile(self):
        """Test saving the profile to a file."""
        # Create path for the saved profile
        save_path = os.path.join(self.temp_dir.name, "profile.npy")

        profile = extract_profile_at_percentage(
            self.height_map, self.data_dict, axis="x", percentage=50.0, save_path=save_path
        )

        # File should exist
        self.assertTrue(os.path.exists(save_path))

        # Load the saved profile and verify it matches
        loaded_profile = np.load(save_path)
        np.testing.assert_array_equal(loaded_profile, profile)

    def test_percentage_bounds(self):
        """Test that percentage values are properly bounded."""
        # Check that values outside 0-100% are clipped
        profile_neg = extract_profile_at_percentage(
            self.height_map, self.data_dict, axis="x", percentage=-10.0
        )

        # Should be clipped to 0%
        expected_row = 0
        for j in range(10):
            self.assertAlmostEqual(profile_neg[j], expected_row * 10 + j)

        profile_over = extract_profile_at_percentage(
            self.height_map, self.data_dict, axis="x", percentage=150.0
        )

        # Should be clipped to 100%
        expected_row = 9
        for j in range(10):
            self.assertAlmostEqual(profile_over[j], expected_row * 10 + j)

    def test_invalid_axis(self):
        """Test that invalid axis raises ValueError."""
        with self.assertRaises(ValueError):
            extract_profile_at_percentage(self.height_map, self.data_dict, axis="z")


if __name__ == "__main__":
    unittest.main()
