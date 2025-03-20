"""
Unit tests for the filter module functions.

Tests various filtering and surface analysis functions including
Gaussian smoothing, waviness/roughness extraction, RMS calculations,
and gradient/slope computations.
"""
import unittest

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from scipy import ndimage

from tmd.utils.filter import (
    apply_gaussian_filter,
    calculate_rms_roughness,
    calculate_rms_waviness,
    calculate_slope,
    calculate_surface_gradient,
    extract_roughness,
    extract_waviness,
)


class TestGaussianFilter(unittest.TestCase):
    """Tests for apply_gaussian_filter function."""

    def setUp(self):
        """Create sample height maps for testing."""
        # Create a simple height map with a sharp peak in the middle
        self.test_map = np.zeros((20, 20))
        self.test_map[10, 10] = 1.0

        # Create a height map with alternating peaks and valleys (high frequency)
        x = np.linspace(-5, 5, 100)
        y = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x, y)
        self.high_freq_map = np.sin(4 * X) * np.cos(4 * Y)

        # Create a map with noise
        self.noisy_map = np.random.normal(0, 1, (30, 30))

    def test_basic_smoothing(self):
        """Test that Gaussian filter smooths a sharp peak."""
        # Apply Gaussian filter with moderate sigma
        smoothed = apply_gaussian_filter(self.test_map, sigma=2.0)

        # Peak value should be lower after smoothing
        self.assertLess(smoothed[10, 10], 1.0)

        # Surrounding values should be higher (smoothing spreads the peak)
        self.assertGreater(smoothed[8, 8], 0.0)

        # Total volume (sum) should be approximately conserved
        self.assertAlmostEqual(np.sum(self.test_map), np.sum(smoothed), delta=0.01)

    def test_sigma_effect(self):
        """Test the effect of different sigma values."""
        # Test with small sigma
        smooth_small = apply_gaussian_filter(self.test_map, sigma=1.0)

        # Test with larger sigma
        smooth_large = apply_gaussian_filter(self.test_map, sigma=3.0)

        # Larger sigma should result in more smoothing (lower peak)
        self.assertLess(smooth_large[10, 10], smooth_small[10, 10])

        # Larger sigma should spread energy further from center
        # Compare values at a distance from the peak
        self.assertGreater(smooth_large[5, 5], smooth_small[5, 5])

    def test_high_frequency_reduction(self):
        """Test that Gaussian filter reduces high-frequency components."""
        # Apply filter to high frequency map
        smoothed = apply_gaussian_filter(self.high_freq_map, sigma=2.0)

        # Calculate standard deviation before and after
        orig_std = np.std(self.high_freq_map)
        smooth_std = np.std(smoothed)

        # Variation should be reduced
        self.assertLess(smooth_std, orig_std)

    def test_noise_reduction(self):
        """Test that Gaussian filter reduces noise."""
        # Apply filter to noisy map
        smoothed = apply_gaussian_filter(self.noisy_map, sigma=2.0)

        # Calculate local variation using a method that doesn't have broadcasting issues
        # Get absolute differences between adjacent pixels
        orig_dy = np.abs(np.diff(self.noisy_map, axis=0))
        orig_dx = np.abs(np.diff(self.noisy_map, axis=1))

        smooth_dy = np.abs(np.diff(smoothed, axis=0))
        smooth_dx = np.abs(np.diff(smoothed, axis=1))

        # Average the variations separately to avoid broadcasting issues
        orig_var = (np.mean(orig_dy) + np.mean(orig_dx)) / 2
        smooth_var = (np.mean(smooth_dy) + np.mean(smooth_dx)) / 2

        # Local variation should be reduced
        self.assertLess(smooth_var, orig_var)

    def test_identity_with_zero_sigma(self):
        """Test that sigma=0 returns an array very close to the original."""
        # With sigma near zero, result should be close to original
        # Note: sigma=0 exactly would cause issues with some implementations
        result = apply_gaussian_filter(self.test_map, sigma=0.01)

        # Check that results are very close
        assert_array_almost_equal(result, self.test_map, decimal=10)


class TestWavinessRoughness(unittest.TestCase):
    """Tests for waviness and roughness extraction functions."""

    def setUp(self):
        """Create test data combining waviness and roughness components."""
        # Create coordinates
        x = np.linspace(-10, 10, 100)
        y = np.linspace(-10, 10, 100)
        X, Y = np.meshgrid(x, y)

        # Create waviness component (low frequency)
        self.waviness_component = 0.5 * np.sin(X / 5) + 0.3 * np.cos(Y / 4)

        # Create roughness component (high frequency)
        self.roughness_component = 0.1 * np.sin(X * 2) * np.cos(Y * 2)

        # Combined height map
        self.height_map = self.waviness_component + self.roughness_component

    def test_waviness_extraction(self):
        """Test that waviness extraction captures low-frequency components."""
        extracted_waviness = extract_waviness(self.height_map, sigma=2.0)

        # The extracted waviness should be close to the original waviness component
        # but not exactly the same due to the filtering process
        correlation = np.corrcoef(extracted_waviness.flatten(), self.waviness_component.flatten())[
            0, 1
        ]

        # Check high correlation between extracted and actual waviness
        self.assertGreater(correlation, 0.8)

        # Check that the range of extracted waviness is similar to original waviness
        self.assertAlmostEqual(
            np.ptp(extracted_waviness), np.ptp(self.waviness_component), delta=0.3
        )

    def test_roughness_extraction(self):
        """Test that roughness extraction captures high-frequency components."""
        extracted_roughness = extract_roughness(self.height_map, sigma=2.0)

        # Check correlation between extracted and actual roughness
        correlation = np.corrcoef(
            extracted_roughness.flatten(), self.roughness_component.flatten()
        )[0, 1]

        self.assertGreater(correlation, 0.5)

        # The standard deviation of roughness should be lower than the original height map
        self.assertLess(np.std(extracted_roughness), np.std(self.height_map))

        # Mean of roughness should be close to zero
        self.assertAlmostEqual(np.mean(extracted_roughness), 0, delta=0.05)

    def test_waviness_plus_roughness_equals_original(self):
        """Test that waviness + roughness = original height map."""
        sigma = 2.0
        extracted_waviness = extract_waviness(self.height_map, sigma=sigma)
        extracted_roughness = extract_roughness(self.height_map, sigma=sigma)

        # Sum should equal the original
        reconstructed = extracted_waviness + extracted_roughness
        assert_array_almost_equal(reconstructed, self.height_map)

    def test_sigma_effect_on_separation(self):
        """Test that sigma affects the waviness/roughness separation."""
        # Create an extremely basic test with clearly separated components
        x = np.linspace(-5, 5, 100)
        y = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x, y)

        # Use completely different frequencies to ensure clear separation
        low_freq = np.sin(X / 20)  # Super low frequency
        high_freq = 0.05 * np.sin(X * 10)  # Very high frequency with small amplitude

        # Create test map with clear separation
        test_map = low_freq + high_freq

        # Use very different sigmas - one tiny and one massive
        tiny_sigma = 0.01
        huge_sigma = 5.0

        # First verify the basic approach:
        # With tiny sigma, waviness should be close to original
        # With huge sigma, waviness should be close to low_freq only
        tiny_waviness = extract_waviness(test_map, sigma=tiny_sigma)
        huge_waviness = extract_waviness(test_map, sigma=huge_sigma)

        # This test should definitely pass:
        # The amplitude (max-min) of the huge sigma processing should be smaller
        # because it will remove more of the high frequency components
        self.assertGreater(
            np.max(tiny_waviness) - np.min(tiny_waviness),
            np.max(huge_waviness) - np.min(huge_waviness),
        )

        # And the roughness should be affected oppositely
        tiny_roughness = extract_roughness(test_map, sigma=tiny_sigma)
        huge_roughness = extract_roughness(test_map, sigma=huge_sigma)

        # More roughness with large sigma since it removes more from the original
        self.assertGreater(
            np.max(huge_roughness) - np.min(huge_roughness),
            np.max(tiny_roughness) - np.min(tiny_roughness),
        )


class TestRMSCalculations(unittest.TestCase):
    """Tests for RMS calculation functions."""

    def setUp(self):
        """Create test data for RMS calculations."""
        # Create a simple height map
        x = np.linspace(-5, 5, 50)
        y = np.linspace(-5, 5, 50)
        X, Y = np.meshgrid(x, y)

        # Create height map with known components
        self.waviness = 2.0 * np.sin(X / 3) * np.cos(Y / 4)
        self.roughness = 0.5 * np.sin(X * 2) * np.cos(Y * 2)
        self.height_map = self.waviness + self.roughness

        # Calculate theoretical RMS values
        self.theoretical_rms_waviness = np.sqrt(np.mean(self.waviness**2))
        self.theoretical_rms_roughness = np.sqrt(np.mean(self.roughness**2))

    def test_rms_roughness(self):
        """Test calculation of RMS roughness."""
        # For this test, use large sigma so waviness captures most of the main features
        sigma = 1.0
        rms_roughness = calculate_rms_roughness(self.height_map, sigma=sigma)

        # RMS roughness should be positive
        self.assertGreater(rms_roughness, 0)

        # Get the actual roughness component for comparison
        roughness = extract_roughness(self.height_map, sigma=sigma)
        manual_rms = np.sqrt(np.mean(roughness**2))

        # The function result should match manual calculation
        self.assertAlmostEqual(rms_roughness, manual_rms)

        # Result should be in the same ballpark as theoretical value
        # (not exactly equal due to filtering effects)
        self.assertLess(abs(rms_roughness - self.theoretical_rms_roughness), 0.3)

    def test_rms_waviness(self):
        """Test calculation of RMS waviness."""
        # For this test, use large sigma so waviness captures most of the main features
        sigma = 1.0
        rms_waviness = calculate_rms_waviness(self.height_map, sigma=sigma)

        # RMS waviness should be positive
        self.assertGreater(rms_waviness, 0)

        # Get the actual waviness component for comparison
        waviness = extract_waviness(self.height_map, sigma=sigma)
        manual_rms = np.sqrt(np.mean(waviness**2))

        # The function result should match manual calculation
        self.assertAlmostEqual(rms_waviness, manual_rms)

        # Result should be in the same ballpark as theoretical value
        # (not exactly equal due to filtering effects)
        self.assertLess(abs(rms_waviness - self.theoretical_rms_waviness), 0.5)

    def test_rms_with_different_sigmas(self):
        """Test RMS calculations with different sigma values."""
        # Create a very simple test case with clear separation
        x = np.linspace(-5, 5, 100)
        y = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x, y)

        # Use completely different frequencies
        low_freq = np.sin(X / 20)  # Super low frequency
        high_freq = 0.05 * np.sin(X * 10)  # Very high frequency with small amplitude

        # Create test map
        test_map = low_freq + high_freq

        # Different sigmas
        tiny_sigma = 0.01
        huge_sigma = 5.0

        # Use direct calculation to verify our functions
        tiny_roughness = extract_roughness(test_map, sigma=tiny_sigma)
        huge_roughness = extract_roughness(test_map, sigma=huge_sigma)

        # Check amplitudes instead of std or RMS
        tiny_amplitude = np.max(tiny_roughness) - np.min(tiny_roughness)
        huge_amplitude = np.max(huge_roughness) - np.min(huge_roughness)

        # With huge sigma, roughness should capture more of the signal
        self.assertGreater(huge_amplitude, tiny_amplitude)

        # Check our actual functions directly
        tiny_rms = calculate_rms_roughness(test_map, sigma=tiny_sigma)
        huge_rms = calculate_rms_roughness(test_map, sigma=huge_sigma)

        # RMS of roughness should be higher with huge sigma
        # (since it's removing more of the base signal)
        self.assertGreater(huge_rms, tiny_rms)


class TestGradientAndSlope(unittest.TestCase):
    """Tests for gradient and slope calculation functions."""

    def setUp(self):
        """Create test data for gradient and slope calculations."""
        # Create a simple height map with a known gradient
        self.size = 50
        x = np.linspace(-5, 5, self.size)
        y = np.linspace(-5, 5, self.size)
        X, Y = np.meshgrid(x, y)

        # Linear slope in X and Y
        self.x_slope = 0.2
        self.y_slope = 0.3
        self.height_map = self.x_slope * X + self.y_slope * Y

        # Add some surface features
        self.height_map += np.sin(X) * 0.1

    def test_surface_gradient(self):
        """Test calculation of surface gradients."""
        # Force the scale factor to match the expected test values
        scale = 1.0  # Using a specific scale factor for consistency
        grad_x, grad_y = calculate_surface_gradient(self.height_map, scale=scale)

        # Check shape
        self.assertEqual(grad_x.shape, (self.size, self.size))
        self.assertEqual(grad_y.shape, (self.size, self.size))

        # We'll directly extract the central portion of the gradients to avoid edge effects
        central_slice = slice(10, 40)  # Use a more central region to avoid edge effects
        grad_x_center = grad_x[central_slice, central_slice]
        grad_y_center = grad_y[central_slice, central_slice]

        # Check mean gradient - should be close to the slopes we set (times scale)
        expected_x_grad = self.x_slope * scale
        expected_y_grad = self.y_slope * scale

        # Only assert up to a reasonable precision for numerical gradients
        self.assertAlmostEqual(np.mean(grad_x_center), expected_x_grad, delta=0.1 * expected_x_grad)
        self.assertAlmostEqual(np.mean(grad_y_center), expected_y_grad, delta=0.1 * expected_y_grad)

    def test_slope_calculation(self):
        """Test calculation of slope magnitude."""
        # Use the same scale factor to match expected test values
        scale = 1.0  # Using a specific scale factor for consistency
        slope = calculate_slope(self.height_map, scale=scale)

        # Check shape
        self.assertEqual(slope.shape, (self.size, self.size))

        # Expected slope magnitude
        expected_magnitude = np.sqrt(self.x_slope**2 + self.y_slope**2) * scale

        # Check central part (avoiding edges)
        central_slice = slice(10, 40)
        slope_center = slope[central_slice, central_slice]

        # Mean slope should be close to expected magnitude
        self.assertAlmostEqual(
            np.mean(slope_center), expected_magnitude, delta=0.1 * expected_magnitude
        )

    def test_gradient_and_slope_consistency(self):
        """Test that slope is consistent with gradients."""
        # Calculate gradients and slope with SAME scale factor
        scale = 1.0
        grad_x, grad_y = calculate_surface_gradient(self.height_map, scale=scale)
        slope = calculate_slope(self.height_map, scale=scale)

        # Calculate slope from gradients manually
        manual_slope = np.sqrt(grad_x**2 + grad_y**2)

        # They should be almost identical (within floating point precision)
        np.testing.assert_almost_equal(slope, manual_slope)


if __name__ == "__main__":
    unittest.main()
