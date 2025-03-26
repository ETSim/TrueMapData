"""Unit tests for TMD transformations utility module."""

import unittest
import numpy as np
import pytest
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock

from tmd.utils.transformations import (
    apply_translation,
    apply_rotation,
    apply_scaling,
    register_heightmaps
)
from tmd.utils.utils import create_sample_height_map


class TestTransformationsUtility(unittest.TestCase):
    """Test class for transformations utility functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temp directory for any file operations
        self.temp_dir = tempfile.mkdtemp()
        
        # Create different test height maps to use in tests
        self.flat_map = np.ones((20, 30), dtype=np.float32)
        self.gradient_map = np.linspace(0, 1, 20*30).reshape(20, 30).astype(np.float32)
        self.peak_map = create_sample_height_map(width=30, height=20, pattern='peak', noise_level=0.0)
        self.wave_map = create_sample_height_map(width=30, height=20, pattern='waves', noise_level=0.0)
        
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary directory and its contents
        shutil.rmtree(self.temp_dir)
    
    def test_translation_z_only(self):
        """Test height (Z) translation only."""
        # Create test case
        original = self.flat_map.copy()
        z_offset = 5.0
        
        # Apply transformation
        translated = apply_translation(original, (0.0, 0.0, z_offset))
        
        # Verify results
        self.assertEqual(translated.shape, original.shape)
        self.assertTrue(np.allclose(translated, original + z_offset))
    
    def test_translation_xy(self):
        """Test horizontal (X/Y) translation."""
        # Use a map with a distinct pattern for verifying shifts
        original = self.peak_map.copy()
        
        # Apply translation: half-width right, quarter-height down
        tx, ty = 0.5, 0.25
        translated = apply_translation(original, (tx, ty, 0.0))
        
        # Verify shape is preserved
        self.assertEqual(translated.shape, original.shape)
        
        # Verify the shift - use cross-correlation to detect the offset
        from scipy.signal import correlate2d
        corr = correlate2d(original, translated, mode='same')
        y, x = np.unravel_index(np.argmax(corr), corr.shape)
        
        # Calculate expected shifts in pixels
        expected_x_shift = int(round(tx * original.shape[1])) % original.shape[1]
        expected_y_shift = int(round(ty * original.shape[0])) % original.shape[0]
        
        # Allow for small rounding differences
        self.assertAlmostEqual(x, expected_x_shift, delta=1)
        self.assertAlmostEqual(y, expected_y_shift, delta=1)
    
    def test_translation_all_axes(self):
        """Test combined X/Y/Z translation."""
        # Use a map with a distinct pattern
        original = self.peak_map.copy()
        
        # Apply translation on all axes
        translation = (0.1, -0.15, 2.5)  # shift right 10%, up 15%, and raise by 2.5
        translated = apply_translation(original, translation)
        
        # Verify z-offset (mean difference should be the z-translation)
        self.assertAlmostEqual(
            np.mean(translated) - np.mean(original),
            translation[2],
            places=5
        )
    
    def test_rotation_z_axis_90_degrees(self):
        """Test 90-degree Z-axis rotation."""
        # Create a rectangular test pattern that's easy to verify when rotated
        test_map = np.zeros((10, 20), dtype=np.float32)
        test_map[2:8, 5:15] = 1.0  # Creates a rectangle in the middle
        
        # Apply a 90-degree rotation (counter-clockwise)
        rotated = apply_rotation(test_map, (0.0, 0.0, 90.0))
        
        # Verify shape is preserved
        self.assertEqual(rotated.shape, test_map.shape)
        
        # The central pattern should now be rotated
        # Given a 90° rotation, highest values should now be in the center columns
        self.assertTrue(np.mean(rotated[:, 9:11]) > np.mean(rotated[:, :3]))
        self.assertTrue(np.mean(rotated[:, 9:11]) > np.mean(rotated[:, -3:]))
    
    def test_rotation_x_y_axis(self):
        """Test X and Y axis rotations."""
        # Use wave pattern which will be affected by X/Y rotations
        original = self.wave_map.copy()
        
        # Apply small X rotation
        rotated_x = apply_rotation(original, (15.0, 0.0, 0.0))
        
        # Apply small Y rotation 
        rotated_y = apply_rotation(original, (0.0, 15.0, 0.0))
        
        # Verify shapes are preserved
        self.assertEqual(rotated_x.shape, original.shape)
        self.assertEqual(rotated_y.shape, original.shape)
        
        # Verify rotations had an effect (not equal to original)
        self.assertTrue(np.sum(np.abs(rotated_x - original)) > 0.01)
        self.assertTrue(np.sum(np.abs(rotated_y - original)) > 0.01)
        
        # Verify X and Y rotations produce different results
        self.assertTrue(np.sum(np.abs(rotated_x - rotated_y)) > 0.01)
    
    def test_combined_rotation(self):
        """Test combined X/Y/Z rotations."""
        original = self.wave_map.copy()
        
        # Apply rotation on all axes
        rotation = (5.0, 10.0, 45.0)
        rotated = apply_rotation(original, rotation)
        
        # Verify shape is preserved
        self.assertEqual(rotated.shape, original.shape)
        
        # Verify rotation had an effect
        self.assertTrue(np.sum(np.abs(rotated - original)) > 0.01)
    
    def test_no_rotation(self):
        """Test that zero rotation returns the original array."""
        original = self.wave_map.copy()
        rotated = apply_rotation(original, (0.0, 0.0, 0.0))
        
        # Arrays should be identical
        np.testing.assert_array_equal(rotated, original)
    
    def test_scaling_z_only(self):
        """Test height (Z) scaling only."""
        original = self.gradient_map.copy()
        scale_z = 2.5
        
        # Apply scaling
        scaled = apply_scaling(original, (1.0, 1.0, scale_z))
        
        # Verify shape is preserved
        self.assertEqual(scaled.shape, original.shape)
        
        # Verify height values are scaled
        np.testing.assert_array_almost_equal(scaled, original * scale_z)
    
    def test_scaling_xy(self):
        """Test horizontal (X/Y) scaling."""
        original = self.peak_map.copy()
        
        # Scale to 1.5x width, 0.75x height
        scale_factors = (1.5, 0.75, 1.0)
        scaled = apply_scaling(original, scale_factors)
        
        # Verify new dimensions
        expected_height = int(round(original.shape[0] * scale_factors[1]))
        expected_width = int(round(original.shape[1] * scale_factors[0]))
        self.assertEqual(scaled.shape, (expected_height, expected_width))
    
    def test_combined_scaling(self):
        """Test combined X/Y/Z scaling."""
        original = self.peak_map.copy()
        
        # Apply scaling on all axes
        scaling = (1.2, 0.9, 3.0)
        scaled = apply_scaling(original, scaling)
        
        # Verify new dimensions
        expected_height = int(round(original.shape[0] * scaling[1]))
        expected_width = int(round(original.shape[1] * scaling[0]))
        self.assertEqual(scaled.shape, (expected_height, expected_width))
        
        # Verify average height is scaled (accounting for interpolation effects)
        z_scale_factor = scaling[2]
        scaled_mean = np.mean(scaled)
        original_mean = np.mean(original)
        self.assertAlmostEqual(scaled_mean / original_mean, z_scale_factor, delta=0.1)
    
    @patch('tmd.utils.transformations.cv2')
    def test_register_heightmaps_phase_correlation(self, mock_cv2):
        """Test registration using phase correlation."""
        # Create a reference and a target with known offset
        reference = self.peak_map.copy()
        
        # Create target with a known shift
        tx, ty = 3, 2  # pixels
        target = np.roll(np.roll(reference, tx, axis=1), ty, axis=0)
        
        # Setup mock for cv2.phaseCorrelate
        mock_cv2.phaseCorrelate.return_value = ((-tx, -ty), 0.95)  # (shift, response)
        
        # Test registration
        registered, transform = register_heightmaps(reference, target, 'phase_correlation')
        
        # Verify registration corrected the offset
        self.assertEqual(registered.shape, reference.shape)
        expected_tx = -tx / reference.shape[1]
        expected_ty = -ty / reference.shape[0]
        self.assertAlmostEqual(transform['translation'][0], expected_tx, places=5)
        self.assertAlmostEqual(transform['translation'][1], expected_ty, places=5)
    
    @patch('tmd.utils.transformations.cv2')
    @patch('tmd.utils.transformations.HAS_CV2', True)
    def test_register_heightmaps_feature_matching(self, mock_cv2):
        """Test registration using feature matching."""
        # Mock necessary OpenCV functions
        kp1 = [MagicMock() for _ in range(10)]
        kp2 = [MagicMock() for _ in range(10)]
        for i, (k1, k2) in enumerate(zip(kp1, kp2)):
            k1.pt = (10 + i, 10 + i)
            k2.pt = (15 + i, 13 + i)  # 5px right, 3px down
        
        des1 = np.random.random((10, 32)).astype(np.uint8)
        des2 = np.random.random((10, 32)).astype(np.uint8)
        
        # Setup mock objects
        orb_mock = MagicMock()
        orb_mock.detectAndCompute.side_effect = [(kp1, des1), (kp2, des2)]
        mock_cv2.ORB_create.return_value = orb_mock
        
        matcher_mock = MagicMock()
        matches = [MagicMock() for _ in range(10)]
        for i, m in enumerate(matches):
            m.queryIdx = i
            m.trainIdx = i
            m.distance = i
        matcher_mock.match.return_value = matches
        mock_cv2.BFMatcher.return_value = matcher_mock
        
        # Mock the homography calculation
        mock_cv2.findHomography.return_value = (
            np.array([[1, 0, -5], [0, 1, -3], [0, 0, 1]]),  # H matrix (shift left 5px, up 3px)
            np.ones(10)  # mask
        )
        
        # Create test data
        reference = self.wave_map.copy()
        target = self.wave_map.copy()
        
        # Test registration
        registered, transform = register_heightmaps(reference, target, 'feature_matching')
        
        # Verify the registration parameters
        height, width = reference.shape
        expected_tx = -5 / width
        expected_ty = -3 / height
        self.assertAlmostEqual(transform['translation'][0], expected_tx, places=3)
        self.assertAlmostEqual(transform['translation'][1], expected_ty, places=3)
    
    def test_register_heightmaps_fallback(self):
        """Test registration fallback when OpenCV is not available."""
        # Create a reference and a target with known offset
        reference = self.peak_map.copy()
        tx, ty = 2, 1  # small shift
        target = np.roll(np.roll(reference, tx, axis=1), ty, axis=0)
        
        # Force fallback by using an unknown method
        registered, transform = register_heightmaps(reference, target, 'unknown_method')
        
        # Should still capture the shift approximately
        self.assertTrue(abs(transform['translation'][0]) > 0)
        self.assertTrue(abs(transform['translation'][1]) > 0)
    
    def test_register_heightmaps_different_sizes(self):
        """Test registration with differently sized images."""
        # Create a reference and a smaller target
        reference = self.peak_map.copy()
        smaller_target = self.peak_map[::2, ::2].copy()  # Take every 2nd pixel
        
        # Test registration (should resize target to match reference)
        registered, transform = register_heightmaps(reference, smaller_target)
        
        # Verify registered image matches reference size
        self.assertEqual(registered.shape, reference.shape)


if __name__ == '__main__':
    unittest.main()
