"""Unit tests for TMD filters utility module."""

import unittest
import numpy as np
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock

from tmd.utils.filters import (
    apply_gaussian_filter,
    extract_waviness,
    extract_roughness,
    calculate_rms_roughness,
    calculate_rms_waviness,
    calculate_surface_gradient,
    calculate_slope,
    apply_median_filter,
    apply_morphological_filter,
    apply_wavelet_filter,
    apply_fft_filter,
    apply_klt_filter,
    # New functions to test
    apply_window,
    calculate_frequency_spectrum,
    calculate_power_spectral_density,
    calculate_surface_isotropy,
    detect_surface_periodicity,
    calculate_autocorrelation,
    calculate_intercorrelation,
    denoise_by_fft,
    apply_continuous_wavelet_transform,
    apply_discrete_wavelet_transform,
    discrete_wavelet_filtering,
    get_available_wavelets
)


class TestFiltersUtility(unittest.TestCase):
    """Test class for filters utility functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for any file output tests
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test height maps with different characteristics
        self.flat_map = np.ones((20, 20), dtype=np.float32)
        
        # Create grid coordinates for generating test patterns
        x = np.linspace(0, 1, 20)
        y = np.linspace(0, 1, 20)
        X, Y = np.meshgrid(x, y)
        self.gradient_map = X.astype(np.float32)
        
        # Create sine patterns for frequency-based tests
        x = np.linspace(-4*np.pi, 4*np.pi, 20)
        y = np.linspace(-4*np.pi, 4*np.pi, 20)
        X, Y = np.meshgrid(x, y)
        self.sine_map = np.sin(X) * np.cos(Y)
        self.combined_map = np.sin(X) + np.sin(5*X)
        self.noisy_map = np.sin(X) + np.random.normal(0, 0.2, (20, 20))
        
        # 1D profiles for testing 1D functions
        self.profile_flat = np.ones(100, dtype=np.float32)
        self.profile_sine = np.sin(np.linspace(0, 8*np.pi, 100)).astype(np.float32)
        self.profile_complex = np.sin(np.linspace(0, 4*np.pi, 100)) + 0.2 * np.sin(np.linspace(0, 20*np.pi, 100))
        self.profile_complex = self.profile_complex.astype(np.float32)
        
        # Create test pattern with periodicity
        period_x = np.sin(np.linspace(0, 8*np.pi, 50))
        period_y = np.sin(np.linspace(0, 6*np.pi, 50))
        self.periodic_map = np.outer(period_y, period_x).astype(np.float32)
        
        # Ensure consistent data types
        self.sine_map = self.sine_map.astype(np.float32)
        self.combined_map = self.combined_map.astype(np.float32)
        self.noisy_map = self.noisy_map.astype(np.float32)
    
    def tearDown(self):
        """Tear down test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_gaussian_filter(self):
        """Test Gaussian smoothing filter."""
        filtered = apply_gaussian_filter(self.noisy_map, sigma=1.0)
        
        # Verify output shape, type and effect
        self.assertEqual(filtered.shape, self.noisy_map.shape)
        self.assertEqual(filtered.dtype, self.noisy_map.dtype)
        self.assertLess(np.std(filtered), np.std(self.noisy_map))
        
        # Verify that larger sigma produces more smoothing
        filtered_small = apply_gaussian_filter(self.noisy_map, sigma=0.5)
        filtered_large = apply_gaussian_filter(self.noisy_map, sigma=2.0)
        self.assertLess(np.std(filtered_large), np.std(filtered_small))
    
    def test_waviness_and_roughness_extraction(self):
        """Test extraction of waviness and roughness components."""
        # Test waviness extraction
        waviness = extract_waviness(self.combined_map, sigma=1.0)
        self.assertEqual(waviness.shape, self.combined_map.shape)
        self.assertLess(np.std(waviness), np.std(self.combined_map))
        
        # Test roughness extraction and verify sum equals original
        roughness = extract_roughness(self.combined_map, sigma=1.0)
        reconstructed = roughness + waviness
        np.testing.assert_allclose(reconstructed, self.combined_map, rtol=1e-6, atol=1e-6)
        
        # Check that larger sigma includes more in waviness, less in roughness
        waviness_large = extract_waviness(self.combined_map, sigma=3.0)
        roughness_large = extract_roughness(self.combined_map, sigma=3.0)
        self.assertLess(np.std(waviness_large), np.std(waviness))
        self.assertGreater(np.std(roughness_large), np.std(roughness))
    
    def test_rms_calculations(self):
        """Test RMS roughness and waviness calculations."""
        # Flat map should have zero roughness and constant waviness
        rms_roughness_flat = calculate_rms_roughness(self.flat_map)
        rms_waviness_flat = calculate_rms_waviness(self.flat_map)
        self.assertAlmostEqual(rms_roughness_flat, 0.0, delta=1e-6)
        self.assertAlmostEqual(rms_waviness_flat, 1.0, delta=1e-6)
        
        # Test with different sigma values on the combined map
        rms_roughness_small = calculate_rms_roughness(self.combined_map, sigma=1.0)
        rms_roughness_large = calculate_rms_roughness(self.combined_map, sigma=3.0)
        rms_waviness_small = calculate_rms_waviness(self.combined_map, sigma=1.0)
        rms_waviness_large = calculate_rms_waviness(self.combined_map, sigma=3.0)
        
        # Larger sigma should include more in roughness
        self.assertGreater(rms_roughness_large, rms_roughness_small)
        self.assertLess(rms_waviness_large, rms_waviness_small)
    
    def test_surface_gradient_and_slope(self):
        """Test gradient and slope calculations."""
        # Calculate gradients for the gradient map
        grad_x, grad_y = calculate_surface_gradient(self.gradient_map, scale=1.0)
        
        # For a simple x-gradient map, grad_y should be near zero, grad_x positive
        self.assertAlmostEqual(np.mean(np.abs(grad_y)), 0.0, delta=1e-6)
        self.assertGreater(np.mean(grad_x), 0.0)
        
        # Test that scale parameter works correctly
        grad_x_scaled, _ = calculate_surface_gradient(self.gradient_map, scale=2.0)
        self.assertAlmostEqual(np.mean(grad_x_scaled) / np.mean(grad_x), 2.0, delta=0.5)
        
        # Test slope calculation
        flat_slope = calculate_slope(self.flat_map)
        self.assertAlmostEqual(np.mean(flat_slope), 0.0, delta=1e-6)
        
        # Test slope scale parameter
        slope = calculate_slope(self.gradient_map)
        scaled_slope = calculate_slope(self.gradient_map, scale=2.0)
        self.assertAlmostEqual(np.mean(scaled_slope) / np.mean(slope), 2.0, delta=0.5)
    
    def test_filtering_operations(self):
        """Test various filtering operations."""
        # Test median filter with impulse noise
        impulse_map = self.flat_map.copy()
        impulse_map[5, 5] = 10.0  # Add spike
        filtered_median = apply_median_filter(impulse_map, size=3)
        self.assertLess(abs(filtered_median[5, 5] - 1.0), abs(impulse_map[5, 5] - 1.0))
        
        # Test morphological operations
        test_map = np.zeros((20, 20), dtype=np.float32)
        test_map[5:15, 5:15] = 1.0  # Create a square
        
        # Test each operation and verify expected behavior
        opened = apply_morphological_filter(test_map, operation='opening')
        self.assertLess(np.sum(opened), np.sum(test_map))
        
        closed = apply_morphological_filter(test_map, operation='closing')
        self.assertGreaterEqual(np.sum(closed), np.sum(test_map))
        
        eroded = apply_morphological_filter(test_map, operation='erosion')
        self.assertLess(np.sum(eroded), np.sum(test_map))
        
        dilated = apply_morphological_filter(test_map, operation='dilation')
        self.assertGreater(np.sum(dilated), np.sum(test_map))
        
        # Test invalid operation raises ValueError
        with self.assertRaises(ValueError):
            apply_morphological_filter(test_map, operation='invalid')
        
        # Test wavelet filter
        wavelet_filtered = apply_wavelet_filter(self.sine_map, level=1)
        self.assertEqual(wavelet_filtered.shape, self.sine_map.shape)
        self.assertLess(np.std(wavelet_filtered), np.std(self.sine_map))
    
    def test_frequency_domain_filters(self):
        """Test frequency domain filtering methods."""
        # Test FFT-based filtering
        filtered_lowpass = apply_fft_filter(
            self.combined_map, cutoff_high=5.0, filter_type='lowpass'
        )
        self.assertEqual(filtered_lowpass.shape, self.combined_map.shape)
        self.assertLess(np.std(filtered_lowpass), np.std(self.combined_map))
        
        filtered_highpass = apply_fft_filter(
            self.combined_map, cutoff_low=5.0, filter_type='highpass'
        )
        self.assertGreater(np.std(filtered_highpass), 0.0)
        
        filtered_bandpass = apply_fft_filter(
            self.combined_map, cutoff_low=2.0, cutoff_high=8.0, 
            filter_type='bandpass'
        )
        self.assertGreater(np.std(filtered_bandpass), 0.0)
        
        # Test invalid filter type
        with self.assertRaises(ValueError):
            apply_fft_filter(self.combined_map, filter_type='invalid')
    
    def test_klt_filter(self):
        """Test KLT filtering on various inputs."""
        # Create test pattern with noise
        base_pattern = np.outer(
            np.sin(np.linspace(0, 3*np.pi, 20)),
            np.cos(np.linspace(0, 2*np.pi, 20))
        )
        noisy_map = base_pattern + np.random.normal(0, 0.2, (20, 20))
        noisy_map = noisy_map.astype(np.float32)
        
        # Test whole map KLT
        filtered_whole = apply_klt_filter(noisy_map, retain_components=0.9)
        self.assertEqual(filtered_whole.shape, noisy_map.shape)
        self.assertLess(np.std(filtered_whole - base_pattern), np.std(noisy_map - base_pattern))
        
        # Test component retention level effect
        filtered_less = apply_klt_filter(noisy_map, retain_components=0.6)
        filtered_more = apply_klt_filter(noisy_map, retain_components=0.95)
        mse_less = np.mean((filtered_less - noisy_map)**2)
        mse_more = np.mean((filtered_more - noisy_map)**2)
        self.assertGreater(mse_less, mse_more)
        
        # Test patch-based KLT
        filtered_patches = apply_klt_filter(
            noisy_map, 
            retain_components=0.8,
            patch_size=(8, 8),
            stride=2
        )
        self.assertEqual(filtered_patches.shape, noisy_map.shape)
        
        # Test NaN handling
        nan_map = noisy_map.copy()
        nan_map[5:7, 8:10] = np.nan
        filtered_nan = apply_klt_filter(nan_map, retain_components=0.9)
        self.assertTrue(np.array_equal(np.isnan(filtered_nan), np.isnan(nan_map)))
        valid_mask = ~np.isnan(nan_map)
        self.assertGreater(np.sum(np.abs(filtered_nan[valid_mask] - nan_map[valid_mask])), 0)
    
    # NEW TESTS FOR ADDED FUNCTIONS
    
    def test_apply_window(self):
        """Test applying window functions to 1D and 2D data."""
        # Test 1D windowing
        windowed_1d = apply_window(self.profile_sine, window_type='hann')
        
        # Check shape and type preservation
        self.assertEqual(windowed_1d.shape, self.profile_sine.shape)
        self.assertEqual(windowed_1d.dtype, self.profile_sine.dtype)
        
        # Check that values at edges are reduced (window effect)
        self.assertLess(windowed_1d[0], self.profile_sine[0])
        self.assertLess(windowed_1d[-1], self.profile_sine[-1])
        
        # Test 2D windowing
        windowed_2d = apply_window(self.sine_map, window_type='hamming')
        
        # Check shape and type preservation
        self.assertEqual(windowed_2d.shape, self.sine_map.shape)
        self.assertEqual(windowed_2d.dtype, self.sine_map.dtype)
        
        # Check that values at edges are reduced (window effect)
        self.assertLess(np.mean(windowed_2d[0, :]), np.mean(self.sine_map[0, :]))
        self.assertLess(np.mean(windowed_2d[-1, :]), np.mean(self.sine_map[-1, :]))
        
        # Test invalid dimension
        with self.assertRaises(ValueError):
            apply_window(np.zeros((2, 3, 4)), window_type='hann')
    
    def test_calculate_frequency_spectrum_1d(self):
        """Test calculating frequency spectrum of 1D profiles."""
        # Calculate spectrum for 1D sine wave
        spectrum = calculate_frequency_spectrum(
            self.profile_sine, pixel_size=0.1, apply_windowing=True
        )
        
        # Check return type and expected keys
        self.assertIsInstance(spectrum, dict)
        for key in ['frequencies', 'magnitude', 'phase', 'wavelength']:
            self.assertIn(key, spectrum)
        
        # Check shapes
        n = len(self.profile_sine)
        self.assertEqual(len(spectrum['frequencies']), n//2)
        self.assertEqual(len(spectrum['magnitude']), n//2)
        
        # Find peak frequency (should match the sine wave frequency)
        main_freq_idx = np.argmax(spectrum['magnitude'])
        # With 8π over 100 points at 0.1 spacing, frequency should be around 0.4 Hz
        # (8π cycles / (100*0.1) distance units)
        self.assertAlmostEqual(spectrum['frequencies'][main_freq_idx], 0.4, delta=0.1)
    
    def test_calculate_frequency_spectrum_2d(self):
        """Test calculating frequency spectrum of 2D height maps."""
        # Calculate spectrum for 2D sine pattern
        spectrum = calculate_frequency_spectrum(
            self.sine_map, pixel_size=0.1, apply_windowing=True
        )
        
        # Check return type and expected keys
        self.assertIsInstance(spectrum, dict)
        for key in ['freq_x', 'freq_y', 'magnitude', 'phase', 'wavelength', 'angle']:
            self.assertIn(key, spectrum)
        
        # Check shapes
        self.assertEqual(spectrum['freq_x'].shape, self.sine_map.shape)
        self.assertEqual(spectrum['magnitude'].shape, self.sine_map.shape)
        
        # Maximum magnitude should be at the frequency corresponding to sine pattern
        y_max, x_max = np.unravel_index(np.argmax(spectrum['magnitude']), spectrum['magnitude'].shape)
        
        # Check non-zero magnitude
        self.assertGreater(spectrum['magnitude'].max(), 0)
    
    def test_calculate_power_spectral_density(self):
        """Test calculation of power spectral density."""
        # Calculate PSD for 1D profile
        psd_1d = calculate_power_spectral_density(
            self.profile_complex, 
            pixel_size=0.1, 
            apply_windowing=True,
            smooth_spectrum=True
        )
        
        # Check return type and expected keys
        self.assertIsInstance(psd_1d, dict)
        self.assertIn('psd', psd_1d)
        self.assertIn('frequencies', psd_1d)
        
        # PSD should be positive and properly sized
        self.assertTrue(np.all(psd_1d['psd'] >= 0))
        self.assertEqual(len(psd_1d['psd']), len(self.profile_complex)//2)
        
        # Calculate PSD for 2D height map
        psd_2d = calculate_power_spectral_density(
            self.sine_map, 
            pixel_size=0.1,
            apply_windowing=True
        )
        
        # Check return type and expected keys for 2D
        self.assertIsInstance(psd_2d, dict)
        self.assertIn('psd', psd_2d)
        self.assertIn('freq_x', psd_2d)
        self.assertIn('freq_y', psd_2d)
        
        # PSD should be positive and properly sized
        self.assertTrue(np.all(psd_2d['psd'] >= 0))
        self.assertEqual(psd_2d['psd'].shape, self.sine_map.shape)
        
        # Test with smoothing
        psd_2d_smooth = calculate_power_spectral_density(
            self.sine_map,
            smooth_spectrum=True,
            smooth_window=3
        )
        self.assertEqual(psd_2d_smooth['psd'].shape, self.sine_map.shape)
    
    def test_calculate_surface_isotropy(self):
        """Test calculation of surface isotropy metrics."""
        # Test on perfectly isotropic surface (flat)
        iso_flat = calculate_surface_isotropy(self.flat_map)
        
        # Check return type and expected keys
        self.assertIsInstance(iso_flat, dict)
        for key in ['isotropy_index', 'directionality', 'dominant_angle']:
            self.assertIn(key, iso_flat)
            self.assertIsInstance(iso_flat[key], float)
        
        # Flat map should be highly isotropic
        self.assertGreaterEqual(iso_flat['isotropy_index'], 0.9)
        
        # Test on directional pattern
        stripes = np.zeros((40, 40), dtype=np.float32)
        stripes[:, ::4] = 1.0  # Vertical stripes
        iso_stripes = calculate_surface_isotropy(stripes)
        
        # Striped pattern should have high directionality
        self.assertGreater(iso_stripes['directionality'], 0.5)
        
        # Dominant angle should be around 0 or π (vertical direction)
        angle = iso_stripes['dominant_angle']
        self.assertTrue(
            np.isclose(abs(angle), 0.0, atol=0.2) or 
            np.isclose(abs(angle), np.pi, atol=0.2)
        )
        
        # Test error for 1D input
        with self.assertRaises(ValueError):
            calculate_surface_isotropy(self.profile_sine)
    
    def test_detect_surface_periodicity(self):
        """Test detection of surface periodicity."""
        # Test on periodic pattern
        periodic_result = detect_surface_periodicity(self.periodic_map, threshold=0.2)
        
        # Check return type and expected keys
        self.assertIsInstance(periodic_result, dict)
        for key in ['is_periodic', 'periods', 'strengths', 'directions']:
            self.assertIn(key, periodic_result)
        
        # Should detect periodicity
        self.assertTrue(periodic_result['is_periodic'])
        self.assertGreater(len(periodic_result['periods']), 0)
        
        # Test on random noise (should have low or no periodicity)
        noise_map = np.random.random((30, 30)).astype(np.float32)
        noise_result = detect_surface_periodicity(noise_map, threshold=0.5)
        
        # Either no periods or weak periodicity
        if noise_result['is_periodic']:
            self.assertLess(noise_result['strengths'][0], 0.7)  # Weak periodicity
        
        # Test error for 1D input
        with self.assertRaises(ValueError):
            detect_surface_periodicity(self.profile_sine)
    
    def test_calculate_autocorrelation(self):
        """Test calculation of autocorrelation."""
        # Test 1D autocorrelation
        acorr_1d = calculate_autocorrelation(self.profile_sine, normalize=True)
        
        # Check shape and properties
        self.assertEqual(len(acorr_1d), len(self.profile_sine))
        self.assertAlmostEqual(acorr_1d[0], 1.0, delta=0.01)  # Normalized at zero lag
        
        # Test periodicity in autocorrelation (should match sine periodicity)
        # Find first peak after zero lag
        peaks = np.where(np.diff(np.signbit(np.diff(acorr_1d))))[0] + 1
        peaks = peaks[peaks > 1]  # Skip the zero lag
        if len(peaks) > 0:
            first_peak = peaks[0]
            # Period should be around 25 samples (100 samples / 4 cycles)
            self.assertAlmostEqual(first_peak, 25, delta=5)
        
        # Test 2D autocorrelation
        acorr_2d = calculate_autocorrelation(
            self.sine_map, normalize=True, max_lag=10
        )
        
        # Check shape and properties
        self.assertEqual(acorr_2d.shape, (10, 10))  # Based on max_lag
        center = acorr_2d.shape[0] // 2, acorr_2d.shape[1] // 2
        if center[0] < acorr_2d.shape[0] and center[1] < acorr_2d.shape[1]:
            self.assertAlmostEqual(acorr_2d[center[0], center[1]], 1.0, delta=0.01)
        
        # Test with non-normalized output
        acorr_non_norm = calculate_autocorrelation(
            self.profile_flat, normalize=False
        )
        # Should be a constant value for flat input
        std_dev = np.std(acorr_non_norm)
        self.assertLessEqual(std_dev, 1e-5)
    
    def test_calculate_intercorrelation(self):
        """Test calculation of cross-correlation between height maps."""
        # Create a shifted version of profile
        shift = 10
        shifted_profile = np.roll(self.profile_sine, shift)
        
        # Calculate cross-correlation
        xcorr = calculate_intercorrelation(
            self.profile_sine, shifted_profile, normalize=True
        )
        
        # Find the peak position (should be at the shift amount)
        peak_idx = np.argmax(xcorr)
        self.assertAlmostEqual(peak_idx, shift, delta=2)
        
        # Test 2D cross-correlation
        shift_y, shift_x = 3, 4
        shifted_map = np.roll(np.roll(self.sine_map, shift_y, axis=0), shift_x, axis=1)
        
        xcorr_2d = calculate_intercorrelation(
            self.sine_map, shifted_map, normalize=True
        )
        
        # Find peak position
        peak_y, peak_x = np.unravel_index(np.argmax(xcorr_2d), xcorr_2d.shape)
        center_y, center_x = xcorr_2d.shape[0] // 2, xcorr_2d.shape[1] // 2
        
        # Peak should be offset from center by the shift amount
        self.assertAlmostEqual(peak_y - center_y, shift_y, delta=1)
        self.assertAlmostEqual(peak_x - center_x, shift_x, delta=1)
        
        # Test with different shapes
        with self.assertRaises(ValueError):
            calculate_intercorrelation(self.profile_sine, self.profile_sine[:50])
    
    def test_denoise_by_fft(self):
        """Test advanced FFT denoising with smooth transitions."""
        # Create test signal with noise
        noise_level = 0.3
        noisy_profile = self.profile_sine + np.random.normal(0, noise_level, self.profile_sine.shape)
        
        # Apply denoising with low-pass filter
        denoised = denoise_by_fft(
            noisy_profile,
            high_cutoff=0.2,  # Keep only low frequencies
            smooth_transition=True
        )
        
        # Check shape preservation
        self.assertEqual(denoised.shape, noisy_profile.shape)
        
        # Denoising should reduce variance
        self.assertLess(np.var(denoised - self.profile_sine), np.var(noisy_profile - self.profile_sine))
        
        # Test 2D denoising
        noisy_map = self.sine_map + np.random.normal(0, noise_level, self.sine_map.shape)
        
        # Apply band-pass filter
        denoised_2d = denoise_by_fft(
            noisy_map,
            low_cutoff=0.05,
            high_cutoff=0.2,
            filter_type='bandpass'
        )
        
        # Check shape preservation
        self.assertEqual(denoised_2d.shape, noisy_map.shape)
        
        # Test with and without windowing
        denoised_no_window = denoise_by_fft(
            noisy_profile,
            high_cutoff=0.2,
            apply_windowing=False
        )
        self.assertEqual(denoised_no_window.shape, noisy_profile.shape)
    
    def test_apply_continuous_wavelet_transform(self):
        """Test continuous wavelet transform on 1D profiles."""
        # Apply CWT to sine profile
        cwt_result = apply_continuous_wavelet_transform(
            self.profile_sine,
            wavelet='morl',
            num_scales=12
        )
        
        # Check return structure and shapes
        self.assertIsInstance(cwt_result, dict)
        self.assertIn('coefficients', cwt_result)
        self.assertIn('scales', cwt_result)
        self.assertIn('coi', cwt_result)
        
        # Check dimensions
        self.assertEqual(cwt_result['coefficients'].shape[1], len(self.profile_sine))
        self.assertEqual(len(cwt_result['scales']), cwt_result['coefficients'].shape[0])
        self.assertEqual(len(cwt_result['coi']), len(self.profile_sine))
        
        # Test with custom scales
        custom_scales = np.arange(1, 13)
        cwt_custom = apply_continuous_wavelet_transform(
            self.profile_sine,
            scales=custom_scales,
            wavelet='mexh'
        )
        
        self.assertEqual(len(cwt_custom['scales']), len(custom_scales))
        np.testing.assert_array_equal(cwt_custom['scales'], custom_scales)
        
        # Test with invalid wavelet
        with self.assertRaises(ValueError):
            apply_continuous_wavelet_transform(self.profile_sine, wavelet='invalid_wavelet')
        
        # Test with 2D input (should raise error)
        with self.assertRaises(ValueError):
            apply_continuous_wavelet_transform(self.sine_map)
    
    def test_apply_discrete_wavelet_transform(self):
        """Test discrete wavelet transform decomposition."""
        # Apply DWT to 1D profile
        dwt_result_1d = apply_discrete_wavelet_transform(
            self.profile_sine,
            wavelet='db4',
            level=3
        )
        
        # Check return structure
        self.assertIsInstance(dwt_result_1d, dict)
        self.assertIn('coeffs', dwt_result_1d)
        self.assertIn('rec_levels', dwt_result_1d)
        self.assertIn('details', dwt_result_1d)
        self.assertIn('approximation', dwt_result_1d)
        
        # Check number of reconstruction levels
        self.assertEqual(len(dwt_result_1d['rec_levels']), 3)
        
        # Apply DWT to 2D height map
        dwt_result_2d = apply_discrete_wavelet_transform(
            self.sine_map,
            wavelet='db4',
            level=2
        )
        
        # Check return structure for 2D
        self.assertIsInstance(dwt_result_2d, dict)
        self.assertIn('coeffs', dwt_result_2d)
        self.assertIn('rec_levels', dwt_result_2d)
        
        # Check number of reconstruction levels for 2D
        self.assertEqual(len(dwt_result_2d['rec_levels']), 2)
        
        # Test with automatic level determination
        dwt_auto = apply_discrete_wavelet_transform(
            self.profile_sine,
            wavelet='db2',
            level=None
        )
        
        self.assertGreater(len(dwt_auto['rec_levels']), 0)
    
    def test_discrete_wavelet_filtering(self):
        """Test filtering using discrete wavelet transform."""
        # Filter 1D profile keeping only certain levels
        filtered_1d = discrete_wavelet_filtering(
            self.profile_complex,
            wavelet='db4',
            level=3,
            keep_levels=[1, 2]  # Keep medium frequency details
        )
        
        # Check shape preservation
        self.assertEqual(filtered_1d.shape, self.profile_complex.shape)
        
        # Filter 2D height map
        filtered_2d = discrete_wavelet_filtering(
            self.sine_map,
            wavelet='db4',
            level=2,
            keep_levels=[0],  # Keep only first level details
            keep_approximation=False  # Remove approximation (low freq)
        )
        
        # Check shape preservation for 2D
        self.assertEqual(filtered_2d.shape, self.sine_map.shape)
        
        # Test with all levels and approximation
        filtered_all = discrete_wavelet_filtering(
            self.profile_sine,
            wavelet='db4',
            level=3,
            keep_levels=None,  # Keep all levels
            keep_approximation=True
        )
        
        # Should be very similar to original
        mse = np.mean((filtered_all - self.profile_sine)**2)
        self.assertLess(mse, 0.01)
    
    def test_get_available_wavelets(self):
        """Test retrieval of available wavelet families."""
        wavelets = get_available_wavelets()
        
        # Check return type and content
        self.assertIsInstance(wavelets, dict)
        self.assertIn('daubechies', wavelets)
        self.assertIn('symlet', wavelets)
        
        # Check some specific wavelets
        self.assertIn('db4', wavelets['daubechies'])
        self.assertIn('sym8', wavelets['symlet'])
        self.assertIn('morl', wavelets['morlet'])


if __name__ == '__main__':
    unittest.main()
