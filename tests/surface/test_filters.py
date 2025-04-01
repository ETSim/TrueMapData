#!/usr/bin/env python3
"""
Tests for Height Map Filtering & Analysis Module.

This module contains unit tests for the various filtering and analysis
functions provided in the height map filtering module.
"""

import numpy as np
import pytest
from scipy import ndimage, signal
import pywt

# Import the module to test
import tmd.surface.filters as filtering


class TestBasicFilters:
    """Test cases for basic filtering functions."""

    def setup_method(self):
        """Set up test data for each test."""
        # Create a simple synthetic height map with known features
        self.height_map_1d = np.sin(np.linspace(0, 10 * np.pi, 100)) + 0.2 * np.random.randn(100)
        
        # Create a 2D height map with combined low and high frequency features
        x = np.linspace(0, 5, 50)
        y = np.linspace(0, 5, 50)
        X, Y = np.meshgrid(x, y)
        # Low frequency component
        low_freq = np.sin(X) + np.cos(Y)
        # High frequency component (noise)
        high_freq = 0.2 * np.random.randn(50, 50)
        # Combined height map
        self.height_map_2d = low_freq + high_freq

    def test_gaussian_filter(self):
        """Test Gaussian filter on 2D height map."""
        # Apply filter with different sigma values
        filtered_small_sigma = filtering.apply_gaussian_filter(self.height_map_2d, sigma=0.5)
        filtered_large_sigma = filtering.apply_gaussian_filter(self.height_map_2d, sigma=2.0)
        
        # Check that output shape matches input
        assert filtered_small_sigma.shape == self.height_map_2d.shape
        assert filtered_large_sigma.shape == self.height_map_2d.shape
        
        # Check that larger sigma gives more smoothing (should have lower variance)
        assert np.var(filtered_large_sigma) < np.var(filtered_small_sigma)
        
        # Verify that filtering didn't modify the original data
        assert not np.array_equal(filtered_small_sigma, self.height_map_2d)

    def test_waviness_roughness_extraction(self):
        """Test waviness and roughness extraction."""
        # Extract waviness and roughness
        waviness = filtering.extract_waviness(self.height_map_2d, sigma=2.0)
        roughness = filtering.extract_roughness(self.height_map_2d, sigma=2.0)
        
        # Check that shapes match
        assert waviness.shape == self.height_map_2d.shape
        assert roughness.shape == self.height_map_2d.shape
        
        # Check that waviness + roughness approximately equals original (within small epsilon)
        combined = waviness + roughness
        assert np.allclose(combined, self.height_map_2d, rtol=1e-6)
        
        # Check that waviness has lower variance than original
        assert np.var(waviness) < np.var(self.height_map_2d)
        
        # Check that roughness has lower mean than original
        assert np.abs(np.mean(roughness)) < np.abs(np.mean(self.height_map_2d))

    def test_rms_calculations(self):
        """Test RMS roughness and waviness calculations."""
        # Calculate RMS values
        rms_roughness = filtering.calculate_rms_roughness(self.height_map_2d, sigma=2.0)
        rms_waviness = filtering.calculate_rms_waviness(self.height_map_2d, sigma=2.0)
        
        # Check that results are positive scalars
        assert isinstance(rms_roughness, float)
        assert isinstance(rms_waviness, float)
        assert rms_roughness > 0
        assert rms_waviness > 0
        
        # For our test data, waviness should have higher RMS than roughness
        assert rms_waviness > rms_roughness

    def test_gradient_calculations(self):
        """Test surface gradient calculations."""
        # Calculate gradients
        grad_x, grad_y = filtering.calculate_surface_gradient(self.height_map_2d)
        
        # Check shapes
        assert grad_x.shape == self.height_map_2d.shape
        assert grad_y.shape == self.height_map_2d.shape
        
        # Test with custom scale factor
        grad_x_scaled, grad_y_scaled = filtering.calculate_surface_gradient(
            self.height_map_2d, scale=2.0
        )
        
        # Check that scaling works correctly
        assert np.allclose(grad_x_scaled, grad_x * 2.0)
        assert np.allclose(grad_y_scaled, grad_y * 2.0)

    def test_slope_calculation(self):
        """Test slope calculation."""
        # Calculate slope
        slope = filtering.calculate_slope(self.height_map_2d)
        
        # Check shape
        assert slope.shape == self.height_map_2d.shape
        
        # Slope should be nonnegative
        assert np.all(slope >= 0)
        
        # Verify against manual gradient calculation
        grad_x, grad_y = filtering.calculate_surface_gradient(self.height_map_2d)
        manual_slope = np.sqrt(grad_x**2 + grad_y**2)
        assert np.allclose(slope, manual_slope)

    def test_median_filter(self):
        """Test median filter."""
        # Add some outliers (impulse noise) to the height map
        noisy_map = self.height_map_2d.copy()
        noisy_map[10:15, 10:15] = 10.0  # Add a "spike"
        
        # Apply median filter
        filtered = filtering.apply_median_filter(noisy_map, size=7)
        
        # Check that output shape matches input
        assert filtered.shape == noisy_map.shape
        
        # Check that outliers are reduced (median filter should remove spikes)
        assert np.max(filtered) < np.max(noisy_map)
        
        # Verify using scipy's implementation as reference
        reference = ndimage.median_filter(noisy_map, size=5)
        assert np.allclose(filtered, reference)


class TestAdvancedFilters:
    """Test cases for more advanced filtering techniques."""

    def setup_method(self):
        """Set up test data for each test."""
        # Create a 2D height map with mixed features
        x = np.linspace(0, 2 * np.pi, 50)
        y = np.linspace(0, 2 * np.pi, 50)
        X, Y = np.meshgrid(x, y)
        
        # Create different frequency components
        low_freq = np.sin(X) + np.cos(Y)
        med_freq = 0.5 * np.sin(5 * X) + 0.5 * np.cos(5 * Y)
        high_freq = 0.2 * np.sin(10 * X) + 0.2 * np.cos(10 * Y)
        noise = 0.1 * np.random.randn(50, 50)
        
        # Combined height map
        self.height_map = low_freq + med_freq + high_freq + noise
        
        # Create a smaller map for faster testing of complex filters
        self.small_map = self.height_map[0:20, 0:20]

    def test_morphological_filter(self):
        """Test morphological filters."""
        # Test with various operations
        operations = ["opening", "closing", "erosion", "dilation"]
        
        for op in operations:
            filtered = filtering.apply_morphological_filter(self.height_map, size=3, operation=op)
            
            # Check shape
            assert filtered.shape == self.height_map.shape
            
            # Specific checks for each operation
            if op == "erosion":
                # Erosion should reduce values
                assert np.mean(filtered) <= np.mean(self.height_map)
            elif op == "dilation":
                # Dilation should increase values
                assert np.mean(filtered) >= np.mean(self.height_map)
        
        # Test invalid operation raises ValueError
        with pytest.raises(ValueError):
            filtering.apply_morphological_filter(self.height_map, operation="invalid")

    def test_wavelet_filter(self):
        """Test wavelet-based filter."""
        # Apply wavelet filter
        filtered = filtering.apply_wavelet_filter(self.small_map, wavelet="db2", level=2)
        
        # Check shape
        assert filtered.shape[0] >= self.small_map.shape[0]
        assert filtered.shape[1] >= self.small_map.shape[1]
        
        # Check type
        assert filtered.dtype == self.small_map.dtype
        
        # Wavelet filtering should remove high frequencies, reducing variance
        assert np.var(filtered[:self.small_map.shape[0], :self.small_map.shape[1]]) < np.var(self.small_map)

    def test_fft_filter(self):
        """Test FFT-based filter."""
        # Test lowpass filter
        lowpass = filtering.apply_fft_filter(
            self.height_map, cutoff_high=5, filter_type="lowpass"
        )
        
        # Test highpass filter
        highpass = filtering.apply_fft_filter(
            self.height_map, cutoff_low=5, filter_type="highpass"
        )
        
        # Test bandpass filter
        bandpass = filtering.apply_fft_filter(
            self.height_map, cutoff_low=3, cutoff_high=8, filter_type="bandpass"
        )
        
        # Check shapes
        assert lowpass.shape == self.height_map.shape
        assert highpass.shape == self.height_map.shape
        assert bandpass.shape == self.height_map.shape
        
        # Lowpass should have lower variance than original
        assert np.var(lowpass) < np.var(self.height_map)
        
        # Highpass should have lower magnitude than original
        assert np.abs(np.mean(highpass)) < np.abs(np.mean(self.height_map))
        
        # Test invalid filter type
        with pytest.raises(ValueError):
            filtering.apply_fft_filter(self.height_map, filter_type="invalid")

    def test_klt_filter(self):
        """Test KLT filter."""
        # Test with different retention levels
        retention_levels = [0.6, 0.95]
        
        for retain in retention_levels:
            # Apply KLT filter
            filtered = filtering.apply_klt_filter(self.small_map, retain_components=retain)
            
            # Check shape
            assert filtered.shape == self.small_map.shape
            
            # Higher retention should preserve more detail
            if retain == 0.6:
                # Lower retention should lead to more filtering (lower MSE)
                assert np.mean((filtered - self.small_map)**2) > 0.001
            elif retain == 0.95:
                # Higher retention should preserve more detail (smaller MSE)
                assert np.mean((filtered - self.small_map)**2) < 0.001


class TestSpectralAnalysis:
    """Test cases for spectral analysis functions."""

    def setup_method(self):
        """Set up test data for spectral analysis tests."""
        # Create 1D profile with known period
        t = np.linspace(0, 10, 100)
        self.profile_1d = np.sin(2 * np.pi * 0.5 * t) + 0.5 * np.sin(2 * np.pi * 1.5 * t) + 0.1 * np.random.randn(100)
        
        # Create 2D height map with directional features
        x = np.linspace(0, 10, 50)
        y = np.linspace(0, 10, 50)
        X, Y = np.meshgrid(x, y)
        
        # Horizontal pattern (x-direction)
        horizontal = np.sin(X)
        # Vertical pattern (y-direction)
        vertical = 0.5 * np.cos(2 * Y)
        # Diagonal pattern
        diagonal = 0.3 * np.sin(X + Y)
        # Noise
        noise = 0.1 * np.random.randn(50, 50)
        
        # Combined height map
        self.height_map_2d = horizontal + vertical + diagonal + noise

    def test_frequency_spectrum(self):
        """Test frequency spectrum calculation."""
        # Test 1D spectrum
        spectrum_1d = filtering.calculate_frequency_spectrum(self.profile_1d, pixel_size=0.1)
        
        # Check that expected keys are present
        expected_keys = ['frequencies', 'magnitude', 'phase', 'wavelength']
        for key in expected_keys:
            assert key in spectrum_1d
        
        # Check shapes
        assert len(spectrum_1d['frequencies']) == 50  # n//2
        assert len(spectrum_1d['magnitude']) == 50
        
        # Test 2D spectrum
        spectrum_2d = filtering.calculate_frequency_spectrum(self.height_map_2d, pixel_size=0.2)
        
        # Check that expected keys are present
        expected_keys = ['freq_x', 'freq_y', 'magnitude', 'phase', 'wavelength', 'angle']
        for key in expected_keys:
            assert key in spectrum_2d
        
        # Check shapes
        assert spectrum_2d['freq_x'].shape == self.height_map_2d.shape
        assert spectrum_2d['magnitude'].shape == self.height_map_2d.shape

    def test_power_spectral_density(self):
        """Test power spectral density calculation."""
        # Test 1D PSD
        psd_1d = filtering.calculate_power_spectral_density(
            self.profile_1d, pixel_size=0.1, smooth_spectrum=True
        )
        
        # Check that expected keys are present
        expected_keys = ['frequencies', 'psd', 'wavelength']
        for key in expected_keys:
            assert key in psd_1d
        
        # Check shapes
        assert len(psd_1d['frequencies']) == 50  # n//2
        assert len(psd_1d['psd']) == 50
        
        # PSD should be nonnegative
        assert np.all(psd_1d['psd'] >= 0)
        
        # Test 2D PSD
        psd_2d = filtering.calculate_power_spectral_density(
            self.height_map_2d, pixel_size=0.2, smooth_spectrum=True
        )
        
        # Check that expected keys are present
        expected_keys = ['freq_x', 'freq_y', 'psd', 'wavelength', 'angle']
        for key in expected_keys:
            assert key in psd_2d
        
        # Check shapes
        assert psd_2d['psd'].shape == self.height_map_2d.shape
        
        # PSD should be nonnegative
        assert np.all(psd_2d['psd'] >= 0)

    def test_surface_isotropy(self):
        """Test surface isotropy metrics calculation."""
        # Calculate isotropy metrics
        isotropy_data = filtering.calculate_surface_isotropy(self.height_map_2d, pixel_size=0.2)
        
        # Check that expected keys are present
        expected_keys = ['isotropy_index', 'directionality', 'dominant_angle', 
                          'directional_strength', 'angle_bins']
        for key in expected_keys:
            assert key in isotropy_data
        
        # Check data types and ranges
        assert 0 <= isotropy_data['isotropy_index'] <= 1
        assert 0 <= isotropy_data['directionality'] <= 1
        assert -np.pi <= isotropy_data['dominant_angle'] <= np.pi
        
        # Check directional strength normalization
        assert np.isclose(np.sum(isotropy_data['directional_strength']), 1.0)
        
        # Test flat surface (should be perfectly isotropic)
        flat_map = np.ones((10, 10))
        flat_isotropy = filtering.calculate_surface_isotropy(flat_map)
        assert flat_isotropy['isotropy_index'] == 1.0
        assert flat_isotropy['directionality'] == 0.0

    def test_detect_surface_periodicity(self):
        """Test surface periodicity detection."""
        # Create a test pattern with known periodicity
        x = np.linspace(0, 10, 50)
        y = np.linspace(0, 10, 50)
        X, Y = np.meshgrid(x, y)
        
        # Horizontal periodic pattern (wavelength = 8)
        periodic_map = np.sin(2 * np.pi * X / 8)
        
        # Test periodicity detection
        periodicity = filtering.detect_surface_periodicity(periodic_map, pixel_size=0.2)
        
        # Check that expected keys are present
        expected_keys = ['is_periodic', 'periods', 'strengths', 'directions']
        for key in expected_keys:
            assert key in periodicity
        
        # Should detect periodicity
        assert periodicity['is_periodic'] is True
        
        # Should find at least one period
        assert len(periodicity['periods']) > 0
        
        # Test random noise (shouldn't be periodic)
        noise_map = np.random.randn(30, 30)
        noise_periodicity = filtering.detect_surface_periodicity(noise_map, threshold=0.9)
        
        # Check if it correctly identifies non-periodic surfaces
        # Note: This might be true or false depending on the threshold and randomness
        if noise_periodicity['is_periodic']:
            assert len(noise_periodicity['periods']) <= 2  # Should find at most a couple false periods


class TestCorrelationAndWavelets:
    """Test cases for correlation and wavelet analysis functions."""

    def setup_method(self):
        """Set up test data for correlation and wavelet tests."""
        # Create 1D profile with periodic pattern
        t = np.linspace(0, 10, 100)
        self.profile_1d = np.sin(2 * np.pi * 0.25 * t) + 0.1 * np.random.randn(100)
        
        # Create reference profile for cross-correlation tests
        self.reference_1d = np.cos(2 * np.pi * 0.25 * t) + 0.1 * np.random.randn(100)
        
        # Create small profiles for intercorrelation test
        self.profile_a = np.sin(np.linspace(0, 2*np.pi, 20))
        self.profile_b = -np.sin(np.linspace(0, 2*np.pi, 20))
        
        # Create 2D height maps
        x = np.linspace(0, 5, 30)
        y = np.linspace(0, 5, 30)
        X, Y = np.meshgrid(x, y)
        
        # Map with pattern
        self.height_map_2d = np.sin(X) + np.cos(Y) + 0.1 * np.random.randn(30, 30)
        
        # Shifted version for cross-correlation test
        self.shifted_2d = np.sin(X - 1) + np.cos(Y - 1) + 0.1 * np.random.randn(30, 30)

    def test_autocorrelation(self):
        """Test autocorrelation function."""
        # Calculate autocorrelation for 1D profile
        acorr_1d = filtering.calculate_autocorrelation(self.profile_1d, normalize=True)
        
        # Check shape
        assert len(acorr_1d) == len(self.profile_1d)
        
        # Autocorrelation should have maximum at lag 0
        assert np.isclose(acorr_1d[0], 1.0)
        
        # For a periodic signal, autocorrelation should show periodicity
        # Find peaks in autocorrelation
        peak_indices = signal.find_peaks(acorr_1d)[0]
        if len(peak_indices) > 1:
            # Average peak spacing should be around the period (T=4 => 100/4 = 25 samples)
            peak_spacing = np.mean(np.diff(peak_indices))
            assert 20 <= peak_spacing <= 30
        
        # Test 2D autocorrelation
        acorr_2d = filtering.calculate_autocorrelation(self.height_map_2d, normalize=True)
        
        # Check shape
        assert acorr_2d.shape[0] <= self.height_map_2d.shape[0]
        assert acorr_2d.shape[1] <= self.height_map_2d.shape[1]
        
        # Center should have maximum value
        center_y, center_x = acorr_2d.shape[0] // 2, acorr_2d.shape[1] // 2
        assert np.isclose(acorr_2d[center_y, center_x], 1.0)

    def test_intercorrelation(self):
        """Test intercorrelation function."""
        # Test with two small 1D profiles
        xcorr = filtering.calculate_intercorrelation(self.profile_a, self.profile_b, normalize=True)
        
        # Check shape
        assert len(xcorr) == len(self.profile_a)
        
        # Since profiles are negatives of each other, cross-correlation should have negative peak
        assert np.min(xcorr) < -0.5
        
        # Test with 2D height maps
        xcorr_2d = filtering.calculate_intercorrelation(self.height_map_2d, self.shifted_2d)
        
        # Check shape
        assert xcorr_2d.shape == self.height_map_2d.shape
        
        # Test with mismatched shapes
        with pytest.raises(ValueError):
            filtering.calculate_intercorrelation(self.profile_1d, self.profile_a)

    def test_denoise_by_fft(self):
        """Test FFT-based denoising."""
        # Create a noisy profile
        t = np.linspace(0, 10, 100)
        clean_signal = np.sin(2 * np.pi * 0.1 * t) + 0.5 * np.sin(2 * np.pi * 0.2 * t)
        noisy_signal = clean_signal + 0.3 * np.random.randn(100)
        
        # Apply FFT denoising
        denoised = filtering.denoise_by_fft(
            noisy_signal, high_cutoff=0.25, filter_type='lowpass', smooth_transition=False
        )
        
        # Check shape
        assert len(denoised) == len(noisy_signal)
        
        # Denoised signal should be closer to clean signal than noisy signal
        noise_mse = np.mean((noisy_signal - clean_signal)**2)
        denoised_mse = np.mean((denoised - clean_signal)**2)
        assert denoised_mse < noise_mse
        
        # Test 2D denoising
        denoised_2d = filtering.denoise_by_fft(
            self.height_map_2d, high_cutoff=0.5, filter_type='lowpass'
        )
        
        # Check shape
        assert denoised_2d.shape == self.height_map_2d.shape
        
        # Denoised map should have lower variance
        assert np.var(denoised_2d) < np.var(self.height_map_2d)

    def test_continuous_wavelet_transform(self):
        """Test continuous wavelet transform."""
        # Apply CWT
        cwt_data = filtering.apply_continuous_wavelet_transform(
            self.profile_1d, wavelet='morl', num_scales=16
        )
        
        # Check that expected keys are present
        expected_keys = ['coefficients', 'scales', 'coi']
        for key in expected_keys:
            assert key in cwt_data
        
        # Check shapes
        assert cwt_data['coefficients'].shape[0] == 16  # num_scales
        assert cwt_data['coefficients'].shape[1] == len(self.profile_1d)
        assert len(cwt_data['scales']) == 16
        assert len(cwt_data['coi']) == len(self.profile_1d)
        
        # Test with invalid wavelet
        with pytest.raises(ValueError):
            filtering.apply_continuous_wavelet_transform(self.profile_1d, wavelet='invalid')
        
        # Test with 2D input (should raise error)
        with pytest.raises(ValueError):
            filtering.apply_continuous_wavelet_transform(self.height_map_2d)

    def test_discrete_wavelet_transform(self):
        """Test discrete wavelet transform."""
        # Apply DWT to 1D profile
        dwt_data = filtering.apply_discrete_wavelet_transform(
            self.profile_1d, wavelet='db4', level=3
        )
        
        # Check that expected keys are present
        expected_keys = ['coeffs', 'rec_levels', 'details', 'approximation']
        for key in expected_keys:
            assert key in dwt_data
        
        # Check that we have correct levels
        assert len(dwt_data['details']) == 3  # 3 levels of details
        assert len(dwt_data['rec_levels']) == 3  # 3 reconstructed levels
        
        # Apply DWT to 2D height map
        dwt_data_2d = filtering.apply_discrete_wavelet_transform(
            self.height_map_2d, wavelet='db4', level=2
        )
        
        # Check that expected keys are present
        for key in expected_keys:
            assert key in dwt_data_2d
        
        # Check that we have correct levels
        assert len(dwt_data_2d['details']) == 2  # 2 levels of details
        assert len(dwt_data_2d['rec_levels']) == 2  # 2 reconstructed levels
        
        # Each detail level in 2D should be a tuple of (H, V, D) coefficients
        assert isinstance(dwt_data_2d['details'][0], tuple)
        assert len(dwt_data_2d['details'][0]) == 3  # (H, V, D)

    def test_discrete_wavelet_filtering(self):
        """Test wavelet-based filtering."""
        # Apply wavelet filtering to 1D profile
        filtered_1d = filtering.discrete_wavelet_filtering(
            self.profile_1d, wavelet='db4', level=3, keep_levels=[0, 1]
        )
        
        # Check shape
        assert len(filtered_1d) == len(self.profile_1d)
        
        # Apply wavelet filtering to 2D height map
        filtered_2d = filtering.discrete_wavelet_filtering(
            self.height_map_2d, wavelet='db4', level=2, keep_levels=[0], keep_approximation=True
        )
        
        # Check shape
        assert filtered_2d.shape == self.height_map_2d.shape
        
        # Filtered map should have lower variance (since we removed detail levels)
        assert np.var(filtered_2d) < np.var(self.height_map_2d)
        
        # Test with no levels kept and no approximation
        filtered_none = filtering.discrete_wavelet_filtering(
            self.profile_1d, keep_levels=[], keep_approximation=False
        )
        
        # Should be all zeros
        assert np.allclose(filtered_none, 0, atol=1e-10)

    def test_get_available_wavelets(self):
        """Test function to get available wavelets."""
        wavelets = filtering.get_available_wavelets()
        
        # Should be a dictionary
        assert isinstance(wavelets, dict)
        
        # Should contain expected families
        expected_families = ['coiflet', 'daubechies', 'symlet', 'discrete_meyer', 
                             'mexican_hat', 'morlet', 'gaussian']
        for family in expected_families:
            assert family in wavelets
        
        # Each family should be a list of string wavelets
        for family, wavelet_list in wavelets.items():
            assert isinstance(wavelet_list, list)
            assert all(isinstance(w, str) for w in wavelet_list)


if __name__ == "__main__":
    pytest.main(["-v", __file__])