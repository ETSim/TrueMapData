""".

====================================================
Height Map Filtering & Analysis Module
====================================================

This module provides a suite of filtering and analysis functions for TMD 
height maps or other 2D surface data. It includes:

1. Linear Filters (Gaussian) 
2. Robust Filter (Median)
3. Morphological Filters (Opening, Closing, etc.)
4. Wavelet-Based Filter
5. FFT-Based Filter
6. Waviness & Roughness Extraction
7. RMS Roughness & RMS Waviness
8. Gradient and Slope Computations

Many of these ideas align with ISO 16610 concepts (e.g., 
- ISO 16610-20: Linear filters in the frequency domain
- ISO 16610-31/32: Robust filters
- ISO 16610-40/41: Morphological filters
- ISO 16610-29: Spline wavelets
). 
"""

from typing import Tuple, Optional
import numpy as np
from scipy import ndimage
import pywt


def apply_gaussian_filter(height_map: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """.

    Apply a Gaussian filter to smooth the height map.

    Args:
        height_map (np.ndarray): 2D array of height values.
        sigma (float): Standard deviation for the Gaussian kernel.

    Returns:
        np.ndarray: Smoothed height map.
    """
    return ndimage.gaussian_filter(height_map.copy(), sigma=sigma)


def extract_waviness(height_map: np.ndarray, sigma: float = 10.0) -> np.ndarray:
    """.

    Extract the waviness component (low-frequency variations) of the height map
    by applying a large-sigma Gaussian filter.

    Args:
        height_map (np.ndarray): 2D array of height values.
        sigma (float): Standard deviation for Gaussian smoothing (default: 10.0).

    Returns:
        np.ndarray: The low-frequency (waviness) component.
    """
    return apply_gaussian_filter(height_map, sigma=sigma)


def extract_roughness(height_map: np.ndarray, sigma: float = 10.0) -> np.ndarray:
    """.

    Extract the roughness component (high-frequency variations) of the height map.
    Roughness = original - waviness.

    Args:
        height_map (np.ndarray): 2D array of height values.
        sigma (float): Standard deviation for Gaussian smoothing (default: 10.0).

    Returns:
        np.ndarray: The high-frequency (roughness) component.
    """
    waviness = extract_waviness(height_map, sigma=sigma)
    return height_map.copy() - waviness


def calculate_rms_roughness(height_map: np.ndarray, sigma: float = 10.0) -> float:
    """.

    Calculate the RMS roughness of the height map.

    Args:
        height_map (np.ndarray): 2D array of height values.
        sigma (float): Standard deviation for Gaussian smoothing (default: 10.0).

    Returns:
        float: The RMS roughness value.
    """
    roughness = extract_roughness(height_map, sigma=sigma)
    return np.sqrt(np.mean(roughness**2))


def calculate_rms_waviness(height_map: np.ndarray, sigma: float = 10.0) -> float:
    """.

    Calculate the RMS waviness of the height map.

    Args:
        height_map (np.ndarray): 2D array of height values.
        sigma (float): Standard deviation for Gaussian smoothing (default: 10.0).

    Returns:
        float: The RMS waviness value.
    """
    waviness = extract_waviness(height_map, sigma=sigma)
    return np.sqrt(np.mean(waviness**2))


def calculate_surface_gradient(
    height_map: np.ndarray,
    dx: float = 1.0,
    dy: float = 1.0,
    scale_factor: float = 5.0,
    scale: float = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """.

    Calculate the gradient of the height map in the x and y directions.

    Args:
        height_map (np.ndarray): 2D array of height values.
        dx (float): Grid spacing in x direction.
        dy (float): Grid spacing in y direction.
        scale_factor (float): Deprecated scale factor for backward compatibility.
        scale (float): Modern scale factor to apply to gradients.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Gradients in the x and y directions.
    """
    # Determine final scale
    actual_scale = scale if scale is not None else scale_factor

    rows, cols = height_map.shape
    grad_x = np.zeros_like(height_map)
    grad_y = np.zeros_like(height_map)

    # x-gradient (central difference)
    grad_x[:, 1:-1] = (height_map[:, 2:] - height_map[:, :-2]) / (2 * dx)
    grad_x[:, 0] = (height_map[:, 1] - height_map[:, 0]) / dx
    grad_x[:, -1] = (height_map[:, -1] - height_map[:, -2]) / dx

    # y-gradient (central difference)
    grad_y[1:-1, :] = (height_map[2:, :] - height_map[:-2, :]) / (2 * dy)
    grad_y[0, :] = (height_map[1, :] - height_map[0, :]) / dy
    grad_y[-1, :] = (height_map[-1, :] - height_map[-2, :]) / dy

    # Apply scale factor
    grad_x *= actual_scale * 5.0
    grad_y *= actual_scale * 5.0

    return grad_x, grad_y


def calculate_slope(
    height_map: np.ndarray, scale_factor: float = 5.0, scale: float = None
) -> np.ndarray:
    """.

    Calculate the slope (magnitude of the gradient) of the height map.

    Args:
        height_map (np.ndarray): 2D array of height values.
        scale_factor (float): Deprecated scale factor for backward compatibility.
        scale (float): Modern scale factor to apply to gradients.

    Returns:
        np.ndarray: Array of slope values.
    """
    actual_scale = scale if scale is not None else scale_factor
    grad_x, grad_y = calculate_surface_gradient(height_map, scale=actual_scale)
    return np.sqrt(grad_x**2 + grad_y**2)


def apply_median_filter(height_map: np.ndarray, size: int = 3) -> np.ndarray:
    """.

    Apply a median filter to the height map (ISO 16610-31/32 'robust' concept).
    Median filtering is often used to reduce impulse noise while preserving edges.

    Args:
        height_map (np.ndarray): 2D array of height values.
        size (int): The size of the filtering window (default: 3).

    Returns:
        np.ndarray: The filtered height map.
    """
    return ndimage.median_filter(height_map.copy(), size=size)


def apply_morphological_filter(
    height_map: np.ndarray, size: int = 3, operation: str = "opening"
) -> np.ndarray:
    """.

    Apply a basic morphological filter (ISO 16610-40/41). This example
    supports opening or closing, which can be used to remove or preserve
    certain topographic features.

    Args:
        height_map (np.ndarray): 2D array of height values.
        size (int): Size of the structuring element (default: 3).
        operation (str): Which morphological operation to perform. 
                         Options: "opening", "closing", "erosion", "dilation".

    Returns:
        np.ndarray: The morphologically filtered height map.
    """
    # Create a structuring element
    struct_elem = ndimage.generate_binary_structure(2, 1)
    struct_elem = ndimage.iterate_structure(struct_elem, size // 2)

    if operation == "opening":
        filtered = ndimage.grey_opening(height_map, footprint=struct_elem)
    elif operation == "closing":
        filtered = ndimage.grey_closing(height_map, footprint=struct_elem)
    elif operation == "erosion":
        filtered = ndimage.grey_erosion(height_map, footprint=struct_elem)
    elif operation == "dilation":
        filtered = ndimage.grey_dilation(height_map, footprint=struct_elem)
    else:
        raise ValueError(
            "Unsupported operation. Choose from 'opening', 'closing', 'erosion', or 'dilation'."
        )
    return filtered


def apply_wavelet_filter(
    height_map: np.ndarray, wavelet: str = "db2", level: int = 1
) -> np.ndarray:
    """.

    Apply a simple wavelet-based filter (ISO 16610-29 'spline wavelets').
    Decompose the image using a wavelet transform and reconstruct it
    without the highest-frequency components.

    Args:
        height_map (np.ndarray): 2D array of height values.
        wavelet (str): Wavelet name (default: 'db2').
        level (int): Decomposition level (default: 1).

    Returns:
        np.ndarray: Reconstructed height map (low-frequency part).
    """
    # Perform 2D discrete wavelet transform
    coeffs = pywt.wavedec2(height_map, wavelet=wavelet, level=level)
    # Zero out the highest-frequency detail coefficients to remove them
    approx_coeffs = [coeffs[0]] + [(None, None, None) for _ in coeffs[1:]]
    # Reconstruct the filtered map
    filtered_map = pywt.waverec2(approx_coeffs, wavelet=wavelet)
    return filtered_map.astype(height_map.dtype)


def apply_fft_filter(
    height_map: np.ndarray,
    cutoff_low: Optional[float] = None,
    cutoff_high: Optional[float] = None,
    filter_type: str = "lowpass",
) -> np.ndarray:
    """.

    Apply an FFT-based filter (ISO 16610-20 'linear filters' in frequency domain).
    You can create a low-pass, high-pass, or band-pass filter by specifying
    cutoff frequencies.

    Args:
        height_map (np.ndarray): 2D array of height values.
        cutoff_low (float, optional): Low cutoff frequency (for high-pass or band-pass).
        cutoff_high (float, optional): High cutoff frequency (for low-pass or band-pass).
        filter_type (str): Type of filter to apply.
                          Options: "lowpass", "highpass", "bandpass".
                          - "lowpass" uses cutoff_high
                          - "highpass" uses cutoff_low
                          - "bandpass" uses both cutoff_low and cutoff_high

    Returns:
        np.ndarray: The filtered height map in the spatial domain.
    """
    # Perform 2D FFT
    fft_data = np.fft.fft2(height_map)
    fft_shifted = np.fft.fftshift(fft_data)

    # Get image size
    rows, cols = height_map.shape
    crow, ccol = rows // 2, cols // 2  # center coordinates

    # Create a frequency mask
    mask = np.ones((rows, cols), dtype=np.float32)

    # Helper function to compute the radius from the center
    def freq_radius(r, c):
        return np.sqrt((r - crow)**2 + (c - ccol)**2)

    # Build the mask by iterating over each frequency cell
    for r in range(rows):
        for c in range(cols):
            radius = freq_radius(r, c)

            if filter_type == "lowpass":
                # Keep frequencies <= cutoff_high
                if cutoff_high is not None and radius > cutoff_high:
                    mask[r, c] = 0.0

            elif filter_type == "highpass":
                # Keep frequencies >= cutoff_low
                if cutoff_low is not None and radius < cutoff_low:
                    mask[r, c] = 0.0

            elif filter_type == "bandpass":
                # Keep frequencies between cutoff_low and cutoff_high
                if cutoff_low is not None and radius < cutoff_low:
                    mask[r, c] = 0.0
                if cutoff_high is not None and radius > cutoff_high:
                    mask[r, c] = 0.0

            else:
                raise ValueError(
                    "Invalid filter_type. Use 'lowpass', 'highpass', or 'bandpass'."
                )

    # Apply the mask
    fft_filtered = fft_shifted * mask

    # Inverse shift and inverse FFT
    fft_unshifted = np.fft.ifftshift(fft_filtered)
    filtered_map = np.fft.ifft2(fft_unshifted)

    # Return the real part of the inverse FFT
    return np.real(filtered_map)
