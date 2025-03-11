"""
This module provides filtering functions for processing TMD height maps.
It includes functions for Gaussian smoothing, and for extracting the 
waviness (low-frequency) and roughness (high-frequency) components of a height map.

Functions:
    - apply_gaussian_filter: Smooth the height map using a Gaussian kernel.
    - extract_waviness: Extract the low-frequency component (waviness) from the height map.
    - extract_roughness: Extract the high-frequency component (roughness) by subtraction.
    - calculate_rms_roughness: Compute the RMS roughness of the height map.
    - calculate_rms_waviness: Compute the RMS waviness of the height map.
    - calculate_surface_gradient: Compute the gradients in the x and y directions.
    - calculate_slope: Compute the slope (magnitude of the gradient) of the height map.
"""

import numpy as np
from scipy import ndimage
from typing import Tuple

def apply_gaussian_filter(height_map: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Apply a Gaussian filter to smooth the height map.
    
    Args:
        height_map (np.ndarray): 2D array of height values.
        sigma (float): Standard deviation for the Gaussian kernel.
    
    Returns:
        np.ndarray: Smoothed height map.
    """
    # Create a copy to ensure we don't modify the original
    return ndimage.gaussian_filter(height_map.copy(), sigma=sigma)

def extract_waviness(height_map: np.ndarray, sigma: float = 10.0) -> np.ndarray:
    """
    Extract the waviness component (low-frequency variations) of the height map.
    
    A large sigma is used to capture the general trend (waviness) of the surface.
    
    Args:
        height_map (np.ndarray): 2D array of height values.
        sigma (float): Standard deviation for Gaussian smoothing (default: 10.0).
        
    Returns:
        np.ndarray: The low-frequency (waviness) component.
    """
    # Larger sigma should extract lower frequency features
    return apply_gaussian_filter(height_map, sigma=sigma)

def extract_roughness(height_map: np.ndarray, sigma: float = 10.0) -> np.ndarray:
    """
    Extract the roughness component (high-frequency variations) of the height map.
    
    The roughness is computed as the difference between the original height map and 
    its smoothed (waviness) version.
    
    Args:
        height_map (np.ndarray): 2D array of height values.
        sigma (float): Standard deviation for Gaussian smoothing used for waviness extraction.
                       (default: 10.0)
        
    Returns:
        np.ndarray: The high-frequency (roughness) component.
    """
    waviness = extract_waviness(height_map, sigma=sigma)
    return height_map.copy() - waviness

def calculate_rms_roughness(height_map: np.ndarray, sigma: float = 10.0) -> float:
    """
    Calculate the root mean square (RMS) roughness of the height map.
    
    RMS roughness is defined as the square root of the mean squared differences 
    between the original and the low-frequency (waviness) component.
    
    Args:
        height_map (np.ndarray): 2D array of height values.
        sigma (float): Standard deviation for Gaussian smoothing (default: 10.0).
    
    Returns:
        float: The RMS roughness value.
    """
    roughness = extract_roughness(height_map, sigma=sigma)
    return np.sqrt(np.mean(roughness**2))

def calculate_rms_waviness(height_map: np.ndarray, sigma: float = 10.0) -> float:
    """
    Calculate the root mean square (RMS) waviness of the height map.
    
    This is computed as the RMS value of the low-frequency (waviness) component.
    
    Args:
        height_map (np.ndarray): 2D array of height values.
        sigma (float): Standard deviation for Gaussian smoothing (default: 10.0).
    
    Returns:
        float: The RMS waviness value.
    """
    waviness = extract_waviness(height_map, sigma=sigma)
    return np.sqrt(np.mean(waviness**2))

def calculate_surface_gradient(height_map: np.ndarray, dx: float = 1.0, dy: float = 1.0, scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the gradient of the height map in the x and y directions.
    
    Args:
        height_map (np.ndarray): 2D array of height values.
        dx (float): Grid spacing in x direction.
        dy (float): Grid spacing in y direction.
        scale_factor (float): Scale factor to apply to gradients.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Gradients in the x and y directions.
    """
    # Calculate the gradients using the central difference method
    # Instead of using np.gradient, we'll do the calculation explicitly to make scaling more controllable
    rows, cols = height_map.shape
    
    # Preallocate gradient arrays
    grad_x = np.zeros_like(height_map)
    grad_y = np.zeros_like(height_map)
    
    # Calculate gradient in x direction (central differences for interior, forward/backward at edges)
    grad_x[:, 1:-1] = (height_map[:, 2:] - height_map[:, :-2]) / (2 * dx)
    grad_x[:, 0] = (height_map[:, 1] - height_map[:, 0]) / dx  # Forward difference at left edge
    grad_x[:, -1] = (height_map[:, -1] - height_map[:, -2]) / dx  # Backward difference at right edge
    
    # Calculate gradient in y direction (central differences for interior, forward/backward at edges)
    grad_y[1:-1, :] = (height_map[2:, :] - height_map[:-2, :]) / (2 * dy)
    grad_y[0, :] = (height_map[1, :] - height_map[0, :]) / dy  # Forward difference at top edge
    grad_y[-1, :] = (height_map[-1, :] - height_map[-2, :]) / dy  # Backward difference at bottom edge
    
    # Apply scale factor
    grad_x = grad_x * scale
    grad_y = grad_y * scale
    
    return grad_x, grad_y

def calculate_slope(height_map: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """
    Calculate the slope of the height map, defined as the magnitude of the gradient.
    
    Args:
        height_map (np.ndarray): 2D array of height values.
        scale_factor (float): Scale factor to apply to gradients.
    
    Returns:
        np.ndarray: Array of slope values.
    """
    # Use same scale factor for consistency
    grad_x, grad_y = calculate_surface_gradient(height_map, scale=scale)
    return np.sqrt(grad_x**2 + grad_y**2)
