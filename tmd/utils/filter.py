"""
filter.py

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
    return ndimage.gaussian_filter(height_map, sigma=sigma)

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
    return height_map - waviness

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

def calculate_surface_gradient(height_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the gradient of the height map in the x and y directions.
    
    Args:
        height_map (np.ndarray): 2D array of height values.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Gradients in the x and y directions.
    """
    grad_y, grad_x = np.gradient(height_map)
    return grad_x, grad_y

def calculate_slope(height_map: np.ndarray) -> np.ndarray:
    """
    Calculate the slope of the height map, defined as the magnitude of the gradient.
    
    Args:
        height_map (np.ndarray): 2D array of height values.
    
    Returns:
        np.ndarray: Array of slope values.
    """
    grad_x, grad_y = calculate_surface_gradient(height_map)
    return np.sqrt(grad_x**2 + grad_y**2)
