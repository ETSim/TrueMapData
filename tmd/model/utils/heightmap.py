"""Heightmap processing utilities."""

import numpy as np
from typing import Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

def validate_heightmap(heightmap: np.ndarray) -> bool:
    """
    Validate a heightmap array.
    
    Args:
        heightmap: 2D numpy array to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    if heightmap is None:
        return False
    if not isinstance(heightmap, np.ndarray):
        return False
    if heightmap.ndim != 2:
        return False
    if heightmap.size == 0:
        return False
    if np.any(np.isnan(heightmap)):
        return False
    return True

def normalize_heightmap(heightmap: np.ndarray) -> np.ndarray:
    """
    Normalize heightmap values to range [0,1].
    
    Args:
        heightmap: Input heightmap array
        
    Returns:
        Normalized heightmap array
    """
    h_min = np.min(heightmap)
    h_max = np.max(heightmap)
    
    if h_max > h_min:
        return (heightmap - h_min) / (h_max - h_min)
    return np.zeros_like(heightmap)

def get_heightmap_stats(heightmap: np.ndarray) -> dict:
    """
    Get statistical information about a heightmap.
    
    Args:
        heightmap: Input heightmap array
        
    Returns:
        Dictionary containing heightmap statistics
    """
    return {
        'min': np.min(heightmap),
        'max': np.max(heightmap),
        'mean': np.mean(heightmap),
        'std': np.std(heightmap),
        'shape': heightmap.shape,
        'size': heightmap.size,
        'dtype': str(heightmap.dtype)
    }

def sample_heightmap(heightmap: np.ndarray, x: float, y: float) -> float:
    """
    Sample heightmap at floating point coordinates using bilinear interpolation.
    
    Args:
        heightmap: Input heightmap array
        x, y: Coordinates to sample
        
    Returns:
        Interpolated height value
    """
    h, w = heightmap.shape
    
    # Get integer coordinates
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    
    # Get fractional parts
    fx = x - x0
    fy = y - y0
    
    # Bilinear interpolation
    h00 = heightmap[y0, x0]
    h10 = heightmap[y0, x1]
    h01 = heightmap[y1, x0]
    h11 = heightmap[y1, x1]
    
    h0 = h00 * (1 - fx) + h10 * fx
    h1 = h01 * (1 - fx) + h11 * fx
    
    return h0 * (1 - fy) + h1 * fy

def resize_heightmap(heightmap: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Resize heightmap to specified dimensions.
    
    Args:
        heightmap: Input heightmap array
        width: Target width
        height: Target height
        
    Returns:
        Resized heightmap array
    """
    from scipy.ndimage import zoom
    
    # Calculate zoom factors
    zoom_y = height / heightmap.shape[0]
    zoom_x = width / heightmap.shape[1]
    
    # Perform resize
    return zoom(heightmap, (zoom_y, zoom_x), order=1)

def smooth_heightmap(heightmap: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Apply Gaussian smoothing to heightmap.
    
    Args:
        heightmap: Input heightmap array
        sigma: Smoothing radius
        
    Returns:
        Smoothed heightmap array
    """
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(heightmap, sigma=sigma)
