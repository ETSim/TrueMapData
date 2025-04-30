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


def to_16bit_grayscale(self, height_map: np.ndarray) -> np.ndarray:
        """
        Convert a heightmap to 16-bit grayscale format.
        
        Args:
            height_map: Input heightmap array
        Returns:
            16-bit normalized heightmap
        """
        
        # Ensure floating point for calculations
        height_map = height_map.astype(np.float32)
        
        # Normalize to [0, 1] range
        min_val = np.min(height_map)
        max_val = np.max(height_map)
        height_range = max_val - min_val
        
        if height_range > 0:
            height_map = (height_map - min_val) / height_range
        else:
            height_map = np.zeros_like(height_map)
        
        # Convert to 16-bit integer range [0, 65535]
        height_map = (height_map * 65535).astype(np.uint16)
        
        # Convert back to float32 but preserve 16-bit precision
        height_map = height_map.astype(np.float32) / 65535.0
        
        logger.debug(f"Converted heightmap: shape={height_map.shape}, dtype={height_map.dtype}, range=[{height_map.min():.3f}, {height_map.max():.3f}]")
        
        return height_map
    
def calculate_terrain_complexity(heightmap: np.ndarray, smoothing: float = 0.0) -> np.ndarray:
    """
    Calculate terrain complexity based on heightmap.
    
    Args:
        heightmap: Input heightmap array
        smoothing: Optional smoothing radius for the complexity map (default: 0.0)
        
    Returns:
        2D array representing local terrain complexity
    """
    # Calculate gradients
    gradient_x = np.gradient(heightmap, axis=1)
    gradient_y = np.gradient(heightmap, axis=0)
    
    # Calculate complexity as absolute gradients
    complexity = np.abs(gradient_x) + np.abs(gradient_y)
    
    # Apply optional smoothing
    if smoothing > 0:
        from scipy.ndimage import gaussian_filter
        complexity = gaussian_filter(complexity, sigma=smoothing)
    
    return complexity

def calculate_heightmap_center(heightmap: np.ndarray) -> Tuple[float, float]:
    """
    Calculate the center of a heightmap.
    
    Args:
        heightmap: Input heightmap array
    Returns:
        Tuple of (center_x, center_y)
    """
    
    h, w = heightmap.shape
    return (w / 2.0, h / 2.0)

def resample_heightmap(heightmap: np.ndarray, target_shape: tuple, method: str = 'bilinear') -> np.ndarray:
    """
    Resample a heightmap to a target shape using the specified interpolation method.
    
    Args:
        heightmap: Input heightmap array
        target_shape: Target shape (height, width)
        method: Interpolation method ('nearest', 'bilinear', 'bicubic')
        
    Returns:
        Resampled heightmap
    """
    from scipy.ndimage import zoom
    
    # Convert method to zoom order
    order_map = {
        'nearest': 0,
        'bilinear': 1, 
        'bicubic': 3
    }
    order = order_map.get(method.lower(), 1)  # Default to bilinear
    
    # Calculate zoom factors
    zoom_y = target_shape[0] / heightmap.shape[0]
    zoom_x = target_shape[1] / heightmap.shape[1]
    
    # Perform resize
    return zoom(heightmap, (zoom_y, zoom_x), order=order)