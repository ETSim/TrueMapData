"""
Utility functions for handling heightmaps.

This module provides functions for heightmap processing, analysis, and
transformation that are used by the model exporters.
"""

import numpy as np
import logging
from typing import Tuple, Optional, Union, List, Dict, Any

# Set up logging
logger = logging.getLogger(__name__)


def normalize_heightmap(
    heightmap: np.ndarray,
    target_min: float = 0.0,
    target_max: float = 1.0
) -> np.ndarray:
    """
    Normalize a heightmap to a specified range.
    
    Args:
        heightmap: 2D numpy array of height values
        target_min: Target minimum value
        target_max: Target maximum value
        
    Returns:
        Normalized heightmap
    """
    # Get current min and max
    h_min = np.min(heightmap)
    h_max = np.max(heightmap)
    h_range = h_max - h_min
    
    # Check for flat heightmap
    if h_range < 1e-10:
        return np.full_like(heightmap, target_min)
    
    # Normalize to [0, 1] then scale to target range
    normalized = (heightmap - h_min) / h_range
    normalized = normalized * (target_max - target_min) + target_min
    
    return normalized


def calculate_heightmap_normals(height_map: np.ndarray, z_scale: float = 1.0) -> np.ndarray:
    """
    Calculate normal vectors for a height map using gradient-based approach.
    
    Args:
        height_map: 2D array of height values
        z_scale: Optional scaling factor for height values affecting normal steepness
        
    Returns:
        3D array of normal vectors with shape (height, width, 3)
    """
    height, width = height_map.shape
    normals = np.zeros((height, width, 3), dtype=np.float32)
    
    # Scale factor affects the normal direction (higher values make terrain appear steeper)
    xy_scale = 1.0
    
    # Calculate gradients using optimized numpy operations
    gradient_x = np.zeros_like(height_map, dtype=np.float32)
    gradient_y = np.zeros_like(height_map, dtype=np.float32)
    
    # Interior points - central differences for better accuracy
    gradient_x[1:-1, 1:-1] = (height_map[1:-1, 2:] - height_map[1:-1, :-2]) / 2.0
    gradient_y[1:-1, 1:-1] = (height_map[2:, 1:-1] - height_map[:-2, 1:-1]) / 2.0
    
    # Boundary points - forward/backward differences for better accuracy
    # Left & right edges
    gradient_x[1:-1, 0] = height_map[1:-1, 1] - height_map[1:-1, 0]
    gradient_x[1:-1, -1] = height_map[1:-1, -1] - height_map[1:-1, -2]
    
    # Top & bottom edges
    gradient_y[0, 1:-1] = height_map[1, 1:-1] - height_map[0, 1:-1]
    gradient_y[-1, 1:-1] = height_map[-1, 1:-1] - height_map[-2, 1:-1]
    
    # Corners - diagonal differences
    gradient_x[0, 0] = height_map[0, 1] - height_map[0, 0]
    gradient_y[0, 0] = height_map[1, 0] - height_map[0, 0]
    
    gradient_x[0, -1] = height_map[0, -1] - height_map[0, -2]
    gradient_y[0, -1] = height_map[1, -1] - height_map[0, -1]
    
    gradient_x[-1, 0] = height_map[-1, 1] - height_map[-1, 0]
    gradient_y[-1, 0] = height_map[-1, 0] - height_map[-2, 0]
    
    gradient_x[-1, -1] = height_map[-1, -1] - height_map[-1, -2]
    gradient_y[-1, -1] = height_map[-1, -1] - height_map[-2, -1]
    
    # Apply scaling to control normal direction
    gradient_x *= xy_scale / z_scale 
    gradient_y *= xy_scale / z_scale
    
    # Construct normals (-grad_x, -grad_y, 1)
    normals[:, :, 0] = -gradient_x
    normals[:, :, 1] = -gradient_y
    normals[:, :, 2] = 1.0
    
    # Normalize to unit length
    norm = np.sqrt(np.sum(normals**2, axis=2, keepdims=True))
    # Avoid division by zero
    norm[norm < 1e-10] = 1.0
    normals /= norm
    
    return normals.astype(np.float32)


def calculate_terrain_complexity(
    height_map: np.ndarray, 
    smoothing: float = 1.0
) -> np.ndarray:
    """
    Calculate terrain complexity map based on gradients and curvature.
    
    This function computes a complexity map that identifies areas with high detail
    or significant features in the heightmap, which can guide adaptive triangulation.
    
    Args:
        height_map: 2D numpy array of height values
        smoothing: Optional smoothing factor (higher = smoother complexity map)
        
    Returns:
        2D array representing terrain complexity (normalized to 0.0-1.0 range)
    """
    try:
        from scipy.ndimage import gaussian_filter, sobel
        
        # Optional smoothing to reduce noise
        if smoothing > 0:
            smoothed = gaussian_filter(height_map, sigma=smoothing)
        else:
            smoothed = height_map
        
        # Calculate gradients in x and y directions
        grad_x = sobel(smoothed, axis=1)
        grad_y = sobel(smoothed, axis=0)
        
        # Gradient magnitude represents slope
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Calculate second derivatives for curvature
        grad_xx = np.diff(np.diff(smoothed, axis=1, append=0), axis=1, append=0)
        grad_yy = np.diff(np.diff(smoothed, axis=0, append=0), axis=0, append=0)
        
        # Use Laplacian as a measure of curvature
        laplacian = np.abs(grad_xx + grad_yy)
        
        # Combine slope and curvature for overall complexity
        complexity = (
            0.7 * gradient_mag + 
            0.3 * laplacian
        )
        
    except ImportError:
        # Fallback if scipy is not available - use simple gradient calculation
        logger.warning("SciPy not available, using simple gradient for complexity calculation")
        
        # Calculate gradients using numpy diff (less accurate than sobel)
        grad_x = np.zeros_like(height_map)
        grad_y = np.zeros_like(height_map)
        
        # Handle inner points
        grad_x[:, 1:-1] = (height_map[:, 2:] - height_map[:, :-2]) / 2.0
        grad_y[1:-1, :] = (height_map[2:, :] - height_map[:-2, :]) / 2.0
        
        # Handle edges
        grad_x[:, 0] = height_map[:, 1] - height_map[:, 0]
        grad_x[:, -1] = height_map[:, -1] - height_map[:, -2]
        grad_y[0, :] = height_map[1, :] - height_map[0, :]
        grad_y[-1, :] = height_map[-1, :] - height_map[-2, :]
        
        # Compute gradient magnitude
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Use gradient magnitude as complexity
        complexity = gradient_mag
    
    # Normalize to [0, 1] range
    complexity_min = np.min(complexity)
    complexity_max = np.max(complexity)
    
    if complexity_max > complexity_min:
        complexity = (complexity - complexity_min) / (complexity_max - complexity_min)
    else:
        complexity = np.zeros_like(complexity)
    
    return complexity


def sample_heightmap(
    height_map: np.ndarray, 
    x: float, 
    y: float, 
    interpolation: str = 'bilinear'
) -> float:
    """
    Sample height value from a heightmap with interpolation.
    
    Args:
        height_map: 2D numpy array of height values
        x, y: Coordinates to sample (can be floating point)
        interpolation: Interpolation method ('nearest', 'bilinear', or 'bicubic')
        
    Returns:
        Interpolated height value
        
    Raises:
        ValueError: If an invalid interpolation method is specified
    """
    rows, cols = height_map.shape
    
    # Ensure coordinates are within bounds
    x = min(max(0, x), cols - 1)
    y = min(max(0, y), rows - 1)
    
    if interpolation == 'nearest':
        # Simple nearest neighbor interpolation
        return height_map[int(round(y)), int(round(x))]
    
    elif interpolation == 'bilinear':
        # Bilinear interpolation
        x0, y0 = int(x), int(y)
        x1, y1 = min(x0 + 1, cols - 1), min(y0 + 1, rows - 1)
        
        # Calculate fractional parts
        fx, fy = x - x0, y - y0
        
        # Get corner values
        v00 = height_map[y0, x0]
        v01 = height_map[y0, x1]
        v10 = height_map[y1, x0]
        v11 = height_map[y1, x1]
        
        # Interpolate along x
        v0 = v00 * (1 - fx) + v01 * fx
        v1 = v10 * (1 - fx) + v11 * fx
        
        # Interpolate along y
        return v0 * (1 - fy) + v1 * fy
    
    elif interpolation == 'bicubic':
        try:
            from scipy.interpolate import RectBivariateSpline
            
            # Get bounds for interpolation window
            x0, y0 = max(0, int(x) - 1), max(0, int(y) - 1)
            x1, y1 = min(cols - 1, int(x) + 2), min(rows - 1, int(y) + 2)
            
            # Create local coordinate grid
            x_grid = np.arange(x0, x1 + 1)
            y_grid = np.arange(y0, y1 + 1)
            
            # Extract local patch
            local_patch = height_map[y0:y1+1, x0:x1+1]
            
            # Create spline interpolator
            spline = RectBivariateSpline(y_grid, x_grid, local_patch, kx=3, ky=3)
            
            # Evaluate at requested coordinates
            return float(spline(y, x))
        except ImportError:
            logger.warning("SciPy not available, falling back to bilinear interpolation")
            # Fall back to bilinear
            return sample_heightmap(height_map, x, y, 'bilinear')
    
    else:
        raise ValueError(f"Invalid interpolation method: {interpolation}")


def resample_heightmap(
    height_map: np.ndarray,
    target_shape: Tuple[int, int],
    interpolation: str = 'bilinear'
) -> np.ndarray:
    """
    Resample a heightmap to a target shape.
    
    Args:
        height_map: 2D numpy array of height values
        target_shape: Tuple of (rows, cols) for the target shape
        interpolation: Interpolation method ('nearest', 'bilinear')
        
    Returns:
        Resampled heightmap
    """
    try:
        import cv2
        
        # Map interpolation method to OpenCV interpolation flags
        interp_flags = {
            'nearest': cv2.INTER_NEAREST,
            'bilinear': cv2.INTER_LINEAR,
            'bicubic': cv2.INTER_CUBIC
        }
        
        # Use OpenCV to resize the heightmap
        interpolation_flag = interp_flags.get(interpolation, cv2.INTER_LINEAR)
        
        # OpenCV uses (width, height) format, numpy uses (rows, cols)
        target_cols, target_rows = target_shape[1], target_shape[0]
        
        # Resize the heightmap
        return cv2.resize(height_map, (target_cols, target_rows), interpolation=interpolation_flag)
        
    except ImportError:
        logger.warning("OpenCV not available, using manual resampling")
        
        # Manual resampling using numpy
        orig_rows, orig_cols = height_map.shape
        target_rows, target_cols = target_shape
        
        # Create coordinate grids
        x = np.linspace(0, orig_cols - 1, target_cols)
        y = np.linspace(0, orig_rows - 1, target_rows)
        
        # Create output array
        resampled = np.zeros(target_shape, dtype=height_map.dtype)
        
        # Sample each point using our sample_heightmap function
        for i in range(target_rows):
            for j in range(target_cols):
                resampled[i, j] = sample_heightmap(height_map, x[j], y[i], interpolation)
        
        return resampled


def generate_heightmap_texture(
    height_map: np.ndarray,
    colormap: str = 'terrain',
    resolution: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Generate a texture map from a heightmap using a colormap.
    
    Args:
        height_map: 2D numpy array of height values
        colormap: Name of the colormap to use (matplotlib colormap name)
        resolution: Optional target resolution (width, height)
        
    Returns:
        RGB texture map as a uint8 numpy array with shape (height, width, 3)
    """
    try:
        from matplotlib import cm
        import matplotlib.pyplot as plt
        
        # Normalize height map to [0, 1]
        normalized = normalize_heightmap(height_map)
        
        # Resize if resolution is specified
        if resolution:
            normalized = resample_heightmap(normalized, resolution, 'bicubic')
        
        # Apply colormap
        try:
            cmap = cm.get_cmap(colormap)
        except ValueError:
            logger.warning(f"Colormap '{colormap}' not found, using 'terrain' instead")
            cmap = cm.get_cmap('terrain')
            
        # Apply colormap to get RGBA values
        colored = cmap(normalized)
        
        # Convert to RGB and scale to [0, 255]
        rgb = (colored[:, :, :3] * 255).astype(np.uint8)
        
        return rgb
        
    except ImportError:
        logger.warning("Matplotlib not available for texture generation. Using grayscale.")
        
        # Normalize height map to [0, 255]
        normalized = normalize_heightmap(height_map, 0, 255).astype(np.uint8)
        
        # Resize if resolution is specified
        if resolution:
            normalized = resample_heightmap(normalized, resolution, 'bilinear')
        
        # Convert to grayscale RGB
        rgb = np.stack([normalized, normalized, normalized], axis=2)
        
        return rgb