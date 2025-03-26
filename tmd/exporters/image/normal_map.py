"""
Normal map generation module for heightmaps.

This module provides functionality to generate normal maps from height maps.
Normal maps are used in 3D rendering to add surface detail without additional geometry.
"""

import os
import numpy as np
import logging
from typing import Optional, Tuple, Union, Dict, Any

from .utils import ensure_directory_exists
from .image_io import save_image

# Set up logging
logger = logging.getLogger(__name__)

def create_normal_map(
    height_map: np.ndarray,
    z_scale: float = 1.0,
    normalize: bool = True,
    output_format: str = "rgb",
    x_length: float = None,
    y_length: float = None,
    dx: float = None,  # Add explicit dx parameter
    dy: float = None,  # Add explicit dy parameter
    **kwargs
) -> np.ndarray:
    """
    Create a normal map from a height map.
    
    Args:
        height_map: 2D numpy array of height values
        z_scale: Scale factor for height values in normal calculation
        normalize: Whether to normalize the height map to [0,1] before processing
        output_format: Format of output normal map ("rgb" or "xyz")
        x_length: Physical length in X direction for correct aspect ratio
        y_length: Physical length in Y direction for correct aspect ratio
        dx: Explicit X step size (overrides x_length)
        dy: Explicit Y step size (overrides y_length)
        **kwargs: Additional options including metadata
        
    Returns:
        3D numpy array with normal vectors (H,W,3)
    """
    # Normalize height map if requested
    if normalize and (np.max(height_map) > 1.0 or np.min(height_map) < 0.0):
        height_norm = (height_map - np.min(height_map)) / (np.max(height_map) - np.min(height_map))
    else:
        height_norm = height_map.copy()
    
    # Get metadata for scaling if available
    metadata = kwargs.get('metadata', {})
    
    # Use provided dx/dy if explicitly specified
    if dx is None or dy is None:
        # Otherwise calculate from x_length/y_length
        # Use provided x_length and y_length or extract from metadata if available
        if x_length is None and 'x_length' in metadata:
            x_length = metadata['x_length']
        if y_length is None and 'y_length' in metadata:
            y_length = metadata['y_length']
            
        # Default to aspect ratio if not provided
        if x_length is None or y_length is None:
            aspect_ratio = height_map.shape[1] / height_map.shape[0] if height_map.shape[0] > 0 else 1.0
            if x_length is None and y_length is None:
                x_length = aspect_ratio
                y_length = 1.0
            elif x_length is None:
                x_length = y_length * aspect_ratio
            elif y_length is None:
                y_length = x_length / aspect_ratio
        
        # Calculate pixel size for proper scaling
        height, width = height_map.shape
        dx = x_length / width if width > 1 else 1.0
        dy = y_length / height if height > 1 else 1.0
    
    # Calculate gradients in x and y directions
    grad_x = np.zeros_like(height_norm)
    grad_y = np.zeros_like(height_norm)
    
    # Use central differences for interior points with correct scaling
    grad_x[1:-1, 1:-1] = (height_norm[1:-1, 2:] - height_norm[1:-1, :-2]) / (2.0 * dx)
    grad_y[1:-1, 1:-1] = (height_norm[2:, 1:-1] - height_norm[:-2, 1:-1]) / (2.0 * dy)
    
    # Use forward/backward differences for edges
    grad_x[1:-1, 0] = (height_norm[1:-1, 1] - height_norm[1:-1, 0]) / dx
    grad_x[1:-1, -1] = (height_norm[1:-1, -1] - height_norm[1:-1, -2]) / dx
    grad_y[0, 1:-1] = (height_norm[1, 1:-1] - height_norm[0, 1:-1]) / dy
    grad_y[-1, 1:-1] = (height_norm[-1, 1:-1] - height_norm[-2, 1:-1]) / dy
    
    # Handle corners
    grad_x[0, 0] = grad_x[0, 1]
    grad_y[0, 0] = grad_y[1, 0]
    grad_x[0, -1] = grad_x[0, -2]
    grad_y[0, -1] = grad_y[1, -1]
    grad_x[-1, 0] = grad_x[-1, 1]
    grad_y[-1, 0] = grad_y[-2, 0]
    grad_x[-1, -1] = grad_x[-1, -2]
    grad_y[-1, -1] = grad_y[-2, -1]
    
    # Create normal map vectors
    normal_map = np.zeros((height_norm.shape[0], height_norm.shape[1], 3), dtype=np.float32)
    
    # Standard tangent-space normal map encoding
    # Note that -grad_x and -grad_y are used because the gradients point in the direction of increasing height
    # But we want normals pointing away from the surface
    normal_map[:, :, 0] = -grad_x * z_scale  # X (tangent)
    normal_map[:, :, 1] = -grad_y * z_scale  # Y (bitangent)
    normal_map[:, :, 2] = 1.0                # Z (always points out)
    
    # Normalize to unit length
    norm = np.sqrt(np.sum(normal_map**2, axis=2, keepdims=True))
    normal_map = normal_map / np.maximum(norm, 1e-10)  # Avoid division by zero
    
    # Convert to desired output format
    if output_format.lower() == "xyz":
        # XYZ format: X=right, Y=forward, Z=up (left-handed coordinate system)
        # No change needed as this is our internal format
        pass
    elif output_format.lower() == "opengl":
        # OpenGL format: X=right, Y=up, Z=toward viewer (right-handed coordinate system)
        # Flip Y component because OpenGL's Y is up, not down
        normal_map[:, :, 1] *= -1
    
    return normal_map

def export_normal_map(
    height_map: np.ndarray,
    output_path: str,
    z_scale: float = 1.0,
    output_format: str = "rgb",
    normalize: bool = False,
    bit_depth: int = 8,
    **kwargs
) -> Optional[str]:
    """
    Generate and export a normal map from a height map.
    
    Args:
        height_map: 2D numpy array of height values
        output_path: Output filepath
        z_scale: Scale factor for height values in normal calculation
        output_format: Format of output normal map ("rgb" or "xyz")
        normalize: Whether to normalize the height map before processing
        bit_depth: Bit depth for output image (8 or 16)
        **kwargs: Additional options including metadata
        
    Returns:
        Path to the saved image or None if failed
    """
    try:
        # Ensure output directory exists
        if not ensure_directory_exists(output_path):
            logger.error(f"Failed to create output directory for {output_path}")
            return None
        
        # Create normal map
        normal_map = create_normal_map(
            height_map=height_map,
            z_scale=z_scale,
            normalize=normalize,
            output_format=output_format,
            **kwargs
        )
        
        # Convert from [-1,1] to [0,1] range for image export
        normal_map_export = (normal_map + 1.0) * 0.5
        
        # Save image
        return save_image(normal_map_export, output_path, bit_depth=bit_depth)
        
    except Exception as e:
        logger.error(f"Error exporting normal map: {e}")
        import traceback
        traceback.print_exc()
        return None

def normal_map_to_rgb(normal_map: np.ndarray) -> np.ndarray:
    """
    Convert a normal map from [-1,1] range to RGB [0,255] range.
    
    Args:
        normal_map: 3D numpy array with normal vectors in range [-1,1]
        
    Returns:
        3D numpy array with RGB values in range [0,255]
    """
    # Convert from [-1,1] to [0,1] range
    rgb_map = (normal_map + 1.0) * 0.5
    
    # Convert to 8-bit
    return (rgb_map * 255).astype(np.uint8)

def rgb_to_normal_map(rgb_image: np.ndarray) -> np.ndarray:
    """
    Convert an RGB normal map image to normal vectors in [-1,1] range.
    
    Args:
        rgb_image: 3D numpy array with RGB values
        
    Returns:
        3D numpy array with normal vectors in range [-1,1]
    """
    # Convert to float and normalize to [0,1] range
    rgb_float = rgb_image.astype(np.float32) / 255.0
    
    # Convert from [0,1] to [-1,1] range
    normal_map = rgb_float * 2.0 - 1.0
    
    # Ensure unit length
    norm = np.sqrt(np.sum(normal_map**2, axis=2, keepdims=True))
    return normal_map / np.maximum(norm, 1e-10)  # Avoid division by zero

def convert_heightmap_to_normal_map(
    height_map: np.ndarray, 
    z_scale: float = 10.0,
    **kwargs
) -> np.ndarray:
    """
    Generate a normal map from a heightmap.
    
    Args:
        height_map: 2D array of height values
        z_scale: Scale factor for z-axis
        **kwargs: Additional options including metadata
        
    Returns:
        3D array of normal vectors (nx, ny, nz) for each pixel
    """
    try:
        import cv2
    except ImportError:
        logger.error("OpenCV is required for normal map generation.")
        return np.zeros((*height_map.shape, 3), dtype=np.float32)
    
    # Get metadata for scaling if available
    metadata = kwargs.get('metadata', {})
    
    # Use x_length and y_length from metadata if available
    x_length = metadata.get('x_length')
    y_length = metadata.get('y_length')
        
    # Default to aspect ratio if not provided
    if x_length is None or y_length is None:
        aspect_ratio = height_map.shape[1] / height_map.shape[0] if height_map.shape[0] > 0 else 1.0
        if x_length is None and y_length is None:
            x_length = aspect_ratio
            y_length = 1.0
        elif x_length is None:
            x_length = y_length * aspect_ratio
        elif y_length is None:
            y_length = x_length / aspect_ratio
    
    # Calculate pixel size for proper scaling
    height, width = height_map.shape
    dx = x_length / width if width > 1 else 1.0
    dy = y_length / height if height > 1 else 1.0
    
    # For consistent results, normalize the height map first
    height_map_norm = height_map.copy()
    h_min, h_max = np.min(height_map_norm), np.max(height_map_norm)
    if h_max > h_min:
        height_map_norm = (height_map_norm - h_min) / (h_max - h_min)
    
    # Apply a small amount of blurring to reduce noise
    height_map_smooth = cv2.GaussianBlur(height_map_norm.astype(np.float32), (0, 0), 0.5)
    normal_map = np.zeros((height_map.shape[0], height_map.shape[1], 3), dtype=np.float32)
    
    grad_x = cv2.Sobel(height_map_smooth, cv2.CV_32F, 1, 0, ksize=3) * (1.0 / dx)
    grad_y = cv2.Sobel(height_map_smooth, cv2.CV_32F, 0, 1, ksize=3) * (1.0 / dy)
    
    # Apply z_scale (higher values make normals more aligned with z-axis)
    grad_x = grad_x / z_scale
    grad_y = grad_y / z_scale
    
    # Create normal vectors
    normal_map = np.zeros((height_map.shape[0], height_map.shape[1], 3), dtype=np.float32)
    
    # Standard tangent-space normal map encoding
    normal_map[..., 0] = -grad_x  # X component (tangent)
    normal_map[..., 1] = -grad_y  # Y component (bitangent)
    normal_map[..., 2] = 1.0      # Z component (always points out)
    
    # Normalize vectors to unit length
    norm = np.sqrt(np.sum(normal_map * normal_map, axis=2, keepdims=True))
    norm = np.maximum(norm, 1e-10)  # Avoid division by zero
    normal_map /= norm
    
    return normal_map
