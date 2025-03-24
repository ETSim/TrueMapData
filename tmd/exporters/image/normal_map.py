""".

Normal map generation module for heightmaps.

This module provides functionality to generate normal maps from height maps.
Normal maps are used in 3D rendering to add surface detail without additional geometry.
"""

import os
import numpy as np
import logging
from typing import Optional, Tuple, Union

# Set up logging
logger = logging.getLogger(__name__)

def create_normal_map(
    height_map: np.ndarray,
    z_scale: float = 1.0,
    normalize: bool = True,
    output_format: str = "rgb"
) -> np.ndarray:
    """.

    Create a normal map from a height map.
    
    Args:
        height_map: 2D numpy array of height values
        z_scale: Scale factor for height values in normal calculation
        normalize: Whether to normalize the height map to [0,1] before processing
        output_format: Format of output normal map ("rgb" or "xyz")
        
    Returns:
        3D numpy array with normal vectors (H,W,3)
    """
    # Normalize height map if requested
    if normalize and (np.max(height_map) > 1.0 or np.min(height_map) < 0.0):
        height_norm = (height_map - np.min(height_map)) / (np.max(height_map) - np.min(height_map))
    else:
        height_norm = height_map.copy()
    
    # Calculate gradients in x and y directions
    grad_x = np.zeros_like(height_norm)
    grad_y = np.zeros_like(height_norm)
    
    # Use central differences for interior points
    grad_x[1:-1, 1:-1] = (height_norm[1:-1, 2:] - height_norm[1:-1, :-2]) / 2.0
    grad_y[1:-1, 1:-1] = (height_norm[2:, 1:-1] - height_norm[:-2, 1:-1]) / 2.0
    
    # Use forward/backward differences for edges
    grad_x[1:-1, 0] = height_norm[1:-1, 1] - height_norm[1:-1, 0]
    grad_x[1:-1, -1] = height_norm[1:-1, -1] - height_norm[1:-1, -2]
    grad_y[0, 1:-1] = height_norm[1, 1:-1] - height_norm[0, 1:-1]
    grad_y[-1, 1:-1] = height_norm[-1, 1:-1] - height_norm[-2, 1:-1]
    
    # Handle corners
    grad_x[0, 0] = grad_x[0, 1]
    grad_y[0, 0] = grad_y[1, 0]
    grad_x[0, -1] = grad_x[0, -2]
    grad_y[0, -1] = grad_y[1, -1]
    grad_x[-1, 0] = grad_x[-1, 1]
    grad_y[-1, 0] = grad_y[-2, 0]
    grad_x[-1, -1] = grad_x[-1, -2]
    grad_y[-1, -1] = grad_y[-2, -1]
    
    # Scale gradients by z_scale
    grad_x = grad_x * z_scale
    grad_y = grad_y * z_scale
    
    # Create normal map
    if output_format == "xyz":
        # XYZ format: X=right, Y=forward, Z=up
        normal_map = np.zeros((height_norm.shape[0], height_norm.shape[1], 3))
        normal_map[:, :, 0] = -grad_x
        normal_map[:, :, 1] = -grad_y
        normal_map[:, :, 2] = 1.0
    else:
        # RGB format: R=X, G=Y, B=Z (OpenGL standard)
        # Where X=right, Y=up, Z=toward viewer
        normal_map = np.zeros((height_norm.shape[0], height_norm.shape[1], 3))
        normal_map[:, :, 0] = -grad_x
        normal_map[:, :, 1] = -grad_y  # Invert Y for OpenGL
        normal_map[:, :, 2] = 1.0
    
    # Normalize to unit length
    norm = np.sqrt(np.sum(normal_map**2, axis=2, keepdims=True))
    normal_map = normal_map / np.maximum(norm, 1e-10)  # Avoid division by zero
    
    return normal_map

def export_normal_map(
    height_map: np.ndarray,
    filename: str,
    z_scale: float = 1.0,
    output_format: str = "rgb",
    normalize: bool = True
) -> bool:
    """.

    Generate and export a normal map from a height map.
    
    Args:
        height_map: 2D numpy array of height values
        filename: Output filename
        z_scale: Scale factor for height values in normal calculation
        output_format: Format of output normal map ("rgb" or "xyz")
        normalize: Whether to normalize the height map before processing
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        
        # Create normal map
        normal_map = create_normal_map(
            height_map=height_map,
            z_scale=z_scale,
            normalize=normalize,
            output_format=output_format
        )
        
        # Convert from [-1,1] to [0,1] range for image export
        normal_map_export = (normal_map + 1.0) * 0.5
        
        # Export using PIL
        try:
            from PIL import Image
            
            # Convert to 8-bit
            normal_map_8bit = (normal_map_export * 255).astype(np.uint8)
            
            # Save the image
            Image.fromarray(normal_map_8bit).save(filename)
            logger.info(f"Normal map exported to {filename}")
            return True
            
        except ImportError:
            logger.error("PIL/Pillow is required for image export")
            return False
            
    except Exception as e:
        logger.error(f"Error exporting normal map: {e}")
        return False

def normal_map_to_rgb(normal_map: np.ndarray) -> np.ndarray:
    """.

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
    """.

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

def convert_heightmap_to_normal_map(height_map: np.ndarray, z_scale: float = 10.0) -> np.ndarray:
    """.

    Generate a normal map from a heightmap.
    
    Args:
        height_map: 2D array of height values
        z_scale: Scale factor for z-axis
        
    Returns:
        3D array of normal vectors (nx, ny, nz) for each pixel
    """
    try:
        import cv2
    except ImportError:
        logger.error("OpenCV is required for normal map generation.")
        return np.zeros((*height_map.shape, 3), dtype=np.float32)
    
    # Use Sobel filter to get gradients
    grad_x = cv2.Sobel(height_map, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(height_map, cv2.CV_32F, 0, 1, ksize=3)
    
    # Scale gradients
    grad_x = grad_x * (1.0 / z_scale)
    grad_y = grad_y * (1.0 / z_scale)
    
    # Create normal vectors
    normal_map = np.zeros((height_map.shape[0], height_map.shape[1], 3), dtype=np.float32)
    
    # X component (pointing right)
    normal_map[..., 0] = -grad_x
    
    # Y component (pointing up)
    normal_map[..., 1] = -grad_y
    
    # Z component (pointing out of the heightmap)
    normal_map[..., 2] = 1.0
    
    # Normalize vectors
    norm = np.sqrt(np.sum(normal_map * normal_map, axis=2))
    norm = np.maximum(norm, 1e-10)  # Avoid division by zero
    
    # Normalize each component
    normal_map[..., 0] /= norm
    normal_map[..., 1] /= norm
    normal_map[..., 2] /= norm
    
    return normal_map
