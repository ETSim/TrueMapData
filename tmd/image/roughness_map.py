"""
Roughness map generation module for TMD.

This module provides functions for generating roughness maps from height maps,
which highlight areas of high frequency detail and surface irregularities.
Roughness maps are useful for material texturing and surface analysis.
"""

import os
import logging
import numpy as np
from typing import Optional, Dict, Any, Union, Tuple

from .utils import ensure_directory_exists, normalize_heightmap, handle_nan_values, save_image

# Set up logger
logger = logging.getLogger(__name__)

def generate_roughness_map(
    height_map: np.ndarray,
    kernel_size: int = 3,
    scale: float = 1.0
) -> np.ndarray:
    """
    Generate a roughness map from a height map.
    
    Roughness maps highlight surface irregularities and high-frequency details.
    Two methods are available depending on installed dependencies:
    1. OpenCV (faster): Uses the Laplacian operator to detect changes in surface gradient
    2. SciPy/NumPy (fallback): Uses gradient magnitude to approximate surface roughness
    
    Args:
        height_map: 2D numpy array of height values
        kernel_size: Size of kernel for gradient/Laplacian calculations (odd number)
        scale: Scaling factor for roughness values (higher = more pronounced effect)
        
    Returns:
        2D roughness map as numpy array (uint8, range 0-255)
    """
    # Import optional dependencies
    from tmd.utils.lib_utils import import_optional_dependency
    cv2 = import_optional_dependency('cv2')
    
    # Normalize height map to 0-1 range
    height_array = height_map.astype(np.float32)
    h_min, h_max = np.min(height_array), np.max(height_array)
    
    if h_max > h_min:
        height_array = (height_array - h_min) / (h_max - h_min)
    
    if cv2 is not None:
        # OpenCV implementation (faster)
        # Apply Laplacian operator to detect rapid height changes
        laplacian = cv2.Laplacian(height_array, cv2.CV_32F, ksize=kernel_size)
        roughness = np.abs(laplacian) * scale
    else:
        # Fallback to numpy/scipy gradient
        ndimage = import_optional_dependency('scipy.ndimage')
        if ndimage is None:
            logger.error("Neither OpenCV nor SciPy available for roughness map generation")
            return np.ones_like(height_map, dtype=np.uint8) * 128
        
        # Use gradient magnitude as roughness
        dx, dy = np.gradient(height_array)
        gradient = np.sqrt(dx**2 + dy**2)
        roughness = gradient * scale
    
    # Normalize to 0-255 range
    rough_min, rough_max = roughness.min(), roughness.max()
    if rough_max > rough_min:
        return ((roughness - rough_min) / (rough_max - rough_min) * 255).astype(np.uint8)
    else:
        return np.zeros_like(roughness, dtype=np.uint8)

def export_roughness_map(
    height_map: np.ndarray,
    output_file: str,
    kernel_size: int = 3,
    scale: float = 1.0,
    bit_depth: int = 8,
    **kwargs
) -> Optional[str]:
    """
    Export a roughness map from a height map.
    
    The roughness map highlights areas of high frequency detail and surface
    irregularities. It's useful for material texturing and surface analysis.
    
    Args:
        height_map: 2D numpy array of height values
        output_file: Path to save the output image
        kernel_size: Size of the kernel used for roughness detection (odd number)
        scale: Strength multiplier for the roughness effect
        bit_depth: Bit depth for output image (8 or 16)
        **kwargs: Additional options
            - colormap: Optional colormap to use (default: None)
            - dpi: DPI for output image (default: 300)
            - nan_strategy: Strategy for handling NaNs ('mean', 'zero', 'nearest')
            - normalize: Whether to normalize output (default: True)
        
    Returns:
        Path to the saved image or None if failed
    """
    try:
        # Ensure output directory exists
        if not ensure_directory_exists(os.path.dirname(os.path.abspath(output_file))):
            logger.error(f"Failed to create output directory for {output_file}")
            return None
            
        # Handle NaN values if present
        nan_strategy = kwargs.get('nan_strategy', 'mean')
        if np.any(np.isnan(height_map)):
            height_map = handle_nan_values(height_map, strategy=nan_strategy)
            
        # Create roughness map
        roughness_map = generate_roughness_map(
            height_map=height_map,
            kernel_size=kernel_size,
            scale=scale
        )
        
        # Extract additional parameters
        colormap = kwargs.get('colormap')
        dpi = kwargs.get('dpi', 300)
        normalize = kwargs.get('normalize', True)
        
        # Save the image
        result = save_image(
            roughness_map, 
            output_file, 
            cmap=colormap,
            bit_depth=bit_depth,
            normalize=normalize,
            dpi=dpi
        )
        
        if result:
            logger.info(f"Roughness map saved to {output_file}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error exporting roughness map: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_roughness_map(
    height_map: np.ndarray,
    kernel_size: int = 3,
    scale: float = 1.0,
    **kwargs
) -> np.ndarray:
    """
    Create a roughness map from a height map without saving to file.
    
    Args:
        height_map: Input height map
        kernel_size: Size of the kernel for roughness detection
        scale: Strength multiplier for roughness effect
        **kwargs: Additional options
        
    Returns:
        Roughness map as a normalized 2D array
    """
    # Handle NaN values if present
    nan_strategy = kwargs.get('nan_strategy', 'mean')
    if np.any(np.isnan(height_map)):
        height_map = handle_nan_values(height_map, strategy=nan_strategy)
    
    # Generate roughness map
    return generate_roughness_map(height_map, kernel_size, scale)
