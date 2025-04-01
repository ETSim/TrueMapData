"""
Metallic map generation module for TMD.

This module provides functions for generating metallic maps from height maps
for use in PBR (Physically Based Rendering) material workflows.
"""

import os
import numpy as np
import logging
from typing import Optional, Dict, Any, Union, Tuple

from .utils import ensure_directory_exists, normalize_heightmap, handle_nan_values
from .image_io import save_image

# Set up logger
logger = logging.getLogger(__name__)

def generate_metallic_map(
    height_map: np.ndarray,
    method: str = "constant",
    value: float = 0.0,
    threshold: float = 0.7,
    pattern_scale: float = 1.0,
    pattern_type: str = "grid",
    **kwargs
) -> np.ndarray:
    """
    Generate a metallic map from a height map.
    
    Metallic maps define which areas of a surface are metallic (1.0) or 
    non-metallic (0.0). Common PBR materials use either 0 or 1 values, 
    but fractional values can be used for partial metallic properties.
    
    Args:
        height_map: 2D numpy array of height values
        method: Method to use ('constant', 'height_threshold', 'pattern')
        value: Metallic value to use for constant method (0.0-1.0)
        threshold: Height threshold for 'height_threshold' method
        pattern_scale: Scale factor for patterns
        pattern_type: Pattern type ('grid', 'checker', 'noise')
        **kwargs: Additional parameters
        
    Returns:
        2D numpy array with metallic values in range 0-1
    """
    # Normalize height map for calculations
    height_norm = normalize_heightmap(height_map)
    
    # Create output map of same size
    metallic_map = np.zeros_like(height_norm, dtype=np.float32)
    
    if method == "constant":
        # Constant value throughout (most common)
        return np.ones_like(height_norm) * np.clip(value, 0.0, 1.0)
        
    elif method == "height_threshold":
        # Areas above threshold are metallic
        return np.where(height_norm > threshold, 1.0, 0.0).astype(np.float32)
        
    elif method == "gradient":
        # Gradient based on height (higher = more metallic)
        return height_norm.astype(np.float32)
        
    elif method == "pattern":
        # Create pattern-based metallic map
        h, w = height_norm.shape
        
        # Create base pattern
        if pattern_type == "grid":
            # Grid pattern
            grid_size = int(min(h, w) / (10.0 / pattern_scale))
            grid_size = max(2, grid_size)  # Ensure at least 2 pixels
            
            x = np.arange(w) % grid_size
            y = np.arange(h) % grid_size
            
            x_grid, y_grid = np.meshgrid(x, y)
            border_width = max(1, grid_size // 4)
            
            # Create grid pattern with borders
            mask = (x_grid < border_width) | (x_grid >= grid_size - border_width) | \
                   (y_grid < border_width) | (y_grid >= grid_size - border_width)
            
            return mask.astype(np.float32)
            
        elif pattern_type == "checker":
            # Checkerboard pattern
            checker_size = int(min(h, w) / (10.0 / pattern_scale))
            checker_size = max(2, checker_size)  # Ensure at least 2 pixels
            
            x = (np.arange(w) // checker_size) % 2
            y = (np.arange(h) // checker_size) % 2
            
            x_grid, y_grid = np.meshgrid(x, y)
            checker = (x_grid + y_grid) % 2
            
            return checker.astype(np.float32)
            
        elif pattern_type == "noise":
            # Perlin-like noise pattern (simplified)
            try:
                from scipy.ndimage import gaussian_filter
                
                # Generate random noise
                rng = np.random.RandomState(kwargs.get("seed", 42))
                noise = rng.rand(h, w)
                
                # Smooth the noise
                smoothing = 5.0 / pattern_scale
                noise = gaussian_filter(noise, sigma=smoothing)
                
                # Normalize to 0-1
                noise = (noise - noise.min()) / (noise.max() - noise.min())
                
                # Apply threshold to get binary metallic/non-metallic regions
                noise_threshold = kwargs.get("noise_threshold", 0.5)
                return (noise > noise_threshold).astype(np.float32)
                
            except ImportError:
                logger.warning("SciPy not available for noise pattern. Using checkerboard instead.")
                # Fall back to checkerboard
                return generate_metallic_map(height_map, method="pattern", pattern_type="checker", 
                                            pattern_scale=pattern_scale)
    else:
        # Default: non-metallic
        return np.zeros_like(height_norm, dtype=np.float32)

def export_metallic_map(
    height_map: np.ndarray,
    output_file: str,
    method: str = "constant",
    value: float = 0.0,
    threshold: float = 0.7,
    bit_depth: int = 8,
    **kwargs
) -> Optional[str]:
    """
    Export a metallic map from a height map.
    
    Args:
        height_map: 2D numpy array of height values
        output_file: Path to save the output image
        method: Method to generate the metallic map
        value: Metallic value for constant method
        threshold: Height threshold for height_threshold method
        bit_depth: Bit depth for output image (8 or 16)
        **kwargs: Additional parameters
        
    Returns:
        Path to the saved image or None if failed
    """
    try:
        # Ensure output directory exists
        if not ensure_directory_exists(os.path.dirname(os.path.abspath(output_file))):
            logger.error(f"Failed to create output directory for {output_file}")
            return None
            
        # Handle NaN values if present
        if np.any(np.isnan(height_map)):
            height_map = handle_nan_values(height_map, strategy=kwargs.get("nan_strategy", "mean"))
            
        # Generate metallic map
        metallic_map = generate_metallic_map(
            height_map=height_map,
            method=method,
            value=value,
            threshold=threshold,
            pattern_scale=kwargs.get("pattern_scale", 1.0),
            pattern_type=kwargs.get("pattern_type", "grid"),
            **kwargs
        )
        
        # Save the image (grayscale)
        result = save_image(
            metallic_map, 
            output_file, 
            bit_depth=bit_depth,
            normalize=False  # Already normalized to 0-1
        )
        
        if result:
            logger.info(f"Metallic map saved to {output_file}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error exporting metallic map: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_metallic_map(
    height_map: np.ndarray,
    method: str = "constant",
    value: float = 0.0,
    **kwargs
) -> np.ndarray:
    """
    Create a metallic map from a height map without saving to file.
    
    Args:
        height_map: Input height map
        method: Method to generate metallic map
        value: Metallic value for constant method
        **kwargs: Additional options
        
    Returns:
        Metallic map as a normalized 2D array (0-1)
    """
    # Handle NaN values if present
    if np.any(np.isnan(height_map)):
        height_map = handle_nan_values(height_map, strategy=kwargs.get("nan_strategy", "mean"))
    
    # Generate metallic map
    return generate_metallic_map(
        height_map=height_map,
        method=method,
        value=value,
        **kwargs
    )

def apply_metallic_to_material(
    base_color: np.ndarray,
    metallic_map: np.ndarray,
    specular_tint: np.ndarray = None,
    **kwargs
) -> np.ndarray:
    """
    Apply metallic effect to a base color map.
    
    In PBR rendering, metallic areas reflect environment color rather than base color.
    This function simulates that effect for preview purposes.
    
    Args:
        base_color: RGB base color map (H,W,3)
        metallic_map: Metallic map values (H,W)
        specular_tint: Optional specular tint color (default silvery)
        **kwargs: Additional options
        
    Returns:
        Modified color map with metallic effect
    """
    # Default to silver specular color if not provided
    if specular_tint is None:
        specular_color = np.array([0.8, 0.8, 0.8])
    elif isinstance(specular_tint, (list, tuple)):
        specular_color = np.array(specular_tint)
    else:
        specular_color = specular_tint
        
    # Handle different input shapes
    if metallic_map.ndim == 3 and metallic_map.shape[2] > 1:
        # Use first channel if multichannel
        metallic = metallic_map[:, :, 0]
    else:
        metallic = metallic_map
    
    # Expand metallic to 3 channels
    if metallic.ndim == 2:
        metallic_3ch = np.expand_dims(metallic, axis=2)
        metallic_3ch = np.repeat(metallic_3ch, 3, axis=2)
    else:
        metallic_3ch = metallic
    
    # Calculate metallic effect
    # In real PBR: baseColor * (1-metallic) + (reflection * metallic)
    # Here we just use specular color as reflection approximation
    result = base_color * (1.0 - metallic_3ch) + specular_color * metallic_3ch
    
    # Enhance contrast for metallic areas
    metallic_strength = kwargs.get("metallic_strength", 1.0)
    if metallic_strength != 1.0:
        # Adjust the blend amount
        result = base_color * (1.0 - metallic_3ch * metallic_strength) + \
                specular_color * metallic_3ch * metallic_strength
    
    return np.clip(result, 0, 1)

def convert_heightmap_to_metallic_map(
    height_map: np.ndarray,
    output_file: str,
    **kwargs
) -> Optional[str]:
    """
    Convert a heightmap to a metallic map.
    
    This is an alias for export_metallic_map to maintain a consistent API
    with other converter functions.
    
    Args:
        height_map: 2D numpy array of height values
        output_file: Path to save the output image
        **kwargs: Additional options for export_metallic_map
        
    Returns:
        Path to the saved image or None if failed
    """
    return export_metallic_map(height_map, output_file, **kwargs)
