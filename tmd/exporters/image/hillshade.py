"""
Hillshade generation module.

This module provides functions for creating hillshade visualizations from heightmaps,
which simulate the illumination of terrain from different sun angles.
"""

import os
import numpy as np
import logging
from typing import Optional, Union, Tuple
import math

from .utils import ensure_directory_exists, normalize_heightmap, handle_nan_values
from .image_io import save_image

# Set up logging
logger = logging.getLogger(__name__)

def generate_hillshade(
    height_map: np.ndarray,
    output_file: str,
    azimuth: float = 315.0,
    altitude: float = 45.0,
    z_factor: float = 1.0,
    bit_depth: int = 8,
    **kwargs
) -> Optional[str]:
    """
    Generate a hillshade visualization from a height map.
    
    Args:
        height_map: 2D numpy array of height values
        output_file: Path to save the output image
        azimuth: Azimuth angle of the light source in degrees (0-360, 0=North, 90=East)
        altitude: Light altitude angle in degrees (0-90)
        z_factor: Vertical exaggeration factor
        bit_depth: Bit depth for output image (8 or 16)
        **kwargs: Additional options
        
    Returns:
        Path to the saved image or None if failed
    """
    try:
        # Add .png extension if needed
        if not output_file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
            output_file = output_file + '.png'
            
        # Ensure output directory exists
        if not ensure_directory_exists(output_file):
            logger.error(f"Failed to create output directory for {output_file}")
            return None
        
        # Process height map - handle NaN values
        if np.any(np.isnan(height_map)):
            height_map = handle_nan_values(height_map)
        
        # Create hillshade
        hillshade = create_hillshade(height_map, azimuth, altitude, z_factor)
        
        # Save to file
        return save_image(hillshade, output_file, bit_depth=bit_depth)
        
    except Exception as e:
        logger.error(f"Error generating hillshade: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_hillshade(
    height_map: np.ndarray,
    azimuth: float = 315.0,
    altitude: float = 45.0,
    z_factor: float = 1.0
) -> np.ndarray:
    """
    Create a hillshade array from a height map.
    
    Args:
        height_map: 2D numpy array of height values
        azimuth: Azimuth angle of the light source in degrees (0-360, 0=North, 90=East)
        altitude: Altitude angle of the light source in degrees (0-90)
        z_factor: Vertical exaggeration factor
        
    Returns:
        2D numpy array of hillshade values (0-1)
    """
    # Normalize height map to avoid extreme values
    height_map = normalize_heightmap(height_map)
    
    # Convert input angles to radians
    azimuth_rad = math.radians(360.0 - azimuth + 90.0)  # Convert from azimuth to math angle
    altitude_rad = math.radians(altitude)
    
    # Calculate cell size (assume square cells for simplicity)
    cell_size = 1.0
    
    # Calculate slope and aspect
    # Compute x and y derivatives (dz/dx and dz/dy)
    dx, dy = np.gradient(height_map, cell_size)
    
    # Apply z-factor to adjust vertical exaggeration
    dx *= z_factor
    dy *= z_factor
    
    # Calculate slope in radians
    slope = np.arctan(np.sqrt(dx*dx + dy*dy))
    
    # Calculate aspect in radians
    # (arctan2 returns values in range (-pi, pi))
    aspect = np.arctan2(dy, -dx)
    
    # Calculate hillshade
    # Formula: cos(zenith) * cos(slope) + sin(zenith) * sin(slope) * cos(azimuth - aspect)
    zenith_rad = math.radians(90.0 - altitude)
    hillshade = (np.cos(zenith_rad) * np.cos(slope) + 
                np.sin(zenith_rad) * np.sin(slope) * np.cos(azimuth_rad - aspect))
    
    # Adjust range to 0-1 (no negative values)
    hillshade = np.clip(hillshade, 0, 1)
    
    return hillshade

# Function needed by the test
def export_hillshade(height_map: np.ndarray, output_file: str, **kwargs) -> Optional[str]:
    """
    Export a hillshade visualization from a height map.
    
    This is an alias for generate_hillshade to maintain compatibility with tests.
    
    Args:
        height_map: 2D numpy array of height values
        output_file: Path to save the output image
        **kwargs: Additional options for generate_hillshade
        
    Returns:
        Path to the saved image or None if failed
    """
    return generate_hillshade(height_map, output_file, **kwargs)

def generate_multi_hillshade(
    height_map: np.ndarray,
    output_dir: str,
    base_name: str = "hillshade",
    z_factor: float = 1.0,
    angles: Optional[list] = None,
    **kwargs
) -> dict:
    """
    Generate multiple hillshade images from different angles.
    
    Args:
        height_map: 2D numpy array of height values
        output_dir: Directory to save output images
        base_name: Base filename for outputs
        z_factor: Vertical exaggeration factor
        angles: List of (azimuth, altitude) tuples to use
        
    Returns:
        Dictionary mapping angle names to output files
    """
    if angles is None:
        # Default angles - covering multiple directions
        angles = [
            ("nw", 315, 45),  # Northwest (default)
            ("ne", 45, 45),   # Northeast
            ("sw", 225, 45),  # Southwest
            ("se", 135, 45),  # Southeast
            ("top", 0, 90)    # Top-down
        ]
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    for angle_info in angles:
        if len(angle_info) == 3:
            name, azimuth, altitude = angle_info
        else:
            # If just angles provided, generate name
            azimuth, altitude = angle_info
            name = f"az{azimuth}_alt{altitude}"
        
        # Generate output filename
        output_file = os.path.join(output_dir, f"{base_name}_{name}.png")
        
        # Generate hillshade
        result = generate_hillshade(
            height_map=height_map,
            output_file=output_file,
            azimuth=azimuth,
            altitude=altitude,
            z_factor=z_factor,
            **kwargs
        )
        
        if result:
            results[name] = result
    
    return results

def blend_hillshades(
    hillshades: list,
    output_file: str,
    weights: Optional[list] = None,
    **kwargs
) -> Optional[str]:
    """
    Blend multiple hillshade images into one visualization.
    
    Args:
        hillshades: List of hillshade arrays
        output_file: Path to save blended image
        weights: Optional list of blend weights
        
    Returns:
        Path to saved file or None if failed
    """
    if not hillshades:
        logger.error("No hillshades provided for blending")
        return None
    
    # Default to equal weights if not specified
    if weights is None:
        weights = [1.0 / len(hillshades)] * len(hillshades)
    
    # Ensure weights sum to 1
    weight_sum = sum(weights)
    if weight_sum != 1.0:
        weights = [w / weight_sum for w in weights]
    
    # Create weighted blend
    blend = np.zeros_like(hillshades[0], dtype=np.float32)
    for i, hillshade in enumerate(hillshades):
        blend += hillshade * weights[i]
    
    # Ensure values are in 0-1 range
    blend = np.clip(blend, 0, 1)
    
    # Save blended image
    return save_image(blend, output_file, **kwargs)

def convert_heightmap_to_hillshade(
    height_map: np.ndarray,
    output_file: str,
    **kwargs
) -> Optional[str]:
    """
    Convert a heightmap to a hillshade visualization.
    
    This is a convenience wrapper around generate_hillshade.
    
    Args:
        height_map: 2D numpy array of height values
        output_file: Path to save the output image
        **kwargs: Additional options for generate_hillshade
        
    Returns:
        Path to the saved image or None if failed
    """
    return generate_hillshade(height_map, output_file, **kwargs)
