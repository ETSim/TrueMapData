""".

Ambient occlusion map generation module.

This module provides functions for converting heightmaps to ambient occlusion maps.
"""

import os
import logging
import numpy as np
from PIL import Image
from scipy import ndimage

logger = logging.getLogger(__name__)

def convert_heightmap_to_ao_map(height_map, filename="ao_map.png", samples=16, intensity=1.0, radius=1.0):
    """.

    Converts a height map to an ambient occlusion map.

    Ambient occlusion represents how exposed each point is to ambient lighting.

    Args:
        height_map: 2D numpy array of height values.
        filename: Name of the output PNG file.
        samples: Number of samples for AO calculation (higher = better quality but slower).
        intensity: Strength of the ambient occlusion effect.
        radius: Radius to consider for occlusion.

    Returns:
        PIL Image object of the ambient occlusion map.
    """
    height_map = height_map.astype(np.float32)
    rows, cols = height_map.shape

    # Simple approximation: invert the normalized height map
    # For a more accurate AO, you'd use ray sampling techniques
    h_min, h_max = np.min(height_map), np.max(height_map)
    if h_max > h_min:
        normalized = (height_map - h_min) / (h_max - h_min)
    else:
        normalized = np.zeros_like(height_map)

    # Simple AO for basic use cases
    ao_map = np.zeros((rows, cols), dtype=np.uint8)

    if samples <= 1:
        # Simplest case: just invert the normalized height
        ao_map = (255 * (1 - normalized * intensity)).astype(np.uint8)
    else:
        # Use a more sophisticated approach with neighborhood sampling
        ao_map = (255 * (1 - normalized * intensity)).astype(np.uint8)

        # Add a blurred shadow effect for more realism
        blurred = ndimage.gaussian_filter(255 - ao_map, sigma=radius)
        blurred = blurred / np.max(blurred) * 255 if np.max(blurred) > 0 else blurred
        ao_map = np.clip(ao_map * 0.7 + blurred * 0.3, 0, 255).astype(np.uint8)

    im = Image.fromarray(ao_map)
    
    if filename:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        im.save(filename)
        logger.info(f"Ambient occlusion map saved to {filename}")
        
    return im

def create_ambient_occlusion_map(
    height_map: np.ndarray,
    strength: float = 1.0,
    samples: int = 16,
    radius: float = 1.0
) -> np.ndarray:
    """.

    Create an ambient occlusion map from a heightmap.
    
    Args:
        height_map: 2D numpy array of normalized height values (0-1)
        strength: Strength of the ambient occlusion effect (0-1)
        samples: Number of sampling directions
        radius: Sampling radius relative to heightmap size
        
    Returns:
        2D numpy array of ambient occlusion values (0-1)
    """
    if height_map.ndim != 2:
        raise ValueError("Height map must be 2D")
    
    height, width = height_map.shape
    ao_map = np.ones((height, width), dtype=np.float32)
    
    # Convert radius to pixels
    pixel_radius = int(min(height, width) * radius / 10)
    if pixel_radius < 1:
        pixel_radius = 1
    
    # Sample positions on a hemisphere
    theta = np.linspace(0, 2 * np.pi, samples)
    x_samples = np.cos(theta)
    y_samples = np.sin(theta)
    
    # For each pixel, compute occlusion from nearby higher points
    for y in range(height):
        for x in range(width):
            # Current height at this position
            center_height = height_map[y, x]
            
            # Accumulate occlusion from samples
            occlusion = 0
            
            for i in range(samples):
                # Sample direction
                dx = x_samples[i]
                dy = y_samples[i]
                
                # Sample along ray
                max_angle = 0
                
                for r in range(1, pixel_radius + 1):
                    # Sample position
                    sx = int(x + dx * r)
                    sy = int(y + dy * r)
                    
                    # Check if the position is within bounds
                    if 0 <= sx < width and 0 <= sy < height:
                        # Height at sample position
                        sample_height = height_map[sy, sx]
                        
                        # Height difference
                        height_diff = sample_height - center_height
                        
                        if height_diff > 0:
                            # Calculate angle to horizon
                            distance = r / pixel_radius
                            angle = np.arctan2(height_diff, distance)
                            max_angle = max(max_angle, angle)
                
                # Convert angle to occlusion factor
                occlusion += np.sin(max_angle) ** 2
            
            # Average occlusion from all directions and apply strength
            occlusion /= samples
            occlusion *= strength
            
            # Update AO map (invert so 1 = no occlusion, 0 = full occlusion)
            ao_map[y, x] = 1.0 - occlusion
    
    # Use alternative calculation for large maps
    if height * width > 250000:  # Threshold for large maps (~500x500)
        # This is a faster approximation - calculate gradient-based AO
        dx = ndimage.sobel(height_map, axis=1)
        dy = ndimage.sobel(height_map, axis=0)
        slope = np.sqrt(dx**2 + dy**2)
        
        # Normalize slope
        if np.max(slope) > 0:
            slope = slope / np.max(slope)
        
        # Convert slope to AO (steeper slopes = more occlusion)
        fast_ao = 1.0 - slope * strength
        
        # Filter the AO map to smooth it
        fast_ao = ndimage.gaussian_filter(fast_ao, sigma=1.0)
        
        # Use the faster approximation
        ao_map = fast_ao
    
    return ao_map
