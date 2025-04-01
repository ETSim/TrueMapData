"""
Bump map generator.

This module provides a generator for creating bump maps from height maps,
which represent surface details through grayscale values.
"""
import logging
import numpy as np

from .base_generator import MapGenerator

logger = logging.getLogger(__name__)

class BumpMapGenerator(MapGenerator):
    """Generator for Bump maps."""
    
    def __init__(self, strength: float = 1.0, blur_radius: float = 1.0, **kwargs):
        """
        Initialize the bump map generator.
        
        Args:
            strength: Factor to control the strength of bumps
            blur_radius: Radius for Gaussian blur to smooth the result
            **kwargs: Additional default parameters
        """
        super().__init__(strength=strength, blur_radius=blur_radius, **kwargs)
    
    def generate(self, height_map: np.ndarray, **kwargs) -> np.ndarray:
        """
        Generate a bump map from a height map.
        
        Args:
            height_map: Input height map
            **kwargs: Generation parameters including:
                - strength: Factor for bump intensity
                - blur_radius: Radius for smoothing
            
        Returns:
            Bump map as numpy array (0-1 range)
        """
        # Get parameters
        params = self._get_params(**kwargs)
        strength = params['strength']
        blur_radius = params['blur_radius']
        
        # Prepare height map
        height_map_norm = self._prepare_height_map(height_map)
        
        try:
            # Use Sobel filter to detect edges (approximates slope)
            from scipy import ndimage
            sobel_x = ndimage.sobel(height_map_norm, axis=1)
            sobel_y = ndimage.sobel(height_map_norm, axis=0)
            
            # Apply strength factor
            sobel_x *= strength
            sobel_y *= strength
            
            # Compute gradient magnitude
            gradient = np.sqrt(sobel_x**2 + sobel_y**2)
            
            # Scale to 0-1
            if gradient.max() > gradient.min():
                bump_map = (gradient - gradient.min()) / (gradient.max() - gradient.min())
            else:
                bump_map = np.zeros_like(gradient)
            
            # Apply Gaussian blur if requested
            if blur_radius > 0:
                bump_map = ndimage.gaussian_filter(bump_map, sigma=blur_radius)
                
            return bump_map
            
        except ImportError:
            logger.error("SciPy is required for bump map generation")
            return np.zeros_like(height_map)
    
    def _validate_params(self, params):
        """Validate and adjust parameters."""
        # Ensure strength is positive
        if params.get('strength', 0) <= 0:
            params['strength'] = 1.0
            
        # Ensure blur radius is non-negative
        if params.get('blur_radius', 0) < 0:
            params['blur_radius'] = 0
            
        return params
