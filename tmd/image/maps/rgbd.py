"""
RGBD map generator.

This module provides a generator for creating RGBD maps from height maps,
combining color (RGB) and depth (D) information into a single image.
"""
import logging
import numpy as np

from .base_generator import MapGenerator

logger = logging.getLogger(__name__)

class RGBDMapGenerator(MapGenerator):
    """Generator for RGBD maps."""
    
    def __init__(self, color_source: str = "height", depth_scale: float = 1.0, **kwargs):
        """
        Initialize the RGBD map generator.
        
        Args:
            color_source: Source for RGB channels ('height', 'normal', etc.)
            depth_scale: Scaling factor for depth channel
            **kwargs: Additional default parameters
        """
        super().__init__(color_source=color_source, depth_scale=depth_scale, **kwargs)
    
    def generate(self, height_map: np.ndarray, **kwargs) -> np.ndarray:
        """
        Generate an RGBD map from a height map.
        
        Args:
            height_map: Input height map
            **kwargs: Generation parameters including:
                - color_source: Source for RGB channels
                - depth_scale: Scaling factor for depth channel
            
        Returns:
            RGBD map as numpy array (0-1 range)
        """
        # Get parameters
        params = self._get_params(**kwargs)
        color_source = params['color_source']
        depth_scale = params['depth_scale']
        
        # Prepare height map
        height_map_norm = self._prepare_height_map(height_map)
        
        try:
            # Create RGB channels based on color_source
            if color_source == "height":
                # Use height map as grayscale RGB
                rgb = np.stack([height_map_norm] * 3, axis=-1)
            else:
                # Placeholder for other color sources (e.g., normal map)
                rgb = np.zeros_like(height_map_norm)
            
            # Create depth channel
            depth = height_map_norm * depth_scale
            
            # Combine RGB and depth
            rgbd = np.concatenate([rgb, depth[..., np.newaxis]], axis=-1)
            
            return rgbd
            
        except Exception as e:
            logger.error(f"Error generating RGBD map: {e}")
            return np.zeros_like(height_map)
    
    def _validate_params(self, params):
        """Validate and adjust parameters."""
        # Ensure depth_scale is positive
        if params.get('depth_scale', 0) <= 0:
            params['depth_scale'] = 1.0
            
        return params
