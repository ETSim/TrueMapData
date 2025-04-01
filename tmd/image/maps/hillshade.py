"""
Hillshade map generator.

This module provides a generator for creating hillshade maps from height maps,
which simulate the illumination of a surface given a light source direction.
"""
import logging
import numpy as np

from .base_generator import MapGenerator

logger = logging.getLogger(__name__)

class HillshadeMapGenerator(MapGenerator):
    """Generator for Hillshade maps."""
    
    def __init__(
        self, 
        azimuth: float = 315.0, 
        altitude: float = 45.0, 
        z_factor: float = 1.0,
        **kwargs
    ):
        """
        Initialize the hillshade map generator.
        
        Args:
            azimuth: Light source azimuth in degrees (0-360, clockwise from north)
            altitude: Light source altitude in degrees (0-90)
            z_factor: Vertical exaggeration factor
            **kwargs: Additional default parameters
        """
        super().__init__(
            azimuth=azimuth, 
            altitude=altitude, 
            z_factor=z_factor, 
            **kwargs
        )
    
    def generate(self, height_map: np.ndarray, **kwargs) -> np.ndarray:
        """
        Generate a hillshade map from a height map.
        
        Args:
            height_map: Input height map
            **kwargs: Generation parameters including:
                - azimuth: Light direction in degrees
                - altitude: Light elevation in degrees
                - z_factor: Vertical exaggeration
            
        Returns:
            Hillshade map as numpy array (0-1 range)
        """
        # Get parameters
        params = self._get_params(**kwargs)
        azimuth = params['azimuth']
        altitude = params['altitude']
        z_factor = params['z_factor']
        
        # Prepare height map
        height_map_norm = self._prepare_height_map(height_map)
        
        try:
            # Convert angles to radians
            azimuth_rad = np.radians(360.0 - azimuth)
            altitude_rad = np.radians(altitude)
            
            # Calculate x, y, z components of the light source vector
            x = np.sin(azimuth_rad) * np.cos(altitude_rad)
            y = np.cos(azimuth_rad) * np.cos(altitude_rad)
            z = np.sin(altitude_rad)
            
            # Calculate surface gradients
            from scipy import ndimage
            dx = ndimage.sobel(height_map_norm, axis=1) / 8.0
            dy = ndimage.sobel(height_map_norm, axis=0) / 8.0
            
            # Apply z-factor to gradients
            dx *= z_factor
            dy *= z_factor
            
            # Calculate hillshade
            # Slope
            slope = np.pi/2.0 - np.arctan(np.sqrt(dx*dx + dy*dy))
            
            # Aspect (handle flat areas)
            aspect = np.zeros_like(height_map_norm)
            mask = (dx != 0.0)
            aspect[mask] = np.arctan2(dy[mask], -dx[mask])
            
            # Calculate illumination
            hillshade = np.sin(altitude_rad) * np.sin(slope) + \
                       np.cos(altitude_rad) * np.cos(slope) * \
                       np.cos(azimuth_rad - aspect)
            
            # Scale to [0, 1] range
            hillshade = np.clip(hillshade, 0.0, 1.0)
            
            return hillshade
            
        except ImportError:
            logger.error("SciPy is required for hillshade map generation")
            return np.ones_like(height_map_norm) * 0.5
    
    def _validate_params(self, params):
        """Validate and adjust parameters."""
        # Validate azimuth (0-360)
        if 'azimuth' in params:
            params['azimuth'] = params['azimuth'] % 360.0
            
        # Validate altitude (0-90)
        if 'altitude' in params:
            params['altitude'] = np.clip(params['altitude'], 0.0, 90.0)
            
        # Validate z_factor (positive)
        if 'z_factor' in params and params['z_factor'] <= 0:
            params['z_factor'] = 1.0
            
        return params
