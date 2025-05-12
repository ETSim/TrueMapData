"""
Hillshade map generator.

This module provides a generator for creating hillshade maps from height maps,
which simulate the illumination of a surface given a light source direction.
"""
import logging
import numpy as np
import math

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
        
        # Get metadata for scaling
        metadata = kwargs.pop('metadata', {}) or {}
        
        # Get physical dimensions for proper scaling
        x_length = metadata.get('x_length', None)
        y_length = metadata.get('y_length', None)
        
        # Prepare height map
        height_map_norm = self._prepare_height_map(height_map, normalize=True)
        
        try:
            # Calculate cell size based on metadata if available
            if x_length is not None and y_length is not None:
                height, width = height_map_norm.shape
                cell_size_x = x_length / width
                cell_size_y = y_length / height
            else:
                cell_size_x = cell_size_y = 1.0
                
            # Convert angles to radians
            azimuth_rad = math.radians(360.0 - azimuth + 90.0)  # Convert to math angle convention
            altitude_rad = math.radians(altitude)
            zenith_rad = math.radians(90.0 - altitude)
            
            # Calculate x and y gradients (properly scaled)
            try:
                from scipy import ndimage
                dx = ndimage.sobel(height_map_norm, axis=1) / (8.0 * cell_size_x)
                dy = ndimage.sobel(height_map_norm, axis=0) / (8.0 * cell_size_y)
            except ImportError:
                # Fallback to numpy gradient if scipy not available
                dx, dy = np.gradient(height_map_norm, cell_size_x, cell_size_y)
            
            # Apply z-factor to gradients
            dx *= z_factor
            dy *= z_factor
            
            # Calculate slope
            slope = np.arctan(np.sqrt(dx*dx + dy*dy))
            
            # Calculate aspect (handle flat areas)
            aspect = np.zeros_like(height_map_norm)
            mask = (dx != 0.0) | (dy != 0.0)  # Where there's any gradient
            aspect[mask] = np.arctan2(dy[mask], -dx[mask])
            
            # Calculate hillshade using illumination formula
            hillshade = (np.cos(zenith_rad) * np.cos(slope) + 
                        np.sin(zenith_rad) * np.sin(slope) * 
                        np.cos(azimuth_rad - aspect))
            
            # Ensure values are in valid range [0,1]
            hillshade = np.clip(hillshade, 0.0, 1.0)
            
            return hillshade
            
        except Exception as e:
            logger.error(f"Error generating hillshade map: {e}")
            # Return a uniform gray image as fallback
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
    
    @classmethod
    def generate_multi_hillshade(cls, height_map: np.ndarray, angles=None, **kwargs):
        """
        Generate multiple hillshade maps from different light angles.
        
        Args:
            height_map: Input height map
            angles: List of (name, azimuth, altitude) tuples
            **kwargs: Additional generator parameters
            
        Returns:
            Dictionary mapping names to hillshade maps
        """
        if angles is None:
            angles = [
                ("nw", 315, 45),  # Northwest (default)
                ("ne", 45, 45),   # Northeast
                ("sw", 225, 45),  # Southwest
                ("se", 135, 45),  # Southeast
            ]
            
        generator = cls(**kwargs)
        results = {}
        
        for angle_data in angles:
            name, azimuth, altitude = angle_data
            results[name] = generator.generate(
                height_map, 
                azimuth=azimuth, 
                altitude=altitude,
                **kwargs
            )
            
        return results
    
    @classmethod
    def blend_hillshades(cls, hillshades, weights=None):
        """
        Blend multiple hillshade maps into a single map.
        
        Args:
            hillshades: Dictionary or list of hillshade maps
            weights: Optional weights for blending
            
        Returns:
            Blended hillshade map
        """
        if isinstance(hillshades, dict):
            hillshade_list = list(hillshades.values())
        else:
            hillshade_list = hillshades
            
        if not hillshade_list:
            return None
            
        # Use equal weights if none specified
        if weights is None:
            weights = [1.0 / len(hillshade_list)] * len(hillshade_list)
            
        # Normalize weights
        weight_sum = sum(weights)
        if weight_sum != 1.0:
            weights = [w / weight_sum for w in weights]
            
        # Create blend
        blend = np.zeros_like(hillshade_list[0], dtype=np.float32)
        for i, hillshade in enumerate(hillshade_list):
            blend += hillshade * weights[i]
            
        return np.clip(blend, 0.0, 1.0)
