"""
Displacement map generator.

This module provides a generator for creating displacement maps from height maps,
which are used for actual geometric displacement in 3D rendering.
"""
import logging
import numpy as np

from .base_generator import MapGenerator

logger = logging.getLogger(__name__)

class DisplacementMapGenerator(MapGenerator):
    """Generator for Displacement maps."""
    
    def __init__(
        self, 
        intensity: float = 1.0,
        invert: bool = False,
        **kwargs
    ):
        """
        Initialize the displacement map generator.
        
        Args:
            intensity: Intensity factor for displacement values
            invert: Whether to invert the height values
            **kwargs: Additional default parameters
        """
        super().__init__(intensity=intensity, invert=invert, **kwargs)
    
    def generate(self, height_map: np.ndarray, **kwargs) -> np.ndarray:
        """
        Generate a displacement map from a height map.
        
        Args:
            height_map: Input height map
            **kwargs: Generation parameters including:
                - intensity: Intensity factor for displacement
                - invert: Whether to invert heights
            
        Returns:
            Displacement map as numpy array (0-1 range)
        """
        # Get parameters
        params = self._get_params(**kwargs)
        intensity = params['intensity']
        invert = params['invert']
        
        # Prepare height map
        height_map_norm = self._prepare_height_map(height_map)
        
        # Apply intensity
        displacement = height_map_norm * intensity
        
        # Invert if needed
        if invert:
            displacement = 1.0 - displacement
            
        return displacement
    
    def _validate_params(self, params):
        """Validate and adjust parameters."""
        # Ensure intensity is non-zero
        if params.get('intensity', 0) == 0:
            params['intensity'] = 1.0
            
        return params
