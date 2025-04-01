"""
Height map generator.

This module provides a generator for processing and exporting height maps.
This may involve normalization, inversion, and other basic processing.
"""
import logging
from typing import Optional
import numpy as np

from ..core.image_utils import prepare_height_map, save_image
from .base_generator import MapGenerator

# Set up logging
logger = logging.getLogger(__name__)

class HeightMapGenerator(MapGenerator):
    """Generator for Height maps."""
    
    def __init__(self, invert: bool = False, **kwargs):
        """Initialize the height map generator."""
        super().__init__(invert=invert, **kwargs)
    
    def generate(self, height_map: np.ndarray, **kwargs) -> np.ndarray:
        """Process a height map for export."""
        # Get parameters
        params = self._get_params(**kwargs)
        invert = params['invert']
        min_height = params.get('min_height', 0.0)
        max_height = params.get('max_height', 1.0)
        
        # Prepare height map with specified min/max
        height_map_norm = self._prepare_height_map(
            height_map, 
            min_height=min_height, 
            max_height=max_height
        )
        
        # Invert if needed
        if invert:
            return 1.0 - height_map_norm
            
        return height_map_norm