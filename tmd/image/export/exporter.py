"""
Map exporter functionality.

This module provides the main exporter class for generating and exporting
maps from height maps.
"""
import logging
import os
from typing import Optional

import numpy as np

from ..core.image_utils import save_image, get_output_filepath
from ..core.exceptions import MapGeneratorNotFoundError
from .registry import MapRegistry

logger = logging.getLogger(__name__)

class MapExporter:
    """
    Exporter for generating and saving maps from height maps.
    
    This class provides a unified interface for generating different types of maps
    from height maps and saving them to files.
    """
    
    @staticmethod
    def export_map(
        height_map: np.ndarray,
        output_file: str,
        map_type: str,
        **kwargs
    ) -> Optional[str]:
        """
        Generate and export a map from a height map.
        
        Args:
            height_map: Input height map
            output_file: Path to save the output file
            map_type: Type of map to generate
            **kwargs: Additional parameters for generation and saving
            
        Returns:
            Path to the saved file, or None if failed
            
        Raises:
            MapGeneratorNotFoundError: If no generator is found for the specified map type
        """
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        try:
            # Get generator class
            generator_cls = MapRegistry.get(map_type)
            if not generator_cls:
                raise MapGeneratorNotFoundError(f"No generator found for map type: {map_type}")
            
            # Create generator instance
            generator = generator_cls(**kwargs)
            
            # Generate the map
            map_data = generator.generate(height_map, **kwargs)
            
            # Extract export-specific parameters
            bit_depth = kwargs.pop('bit_depth', 8)
            colormap = kwargs.pop('colormap', None)
            
            # Save the map
            saved_path = save_image(
                map_data,
                output_file,
                bit_depth=bit_depth,
                colormap=colormap,
                normalize=False,  # Already normalized by generator
                **kwargs
            )
            
            return saved_path
            
        except Exception as e:
            logger.error(f"Failed to export {map_type} map: {e}")
            import traceback
            traceback.print_exc()
            return None
