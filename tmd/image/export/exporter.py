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
from ...cli.core.ui import console  # Add UI import

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
            # Extract parameters from kwargs
            compress = kwargs.pop('compress', 75) 
            format = kwargs.pop('format', 'png')
            bit_depth = kwargs.pop('bit_depth', 8)
            colormap = kwargs.pop('colormap', None)
            normalize = kwargs.pop('normalize', True)
            
            # Get original metadata - ensure we don't pass it twice
            metadata = kwargs.pop('metadata', None)
            
            # Get generator class
            generator_cls = MapRegistry.get(map_type)
            if not generator_cls:
                raise MapGeneratorNotFoundError(f"No generator found for map type: {map_type}")
            
            # Create generator instance
            generator = generator_cls()
            
            # Generate the map with all remaining parameters
            map_data = generator.generate(height_map, metadata=metadata, **kwargs)
            
            # Save the map with compression
            saved_path = save_image(
                map_data,
                output_file,
                bit_depth=bit_depth,
                colormap=colormap,
                normalize=normalize,
                compress=compress,
                format=format
            )
            
            if saved_path:
                # Get file size and format
                size_kb = os.path.getsize(saved_path) / 1024
                console.print(f"[green]Saved {map_type} map ({size_kb:.1f} KB) with {compress}% compression[/]")
            
            return saved_path
            
        except Exception as e:
            logger.error(f"Failed to export {map_type} map: {e}")
            console.print(f"[red]Error exporting {map_type} map: {e}[/]")
            import traceback
            traceback.print_exc()
            return None
