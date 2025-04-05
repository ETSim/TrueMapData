"""
Model Exporter Factory for TMD.

This module provides a factory pattern implementation for creating and managing
various 3D model exporters (STL, OBJ, PLY, GLTF, USD, etc.) for height map data.
"""

import logging
from typing import Dict, Type, Optional
import numpy as np

from .base import ModelExporter, ExportConfig
from .registry import get_exporter, get_available_formats
from ..core import TMD 

# Setup logging
logger = logging.getLogger(__name__)


class ModelExporterFactory:
    """
    Factory class for creating and managing model exporters.
    
    This class manages the registration and retrieval of exporters for different
    file formats, handling format selection, and exporting heightmaps to the
    appropriate format.
    """
    
    def export(self, 
               input_file: str,
               output_file: str,
               format_name: str,
               config: ExportConfig) -> bool:
        """Export a TMD file to the specified format."""
        try:
            logger.info("Starting export process")
            logger.info(f"Input file: {input_file}")
            logger.info(f"Output file: {output_file}")
            logger.info(f"Format: {format_name}")
            logger.info(f"Configuration: {vars(config)}")
            
            # Load TMD file using TMD class
            tmd_data = TMD.load(input_file)
            if tmd_data is None:
                logger.error("Failed to load TMD file")
                return False
                
            height_map = tmd_data.height_map
            if height_map is None:
                logger.error("No height map data found in TMD file")
                return False
            
            logger.info(f"Loaded height map: {height_map.shape} - {height_map.dtype}")
            logger.info(f"Height range: {height_map.min():.3f} to {height_map.max():.3f}")
            
            # Get appropriate exporter
            exporter_class = get_exporter(format_name)
            if exporter_class is None:
                logger.error(f"No exporter found for format: {format_name}")
                return False
                
            logger.info(f"Using exporter: {exporter_class.__name__}")
            
            # Export the model
            result = exporter_class.export(
                height_map=height_map,
                filename=str(output_file),
                config=config
            )
            
            if result:
                logger.info(f"Successfully exported to: {output_file}")
                return True
                
            logger.error("Export operation failed")
            return False
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False
    
    @staticmethod
    def get_available_formats() -> list:
        """Get list of available export formats."""
        return get_available_formats()