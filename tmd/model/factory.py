"""
Model Exporter Factory for TMD.

This module provides a factory pattern implementation for creating and managing
various 3D model exporters (STL, OBJ, PLY, GLTF, USD, etc.) for height map data.
"""

import logging
import os
from typing import Any, Optional

from .base import ExportConfig
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
            logger.debug(f"Export parameters: format={format_name}, method={config.triangulation_method}")
            
            # Normalize format name
            format_name = format_name.lower().strip()
            
            # Load TMD file using TMD class
            tmd_data = TMD.load(input_file)
            if tmd_data is None:
                logger.error("Failed to load TMD file")
                return False
                
            height_map = tmd_data.height_map
            if height_map is None:
                logger.error("No height map data found in TMD file")
                return False
            
            # Get appropriate exporter
            exporter_class = get_exporter(format_name)
            if exporter_class is None:
                available_formats = get_available_formats()
                logger.error(f"No exporter found for format: {format_name}. Available formats: {', '.join(available_formats)}")
                return False
            
            # Validate triangulation method
            if config.triangulation_method not in ['adaptive', 'quadtree']:
                logger.warning(f"Invalid triangulation method: {config.triangulation_method}. Using adaptive.")
                config.triangulation_method = 'adaptive'
                
            logger.info(f"Using {config.triangulation_method} triangulation method")
            
            # Export the model
            result = exporter_class.export(
                height_map=height_map,
                filename=output_file,
                config=config
            )
            
            if result:
                # Verify the file was actually created
                if os.path.exists(output_file):
                    logger.info(f"Successfully exported to: {output_file}")
                    return True
                else:
                    logger.error(f"Export claimed success but file not found: {output_file}")
                    return False
            
            logger.error("Export operation failed")
            return False
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False

    @staticmethod
    def export_heightmap(
        height_map: Any,
        filename: str,
        format_name: str,
        config: ExportConfig,
    ) -> Optional[str]:
        """Export a numpy height map to a file using the registered format exporter."""
        format_name = format_name.lower().strip()
        try:
            exporter_class = get_exporter(format_name)
        except ValueError as e:
            logger.error(str(e))
            return None

        if config.triangulation_method not in ["adaptive", "quadtree"]:
            logger.warning(
                "Invalid triangulation method: %s. Using adaptive.",
                config.triangulation_method,
            )
            config.triangulation_method = "adaptive"

        logger.info("Using %s triangulation method", config.triangulation_method)
        return exporter_class.export(
            height_map=height_map,
            filename=filename,
            config=config,
        )

    @staticmethod
    def get_available_formats() -> list:
        """Get list of available export formats."""
        return get_available_formats()