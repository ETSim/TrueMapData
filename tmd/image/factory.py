"""
Factory for image exporters.

This module implements a factory pattern for the image
export functionality, handling creation and registry of exporters.
"""

import logging
from typing import Dict, Type, Optional, Any, List

from .base import ExportStrategy, MapExporter

logger = logging.getLogger(__name__)

class ImageExportRegistry:
    """Registry for image export strategies."""
    
    # Class-level storage for registered strategies
    _strategies: Dict[str, Type[ExportStrategy]] = {}
    
    @classmethod
    def register(cls, name: str, strategy_class: Type[ExportStrategy]) -> None:
        """
        Register an export strategy.
        
        Args:
            name: Identifier for the strategy
            strategy_class: Implementation class
        """
        cls._strategies[name.lower()] = strategy_class
        logger.debug(f"Registered image export strategy: {name}")
    
    @classmethod
    def get(cls, name: str) -> Optional[Type[ExportStrategy]]:
        """
        Get a strategy by name.
        
        Args:
            name: Strategy identifier
            
        Returns:
            The strategy class or None if not found
        """
        return cls._strategies.get(name.lower())
    
    @classmethod
    def list_strategies(cls) -> List[str]:
        """
        Get a list of all registered strategies.
        
        Returns:
            List of strategy names
        """
        return list(cls._strategies.keys())
    
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Check if a strategy is registered.
        
        Args:
            name: Strategy identifier
            
        Returns:
            True if registered, False otherwise
        """
        return name.lower() in cls._strategies


class ImageExporterFactory:
    """Factory for creating image exporters."""
    
    @staticmethod
    def create_exporter(map_type: str, **kwargs) -> Optional[MapExporter]:
        """
        Create an exporter for the specified map type.
        
        Args:
            map_type: Type of map to export
            **kwargs: Parameters for the export strategy
            
        Returns:
            Configured MapExporter or None if strategy not found
        """
        strategy_class = ImageExportRegistry.get(map_type)
        if not strategy_class:
            logger.error(f"No export strategy registered for '{map_type}'")
            logger.info(f"Available strategies: {', '.join(ImageExportRegistry.list_strategies())}")
            return None
            
        try:
            strategy = strategy_class(**kwargs)
            return MapExporter(strategy)
        except Exception as e:
            logger.error(f"Error creating exporter for '{map_type}': {e}")
            return None
    
    @staticmethod
    def export_map(
        height_map: Any, 
        output_file: str, 
        map_type: str, 
        **kwargs
    ) -> Optional[str]:
        """
        Quick export method that creates an exporter and exports in one step.
        
        Args:
            height_map: Input height map
            output_file: Path to save the output
            map_type: Type of map to export
            **kwargs: Additional export parameters
            
        Returns:
            Path to the saved file or None if failed
        """
        exporter = ImageExporterFactory.create_exporter(map_type, **kwargs)
        if not exporter:
            available = ImageExportRegistry.list_strategies()
            logger.error(f"Could not create exporter for '{map_type}'. Available: {', '.join(available)}")
            return None
            
        return exporter.export(height_map, output_file, **kwargs)

    @staticmethod
    def get_available_exporters() -> List[str]:
        """
        Get a list of all available export types.
        
        Returns:
            List of registered export strategy names
        """
        return ImageExportRegistry.list_strategies()


# For backward compatibility and simpler naming
ExportRegistry = ImageExportRegistry
MapExporterFactory = ImageExporterFactory