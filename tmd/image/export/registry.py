"""
Registry for map generators.

This module provides functionality for registering and retrieving map generators.
"""
import logging
from typing import Dict, Type, Optional, List

from ..maps.base_generator import MapGenerator

logger = logging.getLogger(__name__)

class MapRegistry:
    """
    Registry for map generators.
    
    This class provides a central registry for all available map generators,
    allowing them to be looked up by name.
    """
    _generators: Dict[str, Type[MapGenerator]] = {}
    
    @classmethod
    def register(cls, name: str, generator_cls: Type[MapGenerator]) -> None:
        """
        Register a map generator.
        
        Args:
            name: Name to register the generator under
            generator_cls: Generator class to register
        """
        cls._generators[name.lower()] = generator_cls
        logger.debug(f"Registered map generator: {name}")
    
    @classmethod
    def get(cls, name: str) -> Optional[Type[MapGenerator]]:
        """
        Get a generator class by name.
        
        Args:
            name: Name of the generator to retrieve
            
        Returns:
            The generator class, or None if not found
        """
        return cls._generators.get(name.lower())
    
    @classmethod
    def list(cls) -> List[str]:
        """
        List all registered generator names.
        
        Returns:
            List of generator names
        """
        return list(cls._generators.keys())

def register_generator(name: str, aliases: List[str] = None):
    """
    Decorator to register a map generator class.
    
    Args:
        name: Primary name for the generator
        aliases: Additional names to register the generator under
        
    Returns:
        Decorator function
    """
    def decorator(cls):
        # Register with primary name
        MapRegistry.register(name, cls)
        
        # Register with aliases if provided
        if aliases:
            for alias in aliases:
                MapRegistry.register(alias, cls)
                
        return cls
    
    return decorator
