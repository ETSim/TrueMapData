"""
Registry for TMD model exporters.

This module provides a registry system for auto-discovery and registration
of model exporters, making it easy to extend the package with new formats.
"""

import os
import importlib
import inspect
import logging
import pkgutil
from typing import Dict, Type, List, Optional, Any, Callable

from .base import ModelExporter

# Set up logging
logger = logging.getLogger(__name__)

# Format registry
_FORMAT_REGISTRY: Dict[str, Type[ModelExporter]] = {}

def register_format(format_name: str, exporter_class: Type[ModelExporter]) -> None:
    """Register a format exporter."""
    format_name = format_name.lower()  # Ensure lowercase for consistent lookup
    _FORMAT_REGISTRY[format_name] = exporter_class
    logger.info(f"Registered format exporter: {format_name} ({exporter_class.__name__})")

def get_exporter(format_name: str) -> Type[ModelExporter]:
    """Get exporter class for a format."""
    format_name = format_name.lower()
    logger.debug(f"Looking up exporter for format: {format_name}")
    logger.debug(f"Available formats: {list(_FORMAT_REGISTRY.keys())}")
    
    if format_name not in _FORMAT_REGISTRY:
        raise ValueError(f"Unknown format: {format_name}. Available formats: {list(_FORMAT_REGISTRY.keys())}")
    return _FORMAT_REGISTRY[format_name]

def get_available_formats() -> List[str]:
    """Get list of available export formats."""
    return list(_FORMAT_REGISTRY.keys())

def register_exporter(cls: Type[ModelExporter]) -> Type[ModelExporter]:
    """Decorator to register a model exporter."""
    register_format(cls.format_name.lower(), cls)
    return cls


class ExporterRegistry:
    """
    Registry for model exporters.
    
    This class manages the discovery, registration, and retrieval of
    model exporters for different file formats.
    """
    
    # Class variables to store registered exporters
    _exporters: Dict[str, Type[ModelExporter]] = {}
    _extension_map: Dict[str, str] = {}
    _initialized = False
    
    @classmethod
    def register(cls, exporter_class: Type[ModelExporter]) -> None:
        """
        Register a model exporter class.
        
        Args:
            exporter_class: Exporter class to register
        """
        # Validate that the exporter class is a subclass of ModelExporter
        if not issubclass(exporter_class, ModelExporter):
            logger.warning(f"Cannot register {exporter_class.__name__}: Not a subclass of ModelExporter")
            return
        
        # Register by format name
        format_name = exporter_class.format_name.lower()
        cls._exporters[format_name] = exporter_class
        logger.debug(f"Registered model exporter for format: {format_name}")
        
        # Register by file extensions
        for ext in exporter_class.file_extensions:
            ext = ext.lower()
            cls._extension_map[ext] = format_name
            logger.debug(f"Registered extension .{ext} for format: {format_name}")
    
    @classmethod
    def get_exporter(cls, format_name: str) -> Optional[Type[ModelExporter]]:
        """
        Get an exporter class for the specified format.
        
        Args:
            format_name: Format name or extension
            
        Returns:
            Exporter class or None if format is not supported
        """
        # Initialize registry if not already done
        cls.ensure_initialized()
        
        # Normalize input
        format_name = format_name.lower()
        
        # Try direct lookup by format name
        exporter_class = cls._exporters.get(format_name)
        
        # Try lookup by extension if not found
        if not exporter_class and format_name in cls._extension_map:
            exporter_class = cls._exporters.get(cls._extension_map[format_name])
        
        # Try dynamic import if not found
        if not exporter_class:
            exporter_class = cls._try_import_format(format_name)

        if not exporter_class:
            logger.warning(f"No exporter found for format: {format_name}")
            
        return exporter_class
    
    @classmethod
    def _try_import_format(cls, format_name: str) -> Optional[Type[ModelExporter]]:
        """
        Try to dynamically import a format module.
        
        Args:
            format_name: Format name or extension
            
        Returns:
            Exporter class if successfully imported, None otherwise
        """
        try:
            # Convert 'stl' to 'formats.stl'
            module_path = f"tmd.exporters.model.formats.{format_name}"
            module = importlib.import_module(module_path)
            
            # Look for *Exporter class in the module
            exporter_class = None
            for name in dir(module):
                if name.endswith('Exporter'):
                    obj = getattr(module, name)
                    if inspect.isclass(obj) and issubclass(obj, ModelExporter):
                        exporter_class = obj
                        break
            
            # Register and return the exporter if found
            if exporter_class:
                cls.register(exporter_class)
                return exporter_class
                
        except (ImportError, AttributeError) as e:
            logger.debug(f"Failed to import format module for {format_name}: {e}")
            
        return None
    
    @classmethod
    def discover_exporters(cls) -> None:
        """
        Discover and register all available exporters.
        
        This method searches for exporter classes in the formats package
        and registers them automatically.
        """
        try:
            # Import the formats package
            from . import formats
            
            # Get the package directory
            package_dir = os.path.dirname(formats.__file__)
            
            # Iterate through package modules
            for _, module_name, is_pkg in pkgutil.iter_modules([package_dir]):
                if not is_pkg and module_name != '__init__':
                    try:
                        # Import the module
                        module = importlib.import_module(f"tmd.exporters.model.formats.{module_name}")
                        
                        # Look for exporter classes
                        for name in dir(module):
                            if name.endswith('Exporter'):
                                obj = getattr(module, name)
                                if inspect.isclass(obj) and issubclass(obj, ModelExporter) and obj != ModelExporter:
                                    cls.register(obj)
                                    
                    except (ImportError, AttributeError) as e:
                        logger.debug(f"Failed to import module {module_name}: {e}")
                        
        except ImportError as e:
            logger.warning(f"Failed to import formats package: {e}")
    
    @classmethod
    def list_registered_formats(cls) -> Dict[str, Type[ModelExporter]]:
        """
        Get a dictionary of all registered exporters.
        
        Returns:
            Dictionary with format names as keys and exporter classes as values
        """
        cls.ensure_initialized()
        return cls._exporters.copy()
    
    @classmethod
    def list_extensions(cls) -> Dict[str, str]:
        """
        Get a dictionary of all registered file extensions.
        
        Returns:
            Dictionary with extensions as keys and format names as values
        """
        cls.ensure_initialized()
        return cls._extension_map.copy()
    
    @classmethod
    def get_format_info(cls) -> List[Dict[str, Any]]:
        """
        Get detailed information about all registered formats.
        
        Returns:
            List of dictionaries with format information
        """
        cls.ensure_initialized()
        
        info = []
        for format_name, exporter_class in cls._exporters.items():
            info.append({
                'name': exporter_class.format_name,
                'extensions': exporter_class.file_extensions,
                'binary_supported': exporter_class.binary_supported
            })
            
        return sorted(info, key=lambda x: x['name'])
    
    @classmethod
    def is_format_available(cls, format_name: str) -> bool:
        """
        Check if a specific format is available.
        
        Args:
            format_name: Format name or extension
            
        Returns:
            True if the format is available, False otherwise
        """
        return cls.get_exporter(format_name) is not None
    
    @classmethod
    def list_available_formats(cls) -> Dict[str, bool]:
        """
        List all potential export formats and their availability status.
        
        Returns:
            Dictionary with format names as keys and availability status as values
        """
        cls.ensure_initialized()
        
        available_formats = {}
        
        # Add already registered formats
        for format_name in cls._exporters:
            available_formats[format_name] = True
        
        # Check for known formats that aren't registered yet
        for format_name in ['stl', 'obj', 'ply', 'gltf', 'glb', 'usd', 'usdz', 'nvbd']:
            if format_name not in available_formats:
                available_formats[format_name] = cls._try_import_format(format_name) is not None
        
        return available_formats
    
    @classmethod
    def ensure_initialized(cls) -> None:
        """Ensure the registry is initialized."""
        if not cls._initialized:
            cls.discover_exporters()
            cls._initialized = True
    
    @classmethod
    def reset(cls) -> None:
        """Reset the registry (primarily for testing)."""
        cls._exporters = {}
        cls._extension_map = {}
        cls._initialized = False