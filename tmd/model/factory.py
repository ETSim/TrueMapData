"""
Model Exporter Factory for TMD.

This module provides a factory pattern implementation for creating and managing
various 3D model exporters (STL, OBJ, PLY, GLTF, USD, etc.) for height map data.
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, Type, Optional, Any, List, Callable, Union
import numpy as np

# Setup logging
logger = logging.getLogger(__name__)

class ModelExporter(ABC):
    """
    Abstract base class for all model exporters.
    """
    
    @classmethod
    @abstractmethod
    def export(cls, 
               height_map: np.ndarray, 
               filename: str, 
               x_offset: float = 0.0,
               y_offset: float = 0.0,
               x_length: float = 1.0,
               y_length: float = 1.0,
               z_scale: float = 1.0,
               base_height: float = 0.0,
               **kwargs) -> Optional[str]:
        """
        Export a height map to a 3D model file.
        
        Args:
            height_map: 2D numpy array of height values
            filename: Output filename
            x_offset: X-axis offset for the model
            y_offset: Y-axis offset for the model
            x_length: Physical length in X direction
            y_length: Physical length in Y direction
            z_scale: Scale factor for Z-axis values
            base_height: Height of solid base to add below the model
            **kwargs: Additional format-specific parameters
            
        Returns:
            Path to the created file if successful, None otherwise
        """
        pass
    
    @classmethod
    def get_extension(cls) -> str:
        """
        Get the file extension for this exporter format.
        
        Returns:
            File extension without leading dot (e.g., 'stl', 'obj')
        """
        return ""
    
    @classmethod
    def get_format_name(cls) -> str:
        """
        Get the human-readable format name.
        
        Returns:
            Format name (e.g., 'STL', 'Wavefront OBJ')
        """
        return ""
    
    @classmethod
    def supports_binary(cls) -> bool:
        """
        Check if this format supports binary export.
        
        Returns:
            True if binary export is supported, False otherwise
        """
        return False
    
    @classmethod
    def ensure_extension(cls, filename: str) -> str:
        """
        Ensure filename has the correct extension for this format.
        
        Args:
            filename: Original filename
            
        Returns:
            Filename with correct extension
        """
        ext = cls.get_extension()
        if not ext:
            return filename
            
        if not filename.lower().endswith(f".{ext.lower()}"):
            filename = f"{filename}.{ext}"
            
        return filename


class ModelExporterFactory:
    """
    Factory class for creating and managing model exporters.
    """
    
    _exporters: Dict[str, Type[ModelExporter]] = {}
    
    @classmethod
    def register_exporter(cls, format_name: str, exporter_class: Type[ModelExporter]) -> None:
        """
        Register a model exporter for a specific format.
        
        Args:
            format_name: Format name (e.g., 'stl', 'obj')
            exporter_class: Exporter class to register
        """
        format_name = format_name.lower()
        cls._exporters[format_name] = exporter_class
        
        # Register by extension too if different from format name
        ext = exporter_class.get_extension().lower()
        if ext and ext != format_name:
            cls._exporters[ext] = exporter_class
            
        logger.debug(f"Registered model exporter for format: {format_name}")
    
    @classmethod
    def get_exporter(cls, format_name: str) -> Optional[Type[ModelExporter]]:
        """
        Get an exporter class for the specified format.
        
        Args:
            format_name: Format name or extension
            
        Returns:
            Exporter class or None if format is not supported
        """
        format_name = format_name.lower()
        exporter_class = cls._exporters.get(format_name)
        
        if not exporter_class:
            logger.warning(f"No exporter found for format: {format_name}")
            
        return exporter_class
    
    @classmethod
    def export_heightmap(cls, 
                        height_map: np.ndarray, 
                        filename: str, 
                        format_name: str,
                        x_offset: float = 0.0,
                        y_offset: float = 0.0,
                        x_length: float = 1.0,
                        y_length: float = 1.0,
                        z_scale: float = 1.0,
                        base_height: float = 0.0,
                        binary: bool = False,
                        **kwargs) -> Optional[str]:
        """
        Export a height map to a 3D model file using the appropriate exporter.
        
        Args:
            height_map: 2D numpy array of height values
            filename: Output filename
            format_name: Format name for the model
            x_offset: X-axis offset for the model
            y_offset: Y-axis offset for the model
            x_length: Physical length in X direction
            y_length: Physical length in Y direction
            z_scale: Scale factor for Z-axis values
            base_height: Height of solid base to add below the model
            binary: Whether to use binary format if supported
            **kwargs: Additional format-specific parameters
            
        Returns:
            Path to the created file if successful, None otherwise
        """
        try:
            # Get the appropriate exporter class
            exporter_class = cls.get_exporter(format_name)
            if not exporter_class:
                logger.error(f"Unsupported model format: {format_name}")
                return None
                
            # Ensure the output directory exists
            os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
            
            # Check if binary format is supported
            if binary and not exporter_class.supports_binary():
                logger.warning(f"Binary format not supported for {format_name}, using text format")
                binary = False
                
            # Add binary flag to kwargs if applicable
            if exporter_class.supports_binary():
                kwargs['binary'] = binary
                
            # Export the height map
            result = exporter_class.export(
                height_map=height_map,
                filename=filename,
                x_offset=x_offset,
                y_offset=y_offset,
                x_length=x_length,
                y_length=y_length,
                z_scale=z_scale,
                base_height=base_height,
                **kwargs
            )
            
            return result
            
        except ImportError as e:
            logger.error(f"Failed to import required module: {e}")
            return None
        except Exception as e:
            logger.error(f"Error exporting to {format_name}: {e}")
            return None
    
    @classmethod
    def supported_formats(cls) -> List[str]:
        """
        Get a list of supported export formats.
        
        Returns:
            List of supported format names
        """
        return list(set(cls._exporters.keys()))
    
    @classmethod
    def get_format_info(cls) -> List[Dict[str, Any]]:
        """
        Get detailed information about all supported formats.
        
        Returns:
            List of dictionaries with format information
        """
        # Get unique exporter classes
        unique_exporters = set(cls._exporters.values())
        
        # Generate info for each unique exporter
        info = []
        for exporter in unique_exporters:
            info.append({
                'name': exporter.get_format_name(),
                'extension': exporter.get_extension(),
                'binary_support': exporter.supports_binary()
            })
            
        return sorted(info, key=lambda x: x['name'])


# Register all available exporters
def _register_all_exporters():
    """
    Register all available model exporters.
    
    This function is called when the module is imported.
    """
    try:
        # Import and register STL exporter
        from .stl import STLExporter
        ModelExporterFactory.register_exporter('stl', STLExporter)
    except ImportError:
        logger.debug("STL exporter not available")
        
    try:
        # Import and register OBJ exporter
        from .obj import OBJExporter
        ModelExporterFactory.register_exporter('obj', OBJExporter)
    except ImportError:
        logger.debug("OBJ exporter not available")
        
    try:
        # Import and register PLY exporter
        from .ply import PLYExporter
        ModelExporterFactory.register_exporter('ply', PLYExporter)
    except ImportError:
        logger.debug("PLY exporter not available")
        
    try:
        # Import and register GLTF/GLB exporter
        from .gltf import GLTFExporter
        ModelExporterFactory.register_exporter('gltf', GLTFExporter)
        ModelExporterFactory.register_exporter('glb', GLTFExporter)
    except ImportError:
        logger.debug("GLTF/GLB exporter not available")
        
    try:
        # Import and register USD/USDZ exporter
        from .usd import USDExporter
        ModelExporterFactory.register_exporter('usd', USDExporter)
        ModelExporterFactory.register_exporter('usdz', USDExporter)
    except ImportError:
        logger.debug("USD/USDZ exporter not available")
        
    try:
        # Import and register NVBD exporter for NVidia format
        from .nvbd import NVBDExporter
        ModelExporterFactory.register_exporter('nvbd', NVBDExporter)
    except ImportError:
        logger.debug("NVBD exporter not available")

# Register exporters when the module is imported
_register_all_exporters()


# Provide a simplified function that uses the factory
def export_heightmap_to_model(
    height_map: np.ndarray,
    filename: str,
    format_name: str,
    x_offset: float = 0,
    y_offset: float = 0,
    x_length: float = 1,
    y_length: float = 1,
    z_scale: float = 1,
    base_height: float = 0.0,
    binary: bool = False,
    **kwargs
) -> Optional[str]:
    """
    Export a height map to a 3D model file using the factory.

    Args:
        height_map: 2D numpy array of height values
        filename: Output filename
        format_name: Format name for the model (e.g., 'stl', 'obj', 'ply')
        x_offset: X-axis offset for the model
        y_offset: Y-axis offset for the model
        x_length: Physical length in X direction
        y_length: Physical length in Y direction
        z_scale: Scale factor for Z-axis values
        base_height: Height of solid base to add below the model
        binary: Whether to use binary format if supported
        **kwargs: Additional keyword arguments to pass to the specific exporter

    Returns:
        str: Path to the created file or None if failed
    """
    return ModelExporterFactory.export_heightmap(
        height_map=height_map,
        filename=filename,
        format_name=format_name,
        x_offset=x_offset,
        y_offset=y_offset,
        x_length=x_length,
        y_length=y_length,
        z_scale=z_scale,
        base_height=base_height,
        binary=binary,
        **kwargs
    )