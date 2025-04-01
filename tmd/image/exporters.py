"""
Concrete exporter implementations for various map types.

This module implements the strategy pattern for different map types,
including normal maps, roughness maps, metallic maps, etc.
"""

import os
import numpy as np
import logging
from typing import Optional, Dict, Any, Union, Tuple, List, Type

from .base import (
    ExportStrategy, 
    MapExporter, 
    ExportRegistry, 
    ensure_directory_exists,
    handle_nan_values,
    normalize_heightmap,
    save_image
)

# Import specific map generation functions
from .normal_map import create_normal_map
from .roughness_map import generate_roughness_map
from .metallic_map import generate_metallic_map
from .ao_map import create_ambient_occlusion_map

# Set up logger
logger = logging.getLogger(__name__)

class NormalMapStrategy(ExportStrategy):
    """Strategy for exporting normal maps."""
    
    def __init__(self, z_scale: float = 1.0, **kwargs):
        """
        Initialize normal map strategy.
        
        Args:
            z_scale: Z-scale factor for normal map generation
            **kwargs: Additional parameters
        """
        self.z_scale = z_scale
        self.additional_params = kwargs
    
    def generate(self, height_map: np.ndarray, **kwargs) -> np.ndarray:
        """
        Generate a normal map from a height map.
        
        Args:
            height_map: Input height map
            **kwargs: Additional parameters
            
        Returns:
            Normal map as numpy array
        """
        # Handle NaN values if present
        if np.any(np.isnan(height_map)):
            height_map = handle_nan_values(height_map, strategy=kwargs.get("nan_strategy", "mean"))
        
        # Get parameters
        z_scale = kwargs.get("z_scale", self.z_scale)
        output_format = kwargs.get("output_format", "rgb")
        
        # Generate normal map
        normal_map = create_normal_map(
            height_map=height_map,
            z_scale=z_scale,
            normalize=True,
            output_format=output_format
        )
        
        # Convert from [-1,1] to [0,1] range for image export if needed
        if normal_map.min() < 0:
            normal_map = (normal_map + 1.0) * 0.5
            
        return normal_map
    
    def export(self, data: np.ndarray, output_file: str, **kwargs) -> Optional[str]:
        """
        Export a normal map to a file.
        
        Args:
            data: Normal map data to export
            output_file: Path to save the output
            **kwargs: Additional export parameters
            
        Returns:
            Path to the saved file or None if failed
        """
        # Ensure output directory exists
        if not ensure_directory_exists(os.path.dirname(os.path.abspath(output_file))):
            logger.error(f"Failed to create output directory for {output_file}")
            return None
        
        # Extract parameters
        bit_depth = kwargs.get("bit_depth", 8)
        
        # Save the image
        return save_image(data, output_file, bit_depth=bit_depth, normalize=False)
    
    def process_parameters(self, **kwargs) -> Dict[str, Any]:
        """Process and validate parameters."""
        params = self.additional_params.copy()
        params.update(kwargs)
        
        # Validate z_scale
        if params.get("z_scale", 0) <= 0:
            params["z_scale"] = 1.0
            
        return params

class RoughnessMapStrategy(ExportStrategy):
    """Strategy for exporting roughness maps."""
    
    def __init__(self, kernel_size: int = 3, scale: float = 1.0, **kwargs):
        """
        Initialize roughness map strategy.
        
        Args:
            kernel_size: Size of kernel for roughness detection
            scale: Strength multiplier for roughness effect
            **kwargs: Additional parameters
        """
        self.kernel_size = kernel_size
        self.scale = scale
        self.additional_params = kwargs
    
    def generate(self, height_map: np.ndarray, **kwargs) -> np.ndarray:
        """
        Generate a roughness map from a height map.
        
        Args:
            height_map: Input height map
            **kwargs: Additional parameters
            
        Returns:
            Roughness map as numpy array
        """
        # Handle NaN values if present
        if np.any(np.isnan(height_map)):
            height_map = handle_nan_values(height_map, strategy=kwargs.get("nan_strategy", "mean"))
        
        # Get parameters
        kernel_size = kwargs.get("kernel_size", self.kernel_size)
        scale = kwargs.get("scale", self.scale)
        
        # Generate roughness map
        return generate_roughness_map(
            height_map=height_map,
            kernel_size=kernel_size,
            scale=scale
        )
    
    def export(self, data: np.ndarray, output_file: str, **kwargs) -> Optional[str]:
        """
        Export a roughness map to a file.
        
        Args:
            data: Roughness map data to export
            output_file: Path to save the output
            **kwargs: Additional export parameters
            
        Returns:
            Path to the saved file or None if failed
        """
        # Ensure output directory exists
        if not ensure_directory_exists(os.path.dirname(os.path.abspath(output_file))):
            logger.error(f"Failed to create output directory for {output_file}")
            return None
        
        # Extract parameters
        bit_depth = kwargs.get("bit_depth", 8)
        colormap = kwargs.get("colormap")
        
        # Save the image
        return save_image(
            data, 
            output_file, 
            cmap=colormap,
            bit_depth=bit_depth,
            normalize=False
        )
    
    def process_parameters(self, **kwargs) -> Dict[str, Any]:
        """Process and validate parameters."""
        params = self.additional_params.copy()
        params.update(kwargs)
        
        # Validate kernel_size (must be odd)
        if params.get("kernel_size", 0) % 2 == 0:
            params["kernel_size"] = max(3, params["kernel_size"] + 1)
        
        return params

class MetallicMapStrategy(ExportStrategy):
    """Strategy for exporting metallic maps."""
    
    def __init__(
        self, 
        method: str = "constant", 
        value: float = 0.0,
        threshold: float = 0.7,
        **kwargs
    ):
        """
        Initialize metallic map strategy.
        
        Args:
            method: Method for generating metallic map
            value: Metallic value for constant method
            threshold: Height threshold for height_threshold method
            **kwargs: Additional parameters
        """
        self.method = method
        self.value = value
        self.threshold = threshold
        self.additional_params = kwargs
    
    def generate(self, height_map: np.ndarray, **kwargs) -> np.ndarray:
        """
        Generate a metallic map from a height map.
        
        Args:
            height_map: Input height map
            **kwargs: Additional parameters
            
        Returns:
            Metallic map as numpy array
        """
        # Handle NaN values if present
        if np.any(np.isnan(height_map)):
            height_map = handle_nan_values(height_map, strategy=kwargs.get("nan_strategy", "mean"))
        
        # Get parameters
        method = kwargs.get("method", self.method)
        value = kwargs.get("value", self.value)
        threshold = kwargs.get("threshold", self.threshold)
        pattern_scale = kwargs.get("pattern_scale", 1.0)
        pattern_type = kwargs.get("pattern_type", "grid")
        
        # Generate metallic map
        return generate_metallic_map(
            height_map=height_map,
            method=method,
            value=value,
            threshold=threshold,
            pattern_scale=pattern_scale,
            pattern_type=pattern_type
        )
    
    def export(self, data: np.ndarray, output_file: str, **kwargs) -> Optional[str]:
        """
        Export a metallic map to a file.
        
        Args:
            data: Metallic map data to export
            output_file: Path to save the output
            **kwargs: Additional export parameters
            
        Returns:
            Path to the saved file or None if failed
        """
        # Ensure output directory exists
        if not ensure_directory_exists(os.path.dirname(os.path.abspath(output_file))):
            logger.error(f"Failed to create output directory for {output_file}")
            return None
        
        # Extract parameters
        bit_depth = kwargs.get("bit_depth", 8)
        
        # Save the image
        return save_image(data, output_file, bit_depth=bit_depth, normalize=False)
    
    def process_parameters(self, **kwargs) -> Dict[str, Any]:
        """Process and validate parameters."""
        params = self.additional_params.copy()
        params.update(kwargs)
        
        # Validate value range
        if "value" in params:
            params["value"] = np.clip(params["value"], 0.0, 1.0)
        
        return params

class AOMapStrategy(ExportStrategy):
    """Strategy for exporting ambient occlusion maps."""
    
    def __init__(self, samples: int = 16, intensity: float = 1.0, **kwargs):
        """
        Initialize AO map strategy.
        
        Args:
            samples: Number of samples for AO calculation
            intensity: Strength of the ambient occlusion effect
            **kwargs: Additional parameters
        """
        self.samples = samples
        self.intensity = intensity
        self.additional_params = kwargs
    
    def generate(self, height_map: np.ndarray, **kwargs) -> np.ndarray:
        """
        Generate an ambient occlusion map from a height map.
        
        Args:
            height_map: Input height map
            **kwargs: Additional parameters
            
        Returns:
            AO map as numpy array
        """
        # Handle NaN values if present
        if np.any(np.isnan(height_map)):
            height_map = handle_nan_values(height_map, strategy=kwargs.get("nan_strategy", "mean"))
        
        # Get parameters
        samples = kwargs.get("samples", self.samples)
        intensity = kwargs.get("intensity", self.intensity)
        
        # Generate AO map
        return create_ambient_occlusion_map(
            height_map=height_map,
            samples=samples,
            strength=intensity
        )
    
    def export(self, data: np.ndarray, output_file: str, **kwargs) -> Optional[str]:
        """
        Export an ambient occlusion map to a file.
        
        Args:
            data: AO map data to export
            output_file: Path to save the output
            **kwargs: Additional export parameters
            
        Returns:
            Path to the saved file or None if failed
        """
        # Ensure output directory exists
        if not ensure_directory_exists(os.path.dirname(os.path.abspath(output_file))):
            logger.error(f"Failed to create output directory for {output_file}")
            return None
        
        # Extract parameters
        bit_depth = kwargs.get("bit_depth", 8)
        colormap = kwargs.get("colormap")
        
        # Save the image
        return save_image(
            data, 
            output_file, 
            cmap=colormap, 
            bit_depth=bit_depth, 
            normalize=False
        )
    
    def process_parameters(self, **kwargs) -> Dict[str, Any]:
        """Process and validate parameters."""
        params = self.additional_params.copy()
        params.update(kwargs)
        
        # Validate samples count
        if params.get("samples", 0) < 1:
            params["samples"] = 16
        
        return params

class HeightMapStrategy(ExportStrategy):
    """Strategy for exporting height maps."""
    
    def __init__(self, normalize: bool = True, invert: bool = False, **kwargs):
        """
        Initialize height map strategy.
        
        Args:
            normalize: Whether to normalize the height values
            invert: Whether to invert the height values
            **kwargs: Additional parameters
        """
        self.normalize = normalize
        self.invert = invert
        self.additional_params = kwargs
    
    def generate(self, height_map: np.ndarray, **kwargs) -> np.ndarray:
        """
        Process a height map for export.
        
        Args:
            height_map: Input height map
            **kwargs: Additional parameters
            
        Returns:
            Processed height map
        """
        # Handle NaN values if present
        if np.any(np.isnan(height_map)):
            height_map = handle_nan_values(height_map, strategy=kwargs.get("nan_strategy", "mean"))
        
        # Make a copy to avoid modifying the original
        processed_map = height_map.copy()
        
        # Normalize if requested
        normalize = kwargs.get("normalize", self.normalize)
        if normalize:
            processed_map = normalize_heightmap(processed_map)
        
        # Invert if requested
        invert = kwargs.get("invert", self.invert)
        if invert:
            processed_map = 1.0 - processed_map
        
        return processed_map
    
    def export(self, data: np.ndarray, output_file: str, **kwargs) -> Optional[str]:
        """
        Export a height map to a file.
        
        Args:
            data: Height map data to export
            output_file: Path to save the output
            **kwargs: Additional export parameters
            
        Returns:
            Path to the saved file or None if failed
        """
        # Ensure output directory exists
        if not ensure_directory_exists(os.path.dirname(os.path.abspath(output_file))):
            logger.error(f"Failed to create output directory for {output_file}")
            return None
        
        # Extract parameters
        bit_depth = kwargs.get("bit_depth", 16)  # Higher default bit depth for height maps
        colormap = kwargs.get("colormap")
        normalize = kwargs.get("normalize", True)
        
        # Save the image
        return save_image(
            data, 
            output_file, 
            cmap=colormap, 
            bit_depth=bit_depth, 
            normalize=normalize
        )
    
    def process_parameters(self, **kwargs) -> Dict[str, Any]:
        """Process and validate parameters."""
        params = self.additional_params.copy()
        params.update(kwargs)
        return params

# Register the exporters with the registry
ExportRegistry.register("normal", NormalMapStrategy)
ExportRegistry.register("roughness", RoughnessMapStrategy)
ExportRegistry.register("metallic", MetallicMapStrategy)
ExportRegistry.register("ao", AOMapStrategy)
ExportRegistry.register("height", HeightMapStrategy)

# Create convenience functions for backward compatibility

def export_normal_map(
    height_map: np.ndarray,
    output_file: str,
    z_scale: float = 1.0,
    **kwargs
) -> Optional[str]:
    """
    Export a normal map using the strategy pattern.
    
    Args:
        height_map: Input height map
        output_file: Path to save the output
        z_scale: Z-scale factor for normal map generation
        **kwargs: Additional export parameters
        
    Returns:
        Path to the saved file or None if failed
    """
    return MapExporterFactory.export_map(
        height_map=height_map,
        output_file=output_file,
        map_type="normal",
        z_scale=z_scale,
        **kwargs
    )

def export_roughness_map(
    height_map: np.ndarray,
    output_file: str,
    kernel_size: int = 3,
    scale: float = 1.0,
    **kwargs
) -> Optional[str]:
    """
    Export a roughness map using the strategy pattern.
    
    Args:
        height_map: Input height map
        output_file: Path to save the output
        kernel_size: Size of kernel for roughness detection
        scale: Strength multiplier for roughness effect
        **kwargs: Additional export parameters
        
    Returns:
        Path to the saved file or None if failed
    """
    return MapExporterFactory.export_map(
        height_map=height_map,
        output_file=output_file,
        map_type="roughness",
        kernel_size=kernel_size,
        scale=scale,
        **kwargs
    )

def export_metallic_map(
    height_map: np.ndarray,
    output_file: str,
    method: str = "constant",
    value: float = 0.0,
    **kwargs
) -> Optional[str]:
    """
    Export a metallic map using the strategy pattern.
    
    Args:
        height_map: Input height map
        output_file: Path to save the output
        method: Method for generating metallic map
        value: Metallic value for constant method
        **kwargs: Additional export parameters
        
    Returns:
        Path to the saved file or None if failed
    """
    return MapExporterFactory.export_map(
        height_map=height_map,
        output_file=output_file,
        map_type="metallic",
        method=method,
        value=value,
        **kwargs
    )

def export_ao_map(
    height_map: np.ndarray,
    output_file: str,
    samples: int = 16,
    intensity: float = 1.0,
    **kwargs
) -> Optional[str]:
    """
    Export an ambient occlusion map using the strategy pattern.
    
    Args:
        height_map: Input height map
        output_file: Path to save the output
        samples: Number of samples for AO calculation
        intensity: Strength of the ambient occlusion effect
        **kwargs: Additional export parameters
        
    Returns:
        Path to the saved file or None if failed
    """
    return MapExporterFactory.export_map(
        height_map=height_map,
        output_file=output_file,
        map_type="ao",
        samples=samples,
        intensity=intensity,
        **kwargs
    )

def export_height_map(
    height_map: np.ndarray,
    output_file: str,
    normalize: bool = True,
    **kwargs
) -> Optional[str]:
    """
    Export a height map using the strategy pattern.
    
    Args:
        height_map: Input height map
        output_file: Path to save the output
        normalize: Whether to normalize the height values
        **kwargs: Additional export parameters
        
    Returns:
        Path to the saved file or None if failed
    """
    return MapExporterFactory.export_map(
        height_map=height_map,
        output_file=output_file,
        map_type="height",
        normalize=normalize,
        **kwargs
    )

# Import the MapExporterFactory at the end to avoid circular imports
from .base import MapExporterFactory
