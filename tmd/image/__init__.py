"""
TMD Image Export Package

This module provides functionality for generating and exporting various map types
from heightmaps:
  - Base classes: ExportStrategy, MapExporter
  - Factory classes: ImageExportRegistry, ImageExporterFactory 
  - Utility functions for image operations and export
"""

# Import base classes and utilities
from .base import (
    ExportStrategy, 
    MapExporter, 
    save_image,
    normalize_heightmap,
    handle_nan_values,
    load_image,
    load_heightmap
)

# Import factory classes
from .factory import (
    ImageExportRegistry, 
    ImageExporterFactory,
    ExportRegistry,  # Alias for backward compatibility
    MapExporterFactory  # Alias for backward compatibility
)

# Make key functions directly available
def get_registered_exporters():
    """
    Get a list of available export strategies.
    
    Returns:
        List[str]: List of registered export strategy names
    """
    return ImageExportRegistry.list_strategies()

def export_map(height_map, output_file, map_type, **kwargs):
    """
    Export a height map as the specified map type.
    
    This is the central function for all map exports and uses the factory pattern
    to determine the appropriate exporter.
    
    Args:
        height_map: Input height map
        output_file: Path to save the output
        map_type: Type of map to export (normal, ao, roughness, etc.)
        **kwargs: Additional parameters specific to the map type
        
    Returns:
        Path to the saved file or None if failed
    """
    return ImageExporterFactory.export_map(
        height_map=height_map,
        output_file=output_file,
        map_type=map_type,
        **kwargs
    )

# Import specific export functions - these modules will register their strategies
from .normal_map import export_normal_map, create_normal_map
from .roughness_map import export_roughness_map, create_roughness_map 
from .metallic_map import export_metallic_map, generate_metallic_map
from .ao_map import export_ambient_occlusion, create_ambient_occlusion_map
from .bump_map import convert_heightmap_to_bump_map
from .displacement_map import export_displacement_map
from .heightmap import export_heightmap
from .hillshade import export_hillshade, generate_hillshade
from .material_set import export_material_set

# Import multi-channel exporters if available
try:
    from .multi_channel import export_multi_channel_image
    from .rgbd import export_rgbd_map
except ImportError:
    pass

# Define __all__ for explicit exports
__all__ = [
    # Base classes
    'ExportStrategy', 
    'MapExporter',
    
    # Factory classes
    'ImageExportRegistry', 
    'ImageExporterFactory',
    'ExportRegistry',
    'MapExporterFactory',
    
    # Utility functions
    'save_image',
    'normalize_heightmap',
    'handle_nan_values',
    'load_image',
    'load_heightmap',
    'get_registered_exporters',
    
    # Main export function
    'export_map',
    
    # Specific export functions
    'export_normal_map',
    'export_roughness_map',
    'export_metallic_map',
    'export_ambient_occlusion',
    'convert_heightmap_to_bump_map',
    'export_displacement_map',
    'export_heightmap',
    'export_hillshade',
    'export_material_set',
    
    # Generation functions
    'create_normal_map',
    'create_roughness_map',
    'generate_metallic_map',
    'create_ambient_occlusion_map',
    'generate_hillshade'
]