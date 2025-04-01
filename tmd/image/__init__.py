"""
Image processing package for texture map generation.

This package provides tools for generating various types of maps from height maps,
such as normal maps, roughness maps, ambient occlusion maps, etc.
"""
import logging

# Set up logger
logger = logging.getLogger(__name__)

# Import core functionality
from .core.image_utils import (
    normalize_array,
    handle_nan_values,
    prepare_height_map,
    save_image,
    resize_image
)

# Import export functionality
from .export.registry import MapRegistry, register_generator
from .export.exporter import MapExporter

# Import map generators
from .maps.base_generator import MapGenerator
from .maps.ao import AOMapGenerator
from .maps.normal import NormalMapGenerator
from .maps.roughness import RoughnessMapGenerator
from .maps.bump import BumpMapGenerator
from .maps.metallic import MetallicMapGenerator
from .maps.displacement import DisplacementMapGenerator
from .maps.heightmap import HeightMapGenerator
from .maps.hillshade import HillshadeMapGenerator

# Register generators with registry
MapRegistry.register("ao", AOMapGenerator)
MapRegistry.register("normal", NormalMapGenerator)
MapRegistry.register("bump", BumpMapGenerator)
MapRegistry.register("roughness", RoughnessMapGenerator)
MapRegistry.register("metallic", MetallicMapGenerator)
MapRegistry.register("displacement", DisplacementMapGenerator)
MapRegistry.register("height", HeightMapGenerator)
MapRegistry.register("hillshade", HillshadeMapGenerator)

# Convenience export functions
def export_ao_map(height_map, output_file, **kwargs):
    """Export an ambient occlusion map."""
    return MapExporter.export_map(height_map, output_file, "ao", **kwargs)

def export_normal_map(height_map, output_file, **kwargs):
    """Export a normal map."""
    return MapExporter.export_map(height_map, output_file, "normal", **kwargs)

def export_bump_map(height_map, output_file, **kwargs):
    """Export a bump map."""
    return MapExporter.export_map(height_map, output_file, "bump", **kwargs)

def export_roughness_map(height_map, output_file, **kwargs):
    """Export a roughness map."""
    return MapExporter.export_map(height_map, output_file, "roughness", **kwargs)

def export_metallic_map(height_map, output_file, **kwargs):
    """Export a metallic map."""
    return MapExporter.export_map(height_map, output_file, "metallic", **kwargs)

def export_displacement_map(height_map, output_file, **kwargs):
    """Export a displacement map."""
    return MapExporter.export_map(height_map, output_file, "displacement", **kwargs)

def export_height_map(height_map, output_file, **kwargs):
    """Export a height map."""
    kwargs.setdefault('colormap', 'viridis')  # Set default colormap
    return MapExporter.export_map(height_map, output_file, "height", **kwargs)

def export_hillshade_map(height_map, output_file, **kwargs):
    """Export a hillshade map."""
    return MapExporter.export_map(height_map, output_file, "hillshade", **kwargs)

# Get list of available map types
def get_available_map_types():
    """Get a list of available map types."""
    return [
        "normal", 
        "ao", 
        "bump", 
        "roughness", 
        "metallic", 
        "displacement", 
        "height", 
        "hillshade"
    ]

__all__ = [
    # Core functionality
    'normalize_array',
    'handle_nan_values',
    'prepare_height_map',
    'save_image',
    'resize_image',
    
    # Map generators
    'MapGenerator',
    'AOMapGenerator',
    'NormalMapGenerator',
    'BumpMapGenerator',
    'RoughnessMapGenerator',
    'MetallicMapGenerator',
    'DisplacementMapGenerator',
    'HeightMapGenerator',
    'HillshadeMapGenerator',
    
    # Export functionality
    'MapRegistry',
    'register_generator',
    'MapExporter',
    
    # Convenience export functions
    'export_ao_map',
    'export_normal_map',
    'export_bump_map',
    'export_roughness_map',
    'export_metallic_map',
    'export_displacement_map',
    'export_height_map',
    'export_hillshade_map',
    'get_available_map_types'
]
