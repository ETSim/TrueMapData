""".

Image export modules for heightmaps.

This package contains modules for converting heightmaps to various image formats
suitable for visualization, texture mapping, and material creation.
"""

from .displacement_map import convert_heightmap_to_displacement_map
from .normal_map import convert_heightmap_to_normal_map
from .bump_map import convert_heightmap_to_bump_map
from .ao_map import convert_heightmap_to_ao_map
from .multi_channel import export_multi_channel_image, convert_heightmap_to_multi_channel_map
from .hillshade import generate_hillshade
from .material_set import generate_maps_from_tmd, generate_all_maps
from .utils import (
    generate_roughness_map,
    create_orm_map,
    generate_edge_map,
    save_texture,
    plot_textures,
    normalize_height_map,
    apply_colormap,
    apply_lighting,
    compose_multi_channel_image
)

__all__ = [
    'convert_heightmap_to_displacement_map',
    'convert_heightmap_to_normal_map',
    'convert_heightmap_to_bump_map',
    'convert_heightmap_to_ao_map',
    'convert_heightmap_to_multi_channel_map',
    'export_multi_channel_image',
    'generate_hillshade',
    'generate_maps_from_tmd',
    'generate_all_maps',
    'generate_roughness_map',
    'create_orm_map',
    'generate_edge_map',
    'save_texture',
    'plot_textures',
    'normalize_height_map'
]

def convert_heightmap_to_image(height_map, filename="heightmap.png", format="png", normalize=True):
    """.

    Converts a height map to a standard image file (PNG, JPEG, etc.).
    
    Args:
        height_map: 2D numpy array of height values.
        filename: Output image path.
        format: Output format (png, jpg, tiff, etc.)
        normalize: Whether to normalize values to [0,1] range.
        
    Returns:
        str: Path to the saved image file.
    """
    from .displacement_map import convert_heightmap_to_displacement_map
    
    # Simply use displacement map function which already handles this case
    convert_heightmap_to_displacement_map(height_map, filename, units=None)
    
    return filename
