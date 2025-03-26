"""
Image exporters for TMD height maps.

This module provides functions for converting height maps to various image formats
such as normal maps, displacement maps, etc.
"""

# Import from submodules
from .normal_map import (
    create_normal_map,
    export_normal_map,
    convert_heightmap_to_normal_map,
    normal_map_to_rgb,
    rgb_to_normal_map
)

from .bump_map import convert_heightmap_to_bump_map

from .ao_map import (
    convert_heightmap_to_ao_map,
    create_ambient_occlusion_map,
    export_ambient_occlusion
)

from .displacement_map import (
    export_displacement_map,
    process_displacement_map,
    convert_heightmap_to_displacement_map
)

from .heightmap import (
    convert_heightmap_to_heightmap,
    export_heightmap
)

from .hillshade import (
    generate_hillshade,
    create_hillshade,
    generate_multi_hillshade,
    blend_hillshades,
    convert_heightmap_to_hillshade
)

from .material_set import generate_material_set

# Define the list of exportable functions
__all__ = [
    # Normal maps
    'create_normal_map',
    'export_normal_map',
    'convert_heightmap_to_normal_map',
    'normal_map_to_rgb',
    'rgb_to_normal_map',
    
    # Bump maps
    'convert_heightmap_to_bump_map',
    
    # AO maps
    'convert_heightmap_to_ao_map',
    'create_ambient_occlusion_map',
    'export_ambient_occlusion',
    
    # Displacement maps
    'export_displacement_map',
    'process_displacement_map',
    'convert_heightmap_to_displacement_map',
    
    # Heightmaps
    'convert_heightmap_to_heightmap',
    'export_heightmap',
    
    # Hillshade
    'generate_hillshade',
    'create_hillshade',
    'generate_multi_hillshade',
    'blend_hillshades',
    'convert_heightmap_to_hillshade',
    
    # Material sets
    'generate_material_set'
]
