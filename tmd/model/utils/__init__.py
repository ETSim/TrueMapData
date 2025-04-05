"""
Utility functions for the TMD model exporters.

This package provides utility functions for mesh generation, heightmap processing,
and validation used by the model exporters.
"""

# Import common utilities from submodules for easier access
from .mesh import (
    create_mesh_from_heightmap,
    calculate_vertex_normals,
    calculate_face_normals,
    generate_uv_coordinates,
    optimize_mesh,
    ensure_watertight_mesh
)

from .validation import validate_heightmap, ensure_directory_exists

from .heightmap import (
    normalize_heightmap,
    calculate_heightmap_normals,
    calculate_terrain_complexity,
    sample_heightmap,
    resample_heightmap,
    generate_heightmap_texture
)

# Define package exports
__all__ = [
    # Mesh utilities
    'create_mesh_from_heightmap',
    'calculate_vertex_normals',
    'calculate_face_normals',
    'generate_uv_coordinates',
    'optimize_mesh',
    'ensure_watertight_mesh',
    'validate_heightmap',
    'ensure_directory_exists',
    
    # Heightmap utilities
    'normalize_heightmap',
    'calculate_heightmap_normals',
    'calculate_terrain_complexity',
    'sample_heightmap',
    'resample_heightmap',
    'generate_heightmap_texture'
]