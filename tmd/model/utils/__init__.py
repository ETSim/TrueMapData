"""Model utility functions."""

from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import logging

# Set up module logger
logger = logging.getLogger(__name__)

from .mesh import (
    create_mesh_from_heightmap,
    calculate_vertex_normals,
    calculate_face_normals,
    generate_uv_coordinates,
    optimize_mesh,
    ensure_watertight_mesh
)

from .heightmap import (
    validate_heightmap,
    normalize_heightmap,
    get_heightmap_stats,
    sample_heightmap,
    resize_heightmap,
    smooth_heightmap
)

def ensure_directory_exists(filepath: str) -> bool:
    """Ensure directory exists for given filepath."""
    import os
    try:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Failed to create directory for {filepath}: {e}")
        return False

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
    'get_heightmap_stats',
    'sample_heightmap',
    'resize_heightmap',
    'smooth_heightmap'
]