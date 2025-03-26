"""
Model exporters for heightmaps.

This module provides various exporters for converting heightmaps to 3D models.
"""

# Import all exportable functions from modules
from .base import create_mesh_from_heightmap
from .gltf import (
    convert_heightmap_to_gltf,
    convert_heightmap_to_glb,
    export_gltf, 
    export_glb,
    heightmap_to_mesh
)
from .stl import (
    convert_heightmap_to_stl,
    export_stl,
    _ensure_watertight_mesh
)
from .obj import (
    convert_heightmap_to_obj,
    export_obj
)
from .ply import (
    convert_heightmap_to_ply
)
from .nvbd import (
    convert_heightmap_to_nvbd
)
from .usd import (
    convert_heightmap_to_usd,
    convert_heightmap_to_usdz
)
from .adaptive_mesh import (
    convert_heightmap_to_adaptive_mesh
)

# Define the list of functions that should be exposed from this package
__all__ = [
    # Base mesh creation
    'create_mesh_from_heightmap',
    'heightmap_to_mesh',
    
    # GLTF/GLB exporters
    'convert_heightmap_to_gltf',
    'convert_heightmap_to_glb',
    'export_gltf',
    'export_glb',
    
    # STL exporter
    'convert_heightmap_to_stl',
    'export_stl',
    
    # OBJ exporter
    'convert_heightmap_to_obj',
    'export_obj',
    
    # PLY exporter
    'convert_heightmap_to_ply',
    
    # NVIDIA binary data exporter
    'convert_heightmap_to_nvbd',
    
    # USD/USDZ exporters
    'convert_heightmap_to_usd',
    'convert_heightmap_to_usdz',
    
    # Adaptive mesh generator
    'convert_heightmap_to_adaptive_mesh'
]
