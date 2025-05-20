"""
Map generators for image processing.

This module provides various map generators for creating specialized maps
from height maps, such as normal maps, roughness maps, etc.
"""
from .base_generator import MapGenerator
from .ao import AOMapGenerator
from .bump import BumpMapGenerator
from .roughness import RoughnessMapGenerator
from .metallic import MetallicMapGenerator
from .displacement import DisplacementMapGenerator
from .heightmap import HeightMapGenerator
from .hillshade import HillshadeMapGenerator
from .curvature import CurvatureMapGenerator
from .angle import AngleMapGenerator

__all__ = [
    'MapGenerator',
    'AOMapGenerator',
    'BumpMapGenerator',
    'RoughnessMapGenerator',
    'MetallicMapGenerator',
    'DisplacementMapGenerator',
    'HeightMapGenerator',
    'HillshadeMapGenerator',
    'CurvatureMapGenerator',
    'AngleMapGenerator'
]
