"""
Heightmap triangulation algorithms.

This package provides algorithms for converting heightmaps to triangular meshes
with various optimization strategies.
"""

from .base import BaseTriangulator
from .adaptive import AdaptiveTriangulator, triangulate_heightmap

# Define package exports
__all__ = [
    'BaseTriangulator',
    'AdaptiveTriangulator',
    'triangulate_heightmap'
]