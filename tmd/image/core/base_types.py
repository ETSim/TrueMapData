"""
Base types for the image processing package.
"""
from typing import NewType, Union, Dict, Any
import numpy as np

# Type definitions for stronger typing
HeightMap = NewType('HeightMap', np.ndarray)
MapData = NewType('MapData', np.ndarray)

# Parameters type for map generation
MapParams = Dict[str, Any]
