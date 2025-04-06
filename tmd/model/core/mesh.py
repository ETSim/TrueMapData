"""Core mesh functionality and data structures."""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from ..utils.validation import validate_vertices, validate_faces

@dataclass
class MeshData:
    """Container for mesh geometry and attributes."""
    vertices: np.ndarray  # Nx3 array of vertex positions
    faces: np.ndarray    # Mx3 array of vertex indices
    normals: Optional[np.ndarray] = None  # Nx3 array of vertex normals
    uvs: Optional[np.ndarray] = None      # Nx2 array of texture coordinates
    colors: Optional[np.ndarray] = None    # Nx3 array of vertex colors
    
    def __post_init__(self):
        """Validate mesh data on creation."""
        if not validate_vertices(self.vertices):
            raise MeshValidationError("Invalid vertex data")
        if not validate_faces(self.faces, len(self.vertices)):
            raise MeshValidationError("Invalid face data")

class MeshError(Exception):
    """Base class for mesh-related exceptions."""
    pass

class MeshValidationError(MeshError):
    """Raised when mesh data fails validation."""
    pass

class MeshOperationError(MeshError):
    """Raised when a mesh operation fails."""
    pass
