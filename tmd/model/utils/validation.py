"""Validation utilities for TMD model operations."""

import os
import numpy as np
import logging
from typing import Union, List, Tuple, Optional, Any, Dict

# Set up logging
logger = logging.getLogger(__name__)


def validate_vertices(vertices: Union[np.ndarray, List[List[float]]]) -> bool:
    """
    Validate vertex array or list.
    
    Args:
        vertices: Array/list of 3D vertices to validate
        
    Returns:
        True if valid, False otherwise
    """
    if vertices is None:
        return False
        
    try:
        # Convert to numpy array if needed
        if not isinstance(vertices, np.ndarray):
            vertices = np.array(vertices)
            
        # Check shape
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            return False
            
        # Check for valid values
        if not np.isfinite(vertices).all():
            return False
            
        return True
        
    except:
        return False


def validate_faces(faces: Union[np.ndarray, List[List[int]]]) -> bool:
    """
    Validate face index array or list.
    
    Args:
        faces: Array/list of triangle indices to validate
        
    Returns:
        True if valid, False otherwise
    """
    if faces is None:
        return False
        
    try:
        # Convert to numpy array if needed
        if not isinstance(faces, np.ndarray):
            faces = np.array(faces)
            
        # Check shape
        if faces.ndim != 2 or faces.shape[1] != 3:
            return False
            
        # Check for valid indices
        if np.any(faces < 0):
            return False
            
        return True
        
    except:
        return False


def validate_heightmap(height_map: np.ndarray) -> bool:
    """
    Validate a heightmap array.
    
    Args:
        height_map: 2D numpy array to validate
        
    Returns:
        True if valid, False otherwise
    """
    if height_map is None:
        return False
        
    if not isinstance(height_map, np.ndarray):
        return False
        
    if height_map.ndim != 2:
        return False
        
    if height_map.size == 0:
        return False
        
    if height_map.shape[0] < 2 or height_map.shape[1] < 2:
        return False
        
    return True


def ensure_directory_exists(filename: str) -> bool:
    """
    Ensure the directory for a file exists, creating it if needed.
    
    Args:
        filename: Path to file
        
    Returns:
        True if directory exists or was created, False on error
    """
    try:
        directory = os.path.dirname(filename)
        if directory:
            os.makedirs(directory, exist_ok=True)
        return True
    except:
        return False


def validate_mesh(
    vertices: np.ndarray, 
    faces: np.ndarray
) -> Tuple[bool, List[str]]:
    """
    Validate mesh data for common issues.
    
    Args:
        vertices: Array of vertex positions
        faces: Array of face indices
        
    Returns:
        Tuple of (is_valid, issues) where issues is a list of problems found
    """
    issues = []
    
    # Basic validation
    if not validate_vertices(vertices):
        issues.append("Invalid vertex data")
        
    if not validate_faces(faces):
        issues.append("Invalid face data")
    
    # Additional checks
    if len(vertices) == 0:
        issues.append("Empty mesh (no vertices)")
        
    if len(faces) == 0:
        issues.append("Empty mesh (no faces)")
    
    # Check for non-manifold edges
    edge_count = {}
    for face in faces:
        edges = [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]
        for v1, v2 in edges:
            edge = tuple(sorted([v1, v2]))
            edge_count[edge] = edge_count.get(edge, 0) + 1
            
    for edge, count in edge_count.items():
        if count > 2:
            issues.append(f"Non-manifold edge found: {edge}")
    
    return len(issues) == 0, issues


def validate_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate configuration parameters against expected types and ranges.
    
    Args:
        config: Dictionary of configuration parameters
        
    Returns:
        Tuple of (is_valid, issues) where issues is a list of problems found
    """
    issues = []
    
    # Validate numeric parameters
    numeric_params = {
        'z_scale': {'min': 0, 'max': None, 'type': (int, float)},
        'base_height': {'min': 0, 'max': None, 'type': (int, float)},
        'x_length': {'min': 0, 'max': None, 'type': (int, float)},
        'y_length': {'min': 0, 'max': None, 'type': (int, float)},
        'max_triangles': {'min': 1, 'max': None, 'type': int},
        'error_threshold': {'min': 0, 'max': None, 'type': (int, float)}
    }
    
    for param, valid in numeric_params.items():
        if param in config:
            value = config[param]
            
            # Check type
            if not isinstance(value, valid['type']):
                issues.append(f"Parameter '{param}' must be of type {valid['type']}, got {type(value)}")
                continue
            
            # Check minimum value
            if valid['min'] is not None and value < valid['min']:
                issues.append(f"Parameter '{param}' must be >= {valid['min']}, got {value}")
            
            # Check maximum value
            if valid['max'] is not None and value > valid['max']:
                issues.append(f"Parameter '{param}' must be <= {valid['max']}, got {value}")
    
    # Validate enum parameters
    enum_params = {
        'coordinate_system': ['right-handed', 'left-handed'],
        'uv_mapping': ['planar', 'cylindrical', 'spherical']
    }
    
    for param, valid_values in enum_params.items():
        if param in config:
            value = config[param]
            if value not in valid_values:
                issues.append(f"Parameter '{param}' must be one of {valid_values}, got '{value}'")
    
    # Validate boolean parameters
    bool_params = ['binary', 'texture', 'optimize', 'calculate_normals', 'origin_at_zero']
    
    for param in bool_params:
        if param in config and not isinstance(config[param], bool):
            issues.append(f"Parameter '{param}' must be a boolean, got {type(config[param])}")
    
    return len(issues) == 0, issues
