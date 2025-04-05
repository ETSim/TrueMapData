"""Validation utilities for model generation."""

import numpy as np
import os
import logging
from typing import Tuple, Optional, Any, Dict, List, Union

# Set up logging
logger = logging.getLogger(__name__)


def validate_heightmap(height_map: np.ndarray) -> bool:
    """
    Validate that input is a valid height map.
    
    Args:
        height_map: Array to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(height_map, np.ndarray):
        return False
        
    if height_map.ndim != 2:
        return False
        
    if height_map.size == 0:
        return False
        
    return True


def ensure_directory_exists(path: str) -> bool:
    """
    Ensure directory exists, creating if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        True if directory exists/created, False on error
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return True
    except Exception:
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
    
    # Check for empty mesh
    if len(vertices) == 0:
        issues.append("Empty mesh (no vertices)")
    
    if len(faces) == 0:
        issues.append("Empty mesh (no faces)")
    
    if not issues:  # Only continue if we have vertices and faces
        # Check for invalid indices
        max_index = np.max(faces) if len(faces) > 0 else -1
        if max_index >= len(vertices):
            issues.append(f"Invalid vertex indices: max index {max_index} >= vertex count {len(vertices)}")
        
        # Check for negative indices
        if np.min(faces) < 0:
            issues.append(f"Invalid negative vertex indices found")
        
        # Check for degenerate faces (duplicated vertices in a face)
        for i, face in enumerate(faces):
            if len(set(face)) != len(face):
                issues.append(f"Degenerate face at index {i}: {face}")
        
        # Check for NaN or infinite values
        if np.isnan(vertices).any():
            issues.append("Mesh contains NaN vertex coordinates")
        
        if np.isinf(vertices).any():
            issues.append("Mesh contains infinite vertex coordinates")
    
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
