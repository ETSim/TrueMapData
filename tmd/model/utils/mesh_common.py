"""Common mesh utilities for all exporters."""
import numpy as np
from typing import Tuple, Optional
import logging
from ..base import MeshData

logger = logging.getLogger(__name__)

def prepare_mesh_for_export(mesh: MeshData, config: dict) -> Optional[MeshData]:
    """
    Prepare mesh for export by ensuring it's watertight and has required attributes.
    
    Args:
        mesh: Input mesh
        config: Export configuration
        
    Returns:
        Processed mesh ready for export
    """
    try:
        # Create copy of vertices and faces
        vertices = mesh.vertices.copy()
        faces = mesh.faces.copy()

        # Add base if needed
        if config.get('base_height', 0.0) > 0:
            from .mesh import ensure_watertight_mesh
            vertices, faces = ensure_watertight_mesh(
                vertices, faces, 
                min_base_height=config['base_height']
            )

        # Create new mesh with processed data
        processed_mesh = MeshData(vertices, faces)

        # Optimize if requested
        if config.get('optimize', True):
            processed_mesh.optimize()

        # Ensure we have normals
        processed_mesh.ensure_normals(force_recalculate=True)

        # Generate UVs if needed
        if config.get('generate_uvs', False):
            processed_mesh.ensure_uvs(method=config.get('uv_method', 'planar'))

        return processed_mesh

    except Exception as e:
        logger.error(f"Failed to prepare mesh: {e}")
        return None

def scale_mesh_vertices(vertices: np.ndarray, scale: float) -> np.ndarray:
    """Scale mesh vertices uniformly."""
    return vertices * scale if scale != 1.0 else vertices

def reorient_mesh_faces(mesh: MeshData) -> None:
    """Ensure consistent face orientation with upward normals."""
    for i, face in enumerate(mesh.faces):
        v1 = mesh.vertices[face[0]]
        v2 = mesh.vertices[face[1]]
        v3 = mesh.vertices[face[2]]
        
        normal = np.cross(v2 - v1, v3 - v1)
        if normal[2] < 0:  # Z component should be positive
            mesh.faces[i] = [face[0], face[2], face[1]]
