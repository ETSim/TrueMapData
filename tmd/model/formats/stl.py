"""STL exporter module for height maps."""

import os
import numpy as np
import struct
import logging
from typing import Optional, List, Tuple, Union

from ..base import ModelExporter, ExportConfig, MeshData
from ..utils.validation import validate_heightmap, ensure_directory_exists
from . import base

from ..base import ModelExporter, ExportConfig, MeshData
from ..registry import register_exporter
from ...cli.core.ui import print_error, print_warning

@register_exporter
class STLExporter(ModelExporter):
    """STL format exporter."""
    format_name = "stl"
    file_extensions = ["stl"]
    binary_supported = True
    
    @classmethod
    def export(cls, height_map: np.ndarray, filename: str, config: ExportConfig) -> Optional[str]:
        """Export height map as STL file."""
        try:
            # Validate heightmap
            if not validate_heightmap(height_map):
                logger.error("Invalid heightmap data")
                return None
            
            # Ensure config has required parameters
            required_params = ['x_length', 'y_length', 'z_scale']
            missing_params = [p for p in required_params if not hasattr(config, p)]
            if missing_params:
                logger.error(f"Missing required config parameters: {missing_params}")
                return None
                
            # Ensure output directory exists
            if not ensure_directory_exists(filename):
                logger.error(f"Failed to create directory for: {filename}")
                return None

            # Generate mesh from heightmap
            mesh = cls.create_mesh_from_heightmap(height_map, config)
            
            # Get base height (default to 1% of total height)
            base_height = config.base_height if hasattr(config, 'base_height') else (np.max(height_map) - np.min(height_map)) * 0.01
            
            # Ensure mesh is watertight and has a base
            vertices, faces = ensure_watertight_mesh(
                mesh.vertices,
                mesh.faces,
                min_base_height=base_height
            )
            mesh = MeshData(vertices, faces)
            
            # Apply any additional processing
            if config.optimize:
                mesh.optimize()
                
            # Ensure normals are recalculated after modifications
            mesh.ensure_normals(force_recalculate=True)

            # Export as binary or ASCII STL based on config
            binary = config.binary if config.binary is not None else True
            if binary:
                write_binary_stl(mesh, filename, config.z_scale)
            else:
                write_ascii_stl(mesh, filename, config.z_scale)

            # Save heightmap visualization if requested
            if config.extra.get('save_heightmap', True):
                # Create image filename by replacing extension
                img_filename = os.path.splitext(filename)[0] + '_heightmap.png'
                
                # Generate colored heightmap
                colored_heightmap = generate_heightmap_texture(
                    height_map,
                    colormap=config.extra.get('colormap', 'terrain'),
                    resolution=config.extra.get('texture_resolution', None)
                )
                
                # Save image using PIL
                try:
                    from PIL import Image
                    Image.fromarray(colored_heightmap).save(img_filename)
                    logger.info(f"Saved heightmap visualization to {img_filename}")
                except ImportError:
                    # Fallback to OpenCV
                    try:
                        import cv2
                        # OpenCV uses BGR, so convert from RGB
                        cv2.imwrite(img_filename, cv2.cvtColor(colored_heightmap, cv2.COLOR_RGB2BGR))
                        logger.info(f"Saved heightmap visualization to {img_filename}")
                    except ImportError:
                        logger.warning("Could not save heightmap visualization - requires PIL or OpenCV")
                
            logger.info(f"Successfully exported STL file: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"STL export failed: {e}")
            import traceback
            traceback.print_exc()
            return None


def write_binary_stl(mesh: MeshData, filename: str, z_scale: float) -> None:
    """Write mesh data to a binary STL file."""
    with open(filename, 'wb') as f:
        # Write header
        header = b'TMD Mesh Exporter'
        f.write(header + b' ' * (80 - len(header)))
        
        # Write triangle count
        f.write(struct.pack('<I', len(mesh.faces)))
        
        # Create scaled vertices array once
        scaled_vertices = mesh.vertices.copy()
        # Only scale Z coordinates
        scaled_vertices[:, 2] *= z_scale
        
        # Write each triangle
        for face in mesh.faces:
            # Get vertex positions for this triangle with proper orientation
            v1 = scaled_vertices[face[0]]
            v2 = scaled_vertices[face[1]]  
            v3 = scaled_vertices[face[2]]
            
            # Calculate normal with correct orientation
            edge1 = v2 - v1
            edge2 = v3 - v1
            normal = np.cross(edge1, edge2)
            length = np.linalg.norm(normal)
            normal = normal / max(length, 1e-10)  # Safe normalization
            
            # Ensure normal points upward (for heightmaps)
            if normal[2] < 0:
                normal *= -1
                v2, v3 = v3, v2  # Swap vertices to maintain correct winding
            
            # Write data
            f.write(struct.pack('<fff', *normal))
            f.write(struct.pack('<fff', *v1))
            f.write(struct.pack('<fff', *v2))
            f.write(struct.pack('<fff', *v3))
            f.write(struct.pack('<H', 0))


def write_ascii_stl(mesh: MeshData, filename: str, z_scale: float) -> None:
    """Write mesh data to an ASCII STL file."""
    with open(filename, 'w') as f:
        f.write("solid\n")
        
        # Create scaled vertices array
        scaled_vertices = mesh.vertices.copy()
        scaled_vertices[:, 2] *= z_scale
        
        for face in mesh.faces:
            # Get vertices with proper orientation
            v1 = scaled_vertices[face[0]]
            v2 = scaled_vertices[face[1]]
            v3 = scaled_vertices[face[2]]
            
            # Calculate normal
            edge1 = v2 - v1
            edge2 = v3 - v1
            normal = np.cross(edge1, edge2)
            length = np.linalg.norm(normal)
            normal = normal / max(length, 1e-10)  # Safe normalization
            
            # Ensure normal points upward
            if normal[2] < 0:
                normal *= -1
                v2, v3 = v3, v2  # Swap vertices to maintain correct winding
            
            # Write facet
            f.write(f"  facet normal {normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}\n")
            f.write("    outer loop\n")
            f.write(f"      vertex {v1[0]:.6e} {v1[1]:.6e} {v1[2]:.6e}\n")
            f.write(f"      vertex {v2[0]:.6e} {v2[1]:.6e} {v2[2]:.6e}\n")
            f.write(f"      vertex {v3[0]:.6e} {v3[1]:.6e} {v3[2]:.6e}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        
        f.write("endsolid\n")


def _calculate_face_normals(mesh: MeshData) -> np.ndarray:
    """
    Calculate normal vectors for each face.
    
    Args:
        mesh: MeshData object
        
    Returns:
        Array of normal vectors for each face
    """
    normals = np.zeros((mesh.face_count, 3), dtype=np.float32)
    
    for i, face in enumerate(mesh.faces):
        # Get vertices of this face
        v0 = mesh.vertices[face[0]]
        v1 = mesh.vertices[face[1]]
        v2 = mesh.vertices[face[2]]
        
        # Calculate edges
        edge1 = v1 - v0
        edge2 = v2 - v0
        
        # Calculate normal using cross product
        normal = np.cross(edge1, edge2)
        
        # Normalize
        norm = np.linalg.norm(normal)
        if norm > 0:
            normal = normal / norm
            
        normals[i] = normal
    
    return normals
