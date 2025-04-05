"""STL exporter implementation."""
import numpy as np
import struct
import logging
from typing import Optional

from ..base import ModelExporter, ExportConfig, MeshData
from ..utils.validation import validate_heightmap, ensure_directory_exists
from ..utils.mesh import calculate_face_normals
from ..registry import register_exporter

# Set up logging
logger = logging.getLogger(__name__)

@register_exporter
class STLExporter(ModelExporter):
    """STL format exporter."""
    format_name = "stl"  # Changed to lowercase for consistent lookup
    file_extensions = ["stl"]
    binary_supported = True
    
    @classmethod
    def export(cls, height_map: np.ndarray, filename: str, config: ExportConfig) -> Optional[str]:
        """Export height map as STL file."""
        try:
            # Generate mesh from heightmap
            mesh = cls.create_mesh_from_heightmap(height_map, config)
            
            # Export as binary or ASCII STL based on config
            binary = config.binary if config.binary is not None else True
            if binary:
                write_binary_stl(mesh, filename)
            else:
                write_ascii_stl(mesh, filename)
                
            logger.info(f"Successfully exported STL file: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"STL export failed: {e}")
            import traceback
            traceback.print_exc()
            return None


def write_binary_stl(mesh: MeshData, filename: str) -> None:
    """
    Write mesh data to a binary STL file.
    
    Args:
        mesh: MeshData object containing the mesh to export
        filename: Output filename
    """
    # Ensure normals are calculated
    mesh.ensure_normals()
    
    # Calculate face normals for STL
    face_normals = _calculate_face_normals(mesh)
    
    with open(filename, 'wb') as f:
        # Write STL header (80 bytes)
        f.write(b'TMD STL Exporter' + b' ' * (80 - 15))
        
        # Write number of triangles (4 bytes)
        f.write(struct.pack('<I', mesh.face_count))
        
        # Write each triangle
        for i, face in enumerate(mesh.faces):
            # Normal vector
            f.write(struct.pack('<fff', *face_normals[i]))
            
            # Vertices (3 points)
            for idx in face:
                f.write(struct.pack('<fff', *mesh.vertices[idx]))
            
            # Attribute byte count (2 bytes, usually zero)
            f.write(struct.pack('<H', 0))


def write_ascii_stl(mesh: MeshData, filename: str) -> None:
    """
    Write mesh data to an ASCII STL file.
    
    Args:
        mesh: MeshData object containing the mesh to export
        filename: Output filename
    """
    # Ensure normals are calculated
    mesh.ensure_normals()
    
    # Calculate face normals for STL
    face_normals = _calculate_face_normals(mesh)
    
    with open(filename, 'w') as f:
        f.write("solid TMD_Generated\n")
        
        for i, face in enumerate(mesh.faces):
            nx, ny, nz = face_normals[i]
            f.write(f"  facet normal {nx:.6e} {ny:.6e} {nz:.6e}\n")
            f.write("    outer loop\n")
            
            for idx in face:
                x, y, z = mesh.vertices[idx]
                f.write(f"      vertex {x:.6e} {y:.6e} {z:.6e}\n")
            
            f.write("    endloop\n")
            f.write("  endfacet\n")
        
        f.write("endsolid TMD_Generated\n")


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