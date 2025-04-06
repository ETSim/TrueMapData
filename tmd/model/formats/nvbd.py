"""NVBD exporter implementation."""
import numpy as np
import struct
import logging
from typing import Optional

from ..base import ModelExporter, ExportConfig, MeshData
from ..utils.validation import validate_heightmap, ensure_directory_exists
from ..registry import register_exporter

logger = logging.getLogger(__name__)

@register_exporter
class NVBDExporter(ModelExporter):
    """NVIDIA Binary Data format exporter."""
    format_name = "nvbd"
    file_extensions = ["nvbd"]
    binary_supported = True
    
    @classmethod
    def export(cls, height_map: np.ndarray, filename: str, config: ExportConfig) -> Optional[str]:
        """Export height map as NVBD file."""
        try:
            # Create mesh
            mesh = cls.create_mesh_from_heightmap(height_map, config)
            
            # Prepare mesh using common utilities
            from ..utils.mesh_common import prepare_mesh_for_export
            processed_mesh = prepare_mesh_for_export(mesh, config.__dict__)
            if processed_mesh is None:
                return None
            
            # Write NVBD format
            write_nvbd(processed_mesh, filename)
            
            return filename
            
        except Exception as e:
            logger.error(f"NVBD export failed: {e}")
            return None

def write_nvbd(mesh: MeshData, filename: str) -> None:
    """Write mesh data to NVBD format."""
    with open(filename, 'wb') as f:
        # Write header
        f.write(b'NVBD')
        f.write(struct.pack('<I', 1))  # Version
        
        # Write counts
        f.write(struct.pack('<I', len(mesh.vertices)))
        f.write(struct.pack('<I', len(mesh.faces)))
        
        # Write vertex data
        for vertex in mesh.vertices:
            f.write(struct.pack('<fff', *vertex))
        
        # Write face data
        for face in mesh.faces:
            f.write(struct.pack('<III', *face))
        
        # Write normal data if available
        if mesh.normals is not None:
            f.write(struct.pack('<I', 1))  # Has normals flag
            for normal in mesh.normals:
                f.write(struct.pack('<fff', *normal))
        else:
            f.write(struct.pack('<I', 0))  # No normals flag