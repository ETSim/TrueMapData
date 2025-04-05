"""
Base classes for model exporters.

This module defines the abstract base classes for model exporters, mesh data,
and common utility functions for creating meshes from heightmaps.
"""

import os
import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Any, List, Union, ClassVar, Protocol, TypeVar

# Set up logging
logger = logging.getLogger(__name__)

# Type definitions
Vertex = Tuple[float, float, float]
VertexList = List[Vertex]
Face = List[int]
FaceList = List[Face]
Mesh = Tuple[VertexList, FaceList]
Normal = Tuple[float, float, float]
NormalList = List[Normal]
UV = Tuple[float, float]
UVList = List[UV]
Color = Tuple[int, int, int]
ColorList = List[Color]


class ExportConfig:
    """Configuration parameters for model export operations."""
    
    def __init__(self, **kwargs):
        """Initialize export configuration with specified parameters."""
        # Scale parameters
        self.z_scale = kwargs.get('z_scale', 1.0)
        self.scale = kwargs.get('scale', None)
        self.x_scale = kwargs.get('x_scale', 1.0)
        self.y_scale = kwargs.get('y_scale', 1.0)
        
        # Apply uniform scale if provided
        if self.scale is not None:
            self.x_scale = self.scale
            self.y_scale = self.scale
            self.z_scale = self.scale
        
        # Mesh generation parameters
        self.triangulation_method = str(kwargs.get('method', 'adaptive')).replace('MeshMethod.', '').lower()
        self.error_threshold = kwargs.get('error_threshold', 0.01)
        self.min_quad_size = kwargs.get('min_quad_size', 2)
        self.max_quad_size = kwargs.get('max_quad_size', 32)
        self.curvature_threshold = kwargs.get('curvature_threshold', 0.1)
        self.max_triangles = kwargs.get('max_triangles', None)
        self.simplify_ratio = kwargs.get('simplify_ratio', None)
        self.use_feature_edges = kwargs.get('use_feature_edges', True)
        self.smoothing = kwargs.get('smoothing', 0.0)
        
        # Format parameters
        self.binary = kwargs.get('binary', None)
        self.texture = kwargs.get('texture', False)
        self.color_map = kwargs.get('color_map', 'terrain')
        self.texture_resolution = kwargs.get('texture_resolution', None)
        self.optimize = kwargs.get('optimize', True)
        
        # Mesh parameters
        self.calculate_normals = kwargs.get('calculate_normals', True)
        self.coordinate_system = kwargs.get('coordinate_system', 'right-handed')
        self.normals_inside = kwargs.get('normals_inside', False)
        self.origin_at_zero = kwargs.get('origin_at_zero', True)
        self.base_height = kwargs.get('base_height', 0.0)
        
        # Additional parameters
        self.extra = {k: v for k, v in kwargs.items() if not hasattr(self, k)}
        
    def __repr__(self) -> str:
        """Return string representation of config."""
        params = [f"{k}={repr(v)}" for k, v in vars(self).items() if not k.startswith('_')]
        return f"ExportConfig({', '.join(params)})"


class MeshData:
    """Container for mesh data with vertices, faces, normals, UVs, and colors."""
    
    def __init__(self, 
                 vertices: Union[VertexList, np.ndarray],
                 faces: Union[FaceList, np.ndarray],
                 normals: Optional[Union[NormalList, np.ndarray]] = None,
                 uvs: Optional[Union[UVList, np.ndarray]] = None,
                 colors: Optional[Union[ColorList, np.ndarray]] = None):
        """
        Initialize mesh data container.
        
        Args:
            vertices: List of (x, y, z) vertices or numpy array
            faces: List of face indices or numpy array
            normals: Optional list of (nx, ny, nz) normals or numpy array
            uvs: Optional list of (u, v) texture coordinates or numpy array
            colors: Optional list of (r, g, b) colors or numpy array
        """
        # Convert to numpy arrays if provided as lists
        self.vertices = np.array(vertices, dtype=np.float32)
        self.faces = np.array(faces, dtype=np.int32)
        
        # Optional attributes
        self.normals = np.array(normals, dtype=np.float32) if normals is not None else None
        self.uvs = np.array(uvs, dtype=np.float32) if uvs is not None else None
        self.colors = np.array(colors, dtype=np.uint8) if colors is not None else None
    
    @property
    def vertex_count(self) -> int:
        """Get the number of vertices in the mesh."""
        return len(self.vertices)
    
    @property
    def face_count(self) -> int:
        """Get the number of faces in the mesh."""
        return len(self.faces)
    
    def ensure_normals(self, force_recalculate: bool = False) -> None:
        """
        Ensure the mesh has normal vectors, calculating them if needed.
        
        Args:
            force_recalculate: If True, recalculate normals even if already present
        """
        if self.normals is None or force_recalculate:
            from .utils.mesh import calculate_vertex_normals
            self.normals = calculate_vertex_normals(self.vertices, self.faces)
    
    def ensure_uvs(self, method: str = 'planar') -> None:
        """
        Ensure the mesh has UV coordinates, generating them if needed.
        
        Args:
            method: UV mapping method ('planar', 'cylindrical', 'spherical')
        """
        if self.uvs is None:
            from .utils.mesh import generate_uv_coordinates
            self.uvs = generate_uv_coordinates(self.vertices, method=method)
    
    def optimize(self, tolerance: float = 1e-10) -> None:
        """
        Optimize the mesh by merging close vertices and removing degenerate faces.
        
        Args:
            tolerance: Distance tolerance for vertex merging
        """
        from .utils.mesh import optimize_mesh
        self.vertices, self.faces = optimize_mesh(self.vertices, self.faces, tolerance)
        
        # Reset derived attributes after optimization
        self.normals = None
        self.uvs = None
    
    def as_dict(self) -> Dict[str, Any]:
        """
        Get mesh data as a dictionary.
        
        Returns:
            Dictionary with mesh components
        """
        result = {
            'vertices': self.vertices,
            'faces': self.faces
        }
        
        if self.normals is not None:
            result['normals'] = self.normals
            
        if self.uvs is not None:
            result['uvs'] = self.uvs
            
        if self.colors is not None:
            result['colors'] = self.colors
            
        return result
    
    def __repr__(self) -> str:
        """String representation of the mesh."""
        attrs = [f"vertices={self.vertex_count}", f"faces={self.face_count}"]
        if self.normals is not None:
            attrs.append("normals=True")
        if self.uvs is not None:
            attrs.append("uvs=True")
        if self.colors is not None:
            attrs.append("colors=True")
            
        return f"MeshData({', '.join(attrs)})"


class ModelExporter(ABC):
    """
    Abstract base class for all model exporters.
    
    Concrete exporters should implement the abstract methods and register
    themselves with the model exporter registry.
    """
    # Class attributes to be defined by subclasses
    format_name: ClassVar[str] = ""
    file_extensions: ClassVar[List[str]] = []
    binary_supported: ClassVar[bool] = False
    
    @classmethod
    @abstractmethod
    def export(cls, 
               height_map: np.ndarray, 
               filename: str, 
               config: ExportConfig) -> Optional[str]:
        """
        Export a height map to a 3D model file.
        
        Args:
            height_map: 2D numpy array of height values
            filename: Output filename
            config: Export configuration parameters
            
        Returns:
            Path to the created file if successful, None otherwise
        """
        pass
    
    @classmethod
    def ensure_extension(cls, filename: str) -> str:
        """
        Ensure filename has the correct extension for this format.
        
        Args:
            filename: Original filename
            
        Returns:
            Filename with correct extension
        """
        # Use the first extension if multiple are supported
        if not cls.file_extensions:
            return filename
            
        ext = cls.file_extensions[0]
        if not filename.lower().endswith(f".{ext.lower()}"):
            filename = f"{os.path.splitext(filename)[0]}.{ext}"
            
        return filename
    
    @classmethod
    def create_mesh_from_heightmap(cls, 
                                  height_map: np.ndarray, 
                                  config: ExportConfig) -> MeshData:
        """
        Create a mesh from a heightmap using the provided configuration.
        
        Args:
            height_map: 2D numpy array of height values
            config: Export configuration parameters
            
        Returns:
            MeshData object containing the created mesh
        """
        # Create the base mesh from the heightmap
        from .utils.mesh import create_mesh_from_heightmap
        vertices, faces = create_mesh_from_heightmap(
            height_map=height_map,
            x_offset=config.x_offset,
            y_offset=config.y_offset,
            x_length=config.x_length,
            y_length=config.y_length,
            z_scale=config.z_scale,
            base_height=config.base_height
        )
        
        # Create MeshData object
        mesh = MeshData(vertices, faces)
        
        # Optimize if requested
        if config.optimize:
            mesh.optimize()
        
        # Calculate normals if requested
        if config.calculate_normals:
            mesh.ensure_normals()
        
        # Generate UVs
        mesh.ensure_uvs()
        
        return mesh


def export_heightmap_to_model(
    height_map: np.ndarray,
    filename: str,
    format_name: str,
    **kwargs
) -> Optional[str]:
    """
    Export a height map to a 3D model file.

    Args:
        height_map: 2D numpy array of height values
        filename: Output filename
        format_name: Format name for the model (e.g., 'stl', 'obj', 'ply')
        **kwargs: Additional parameters for export

    Returns:
        str: Path to the created file or None if failed
    """
    # Import here to avoid circular imports
    from .factory import ModelExporterFactory
    
    # Create configuration from kwargs
    config = ExportConfig(**kwargs)
    
    # Get the exporter and export the heightmap
    return ModelExporterFactory.export_heightmap(
        height_map=height_map,
        filename=filename,
        format_name=format_name,
        config=config
    )