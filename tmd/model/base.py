"""Base classes for model exporters."""

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
        
        # Model dimensions - will be set based on heightmap if not provided
        self.x_length = kwargs.get('x_length', None)
        self.y_length = kwargs.get('y_length', None)
        
        # Apply uniform scale if provided
        if self.scale is not None:
            self.x_scale = self.scale
            self.y_scale = self.scale
            self.z_scale = self.scale
            if self.x_length is not None:
                self.x_length *= self.scale
            if self.y_length is not None:
                self.y_length *= self.scale
        
        # Position parameters
        self.x_offset = kwargs.get('x_offset', 0.0)
        self.y_offset = kwargs.get('y_offset', 0.0)
        self.base_height = kwargs.get('base_height', 0.0)
        
        # Triangulation parameters
        self.triangulation_method = str(kwargs.get('method', 'adaptive')).replace('MeshMethod.', '').lower()
        self.error_threshold = kwargs.get('error_threshold', 0.01)
        self.min_quad_size = kwargs.get('min_quad_size', 2)
        self.max_quad_size = kwargs.get('max_quad_size', 32)
        self.max_triangles = kwargs.get('max_triangles', None)
        
        # Format parameters 
        self.binary = kwargs.get('binary', None)
        self.optimize = kwargs.get('optimize', True)
        self.calculate_normals = kwargs.get('calculate_normals', True)
        
        # Mesh parameters
        self.coordinate_system = kwargs.get('coordinate_system', 'right-handed')
        self.origin_at_zero = kwargs.get('origin_at_zero', True)
        
        # Store any remaining parameters
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
    
    def optimize(self) -> None:
        """
        Optimize the mesh by merging close vertices and removing degenerate faces.
        """
        from .utils.mesh import optimize_mesh
        try:
            result = optimize_mesh(self.vertices, self.faces)
            if result is not None:
                self.vertices, self.faces = result
                # Reset derived attributes after optimization
                self.normals = None
                self.uvs = None
            else:
                logger.warning("Mesh optimization had no effect")
        except Exception as e:
            logger.error(f"Mesh optimization failed: {e}")
            # Continue without optimization
    
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
    def create_mesh_from_heightmap(cls, height_map: np.ndarray, config: ExportConfig) -> MeshData:
        """Create a mesh from a heightmap using the provided configuration."""
        height_map = cls._prepare_heightmap_for_triangulation(height_map)
        
        # Set dimensions based on heightmap if not provided
        rows, cols = height_map.shape
        if config.x_length is None:
            config.x_length = float(cols)
        if config.y_length is None:
            config.y_length = float(rows)

        # Scale parameters - adjust z_scale relative to x/y dimensions
        z_scale = config.z_scale * min(config.x_length, config.y_length) / max(rows, cols)
        x_scale = config.x_length / (cols - 1)
        y_scale = config.y_length / (rows - 1)
        
        # Select triangulation method
        if config.triangulation_method == 'quadtree':
            from .triangulation.quadtree import triangulate_heightmap_quadtree
            vertices, faces, stats = triangulate_heightmap_quadtree(
                height_map=height_map,
                max_triangles=config.max_triangles or 50000,
                error_threshold=config.error_threshold,
                z_scale=z_scale,
                max_subdivisions=config.extra.get('max_subdivisions', 8),
                detail_boost=config.extra.get('detail_boost', 1.0),
                progress_callback=config.progress_callback
            )
        else:
            from .triangulation.adaptive import triangulate_heightmap
            vertices, faces, stats = triangulate_heightmap(
                height_map=height_map,
                max_triangles=config.max_triangles,
                error_threshold=config.error_threshold,
                z_scale=z_scale,
                detail_boost=config.extra.get('detail_boost', 1.0),
                progress_callback=config.progress_callback
            )

        # Convert vertices to numpy array
        vertices = np.array(vertices, dtype=np.float32)
        
        # Apply scaling and offset
        vertices[:, 0] *= x_scale  # Scale X
        vertices[:, 1] *= y_scale  # Scale Y
        vertices[:, 0] += config.x_offset
        vertices[:, 1] += config.y_offset

        # Create MeshData object
        mesh = MeshData(vertices, faces)
        
        # Post-processing
        if config.optimize:
            mesh.optimize()
        if config.calculate_normals:
            mesh.ensure_normals()
        mesh.ensure_uvs()
        
        return mesh

    @staticmethod
    def _prepare_heightmap_for_triangulation(height_map: np.ndarray) -> np.ndarray:
        """
        Prepare heightmap for triangulation by converting to 16-bit grayscale.
        
        Args:
            height_map: Input heightmap array
            
        Returns:
            16-bit normalized heightmap
        """
        # Ensure floating point for calculations
        height_map = height_map.astype(np.float32)
        
        # Normalize to [0, 1] range
        min_val = np.min(height_map)
        max_val = np.max(height_map)
        height_range = max_val - min_val
        
        if height_range > 0:
            height_map = (height_map - min_val) / height_range
        else:
            height_map = np.zeros_like(height_map)
        
        # Convert to 16-bit integer range [0, 65535]
        height_map = (height_map * 65535).astype(np.uint16)
        
        # Convert back to float32 but preserve 16-bit precision
        height_map = height_map.astype(np.float32) / 65535.0
        
        logger.debug(f"Prepared heightmap: shape={height_map.shape}, dtype={height_map.dtype}, range=[{height_map.min():.3f}, {height_map.max():.3f}]")
        
        return height_map


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