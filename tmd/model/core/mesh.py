"""Fixed core mesh functionality and data structures."""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple, Union
import logging

logger = logging.getLogger(__name__)

class MeshError(Exception):
    """Base class for mesh-related exceptions."""
    pass

class MeshValidationError(MeshError):
    """Raised when mesh data fails validation."""
    pass

class MeshOperationError(MeshError):
    """Raised when a mesh operation fails."""
    pass

def validate_vertices(vertices: np.ndarray) -> bool:
    """Validate vertex array."""
    if vertices is None:
        return False
    if not isinstance(vertices, np.ndarray):
        return False
    if vertices.ndim != 2:
        return False
    if vertices.shape[1] != 3:
        return False
    if vertices.size == 0:
        return False
    return True

def validate_faces(faces: np.ndarray, vertex_count: int) -> bool:
    """Validate face array."""
    if faces is None:
        return False
    if not isinstance(faces, np.ndarray):
        return False
    if faces.ndim != 2:
        return False
    if faces.shape[1] != 3:
        return False
    if faces.size == 0:
        return True  # Empty faces array is valid
    
    # Check that all face indices are valid
    if np.any(faces < 0) or np.any(faces >= vertex_count):
        return False
    
    return True

@dataclass
class MeshData:
    """Container for mesh geometry and attributes."""
    vertices: np.ndarray  # Nx3 array of vertex positions
    faces: np.ndarray     # Mx3 array of vertex indices
    normals: Optional[np.ndarray] = None  # Nx3 array of vertex normals
    uvs: Optional[np.ndarray] = None      # Nx2 array of texture coordinates
    colors: Optional[np.ndarray] = None   # Nx3 array of vertex colors
    materials: Optional[Dict[str, Any]] = None  # Material properties
    
    def __post_init__(self):
        """Validate mesh data on creation."""
        # Convert to numpy arrays if needed
        if not isinstance(self.vertices, np.ndarray):
            self.vertices = np.array(self.vertices, dtype=np.float32)
        if not isinstance(self.faces, np.ndarray):
            self.faces = np.array(self.faces, dtype=np.int32)
            
        # Validate required data
        if not validate_vertices(self.vertices):
            raise MeshValidationError("Invalid vertex data")
        if not validate_faces(self.faces, len(self.vertices)):
            raise MeshValidationError("Invalid face data")
            
        # Validate optional data
        if self.normals is not None:
            if not isinstance(self.normals, np.ndarray):
                self.normals = np.array(self.normals, dtype=np.float32)
            if self.normals.shape != self.vertices.shape:
                raise MeshValidationError("Normal array shape doesn't match vertices")
                
        if self.uvs is not None:
            if not isinstance(self.uvs, np.ndarray):
                self.uvs = np.array(self.uvs, dtype=np.float32)
            if self.uvs.shape[0] != self.vertices.shape[0] or self.uvs.shape[1] != 2:
                raise MeshValidationError("UV array shape invalid")
                
        if self.colors is not None:
            if not isinstance(self.colors, np.ndarray):
                self.colors = np.array(self.colors, dtype=np.float32)
            if self.colors.shape != self.vertices.shape:
                raise MeshValidationError("Color array shape doesn't match vertices")
    
    @property
    def vertex_count(self) -> int:
        """Get number of vertices."""
        return len(self.vertices)
    
    @property
    def face_count(self) -> int:
        """Get number of faces."""
        return len(self.faces)
    
    @property
    def triangle_count(self) -> int:
        """Get number of triangles (alias for face_count)."""
        return self.face_count
    
    def ensure_normals(self, force_recalculate: bool = False) -> None:
        """Ensure vertex normals are calculated."""
        if self.normals is None or force_recalculate:
            self.normals = self.calculate_vertex_normals()
    
    def calculate_vertex_normals(self) -> np.ndarray:
        """Calculate vertex normals from face geometry."""
        # Initialize vertex normals
        vertex_normals = np.zeros_like(self.vertices)
        
        # Calculate face normals and accumulate at vertices
        for face in self.faces:
            v0, v1, v2 = self.vertices[face[0]], self.vertices[face[1]], self.vertices[face[2]]
            
            # Calculate face normal
            edge1 = v1 - v0
            edge2 = v2 - v0
            face_normal = np.cross(edge1, edge2)
            
            # Normalize face normal
            norm = np.linalg.norm(face_normal)
            if norm > 1e-10:
                face_normal = face_normal / norm
            
            # Accumulate at vertices
            for vertex_idx in face:
                vertex_normals[vertex_idx] += face_normal
        
        # Normalize vertex normals
        norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0  # Avoid division by zero
        vertex_normals = vertex_normals / norms
        
        return vertex_normals
    
    def generate_uvs(self, method: str = "planar") -> None:
        """Generate UV coordinates for the mesh."""
        if method == "planar":
            # Simple planar projection
            min_vals = np.min(self.vertices, axis=0)
            max_vals = np.max(self.vertices, axis=0)
            
            # Use X and Y coordinates for UV
            u = (self.vertices[:, 0] - min_vals[0]) / (max_vals[0] - min_vals[0])
            v = (self.vertices[:, 1] - min_vals[1]) / (max_vals[1] - min_vals[1])
            
            self.uvs = np.column_stack([u, v])
        else:
            raise ValueError(f"Unsupported UV generation method: {method}")
    
    def optimize(self) -> None:
        """Optimize mesh by removing duplicate vertices and unused vertices."""
        # Find duplicate vertices
        unique_vertices, inverse_indices = np.unique(
            self.vertices, axis=0, return_inverse=True
        )
        
        # If no duplicates found, nothing to do
        if len(unique_vertices) == len(self.vertices):
            return
        
        # Update face indices
        old_to_new = {old_idx: new_idx for old_idx, new_idx in enumerate(inverse_indices)}
        new_faces = np.array([[old_to_new[face[0]], old_to_new[face[1]], old_to_new[face[2]]] 
                             for face in self.faces])
        
        # Update mesh data
        self.vertices = unique_vertices
        self.faces = new_faces
        
        # Update optional arrays
        if self.normals is not None:
            self.normals = self.normals[inverse_indices]
        if self.uvs is not None:
            self.uvs = self.uvs[inverse_indices]
        if self.colors is not None:
            self.colors = self.colors[inverse_indices]
        
        logger.info(f"Optimized mesh: removed {len(self.vertices) - len(unique_vertices)} duplicate vertices")
    
    def get_bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get mesh bounding box."""
        min_bounds = np.min(self.vertices, axis=0)
        max_bounds = np.max(self.vertices, axis=0)
        return min_bounds, max_bounds
    
    def transform(self, matrix: np.ndarray) -> None:
        """Apply transformation matrix to vertices."""
        if matrix.shape != (4, 4):
            raise ValueError("Transformation matrix must be 4x4")
        
        # Convert vertices to homogeneous coordinates
        vertices_homogeneous = np.ones((len(self.vertices), 4))
        vertices_homogeneous[:, :3] = self.vertices
        
        # Apply transformation
        transformed = vertices_homogeneous @ matrix.T
        self.vertices = transformed[:, :3]
        
        # Transform normals if present (only rotation part)
        if self.normals is not None:
            rotation_matrix = matrix[:3, :3]
            self.normals = self.normals @ rotation_matrix.T
            # Renormalize
            norms = np.linalg.norm(self.normals, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
            self.normals = self.normals / norms
    
    def scale(self, scale_factor: Union[float, Tuple[float, float, float]]) -> None:
        """Scale the mesh vertices."""
        if isinstance(scale_factor, (int, float)):
            scale_factor = (scale_factor, scale_factor, scale_factor)
        
        scale_matrix = np.array([
            [scale_factor[0], 0, 0],
            [0, scale_factor[1], 0],
            [0, 0, scale_factor[2]]
        ])
        
        self.vertices = self.vertices @ scale_matrix.T
    
    def translate(self, offset: Tuple[float, float, float]) -> None:
        """Translate the mesh vertices."""
        self.vertices += np.array(offset)
    
    def copy(self) -> 'MeshData':
        """Create a copy of the mesh."""
        return MeshData(
            vertices=self.vertices.copy(),
            faces=self.faces.copy(),
            normals=self.normals.copy() if self.normals is not None else None,
            uvs=self.uvs.copy() if self.uvs is not None else None,
            colors=self.colors.copy() if self.colors is not None else None,
            materials=self.materials.copy() if self.materials is not None else None
        )
    
    def validate(self) -> bool:
        """Validate mesh integrity."""
        try:
            # Basic validation
            if not validate_vertices(self.vertices):
                return False
            if not validate_faces(self.faces, len(self.vertices)):
                return False
            
            # Check for degenerate faces
            for face in self.faces:
                if len(set(face)) != 3:  # Degenerate face
                    return False
            
            # Check optional arrays
            if self.normals is not None and self.normals.shape != self.vertices.shape:
                return False
            if self.uvs is not None and (self.uvs.shape[0] != len(self.vertices) or self.uvs.shape[1] != 2):
                return False
            if self.colors is not None and self.colors.shape != self.vertices.shape:
                return False
            
            return True
        except Exception:
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get mesh statistics."""
        min_bounds, max_bounds = self.get_bounding_box()
        size = max_bounds - min_bounds
        
        stats = {
            "vertex_count": self.vertex_count,
            "face_count": self.face_count,
            "triangle_count": self.triangle_count,
            "bounding_box": {
                "min": min_bounds.tolist(),
                "max": max_bounds.tolist(),
                "size": size.tolist()
            },
            "has_normals": self.normals is not None,
            "has_uvs": self.uvs is not None,
            "has_colors": self.colors is not None,
            "has_materials": self.materials is not None
        }
        
        return stats

def create_mesh_from_heightmap(
    height_map: np.ndarray,
    x_offset: float = 0,
    y_offset: float = 0,
    x_length: float = 1,
    y_length: float = 1,
    z_scale: float = 1,
    base_height: float = 0.0,
    generate_normals: bool = True,
    generate_uvs: bool = True
) -> MeshData:
    """
    Create a mesh from a heightmap.
    
    Args:
        height_map: 2D numpy array of height values
        x_offset, y_offset: Offset for vertex positions
        x_length, y_length: Physical dimensions of the mesh
        z_scale: Scale factor for height values
        base_height: Height of solid base below model
        generate_normals: Whether to calculate vertex normals
        generate_uvs: Whether to generate UV coordinates
        
    Returns:
        MeshData object containing the generated mesh
    """
    try:
        rows, cols = height_map.shape
        
        # Calculate vertex spacing
        dx = x_length / (cols - 1) if cols > 1 else 0
        dy = y_length / (rows - 1) if rows > 1 else 0
        
        # Create vertices
        vertices = []
        for i in range(rows):
            for j in range(cols):
                x = j * dx + x_offset
                y = i * dy + y_offset
                z = float(height_map[i, j]) * z_scale
                vertices.append([x, y, z])
        
        vertices = np.array(vertices, dtype=np.float32)
        
        # Create faces (triangles)
        faces = []
        for i in range(rows - 1):
            for j in range(cols - 1):
                # Get vertex indices for this quad
                v0 = i * cols + j
                v1 = v0 + 1
                v2 = v0 + cols
                v3 = v2 + 1
                
                # Create two triangles with proper winding
                faces.append([v0, v1, v2])  # First triangle
                faces.append([v1, v3, v2])  # Second triangle
        
        faces = np.array(faces, dtype=np.int32)
        
        # Create mesh
        mesh = MeshData(vertices=vertices, faces=faces)
        
        # Generate normals if requested
        if generate_normals:
            mesh.ensure_normals()
        
        # Generate UVs if requested
        if generate_uvs:
            mesh.generate_uvs()
        
        # Add base if requested
        if base_height > 0:
            mesh = add_base_to_mesh(mesh, base_height)
        
        return mesh
        
    except Exception as e:
        logger.error(f"Error creating mesh from heightmap: {e}")
        raise MeshOperationError(f"Failed to create mesh: {e}")

def add_base_to_mesh(mesh: MeshData, base_height: float) -> MeshData:
    """Add a solid base to the mesh."""
    try:
        # Get bounding box
        min_bounds, max_bounds = mesh.get_bounding_box()
        
        # Create base vertices (duplicate bottom vertices and offset down)
        base_vertices = mesh.vertices.copy()
        base_vertices[:, 2] = min_bounds[2] - base_height
        
        # Combine all vertices
        all_vertices = np.vstack([mesh.vertices, base_vertices])
        
        # Create side faces connecting original and base vertices
        num_original_vertices = len(mesh.vertices)
        side_faces = []
        
        # For each face in the original mesh, check if it's on the boundary
        boundary_edges = set()
        edge_count = {}
        
        # Count edge occurrences
        for face in mesh.faces:
            edges = [
                tuple(sorted([face[0], face[1]])),
                tuple(sorted([face[1], face[2]])),
                tuple(sorted([face[2], face[0]]))
            ]
            for edge in edges:
                edge_count[edge] = edge_count.get(edge, 0) + 1
        
        # Find boundary edges (appear only once)
        for edge, count in edge_count.items():
            if count == 1:
                boundary_edges.add(edge)
        
        # Create side faces for boundary edges
        for edge in boundary_edges:
            v1, v2 = edge
            base_v1 = v1 + num_original_vertices
            base_v2 = v2 + num_original_vertices
            
            # Create two triangles for the side face
            side_faces.extend([
                [v1, base_v1, v2],
                [base_v1, base_v2, v2]
            ])
        
        # Create bottom faces (tessellate the base)
        bottom_faces = []
        for face in mesh.faces:
            # Create corresponding face on the bottom with reversed winding
            bottom_face = [
                face[0] + num_original_vertices,
                face[2] + num_original_vertices,
                face[1] + num_original_vertices
            ]
            bottom_faces.append(bottom_face)
        
        # Combine all faces
        all_faces = np.vstack([
            mesh.faces,
            np.array(side_faces, dtype=np.int32),
            np.array(bottom_faces, dtype=np.int32)
        ])
        
        # Create new mesh with base
        new_mesh = MeshData(vertices=all_vertices, faces=all_faces)
        
        # Copy other attributes if present
        if mesh.uvs is not None:
            # Duplicate UVs for base vertices
            new_uvs = np.vstack([mesh.uvs, mesh.uvs])
            new_mesh.uvs = new_uvs
        
        if mesh.colors is not None:
            # Duplicate colors for base vertices
            new_colors = np.vstack([mesh.colors, mesh.colors])
            new_mesh.colors = new_colors
        
        # Recalculate normals
        new_mesh.ensure_normals(force_recalculate=True)
        
        return new_mesh
        
    except Exception as e:
        logger.error(f"Error adding base to mesh: {e}")
        raise MeshOperationError(f"Failed to add base: {e}")