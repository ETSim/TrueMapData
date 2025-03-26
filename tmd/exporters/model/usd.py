"""
USD (Universal Scene Description) exporter for height maps.

This module provides functions for converting height maps to USD files,
which are used by many 3D applications and renderers.
"""

import os
import numpy as np
import logging
from typing import Optional, Dict, Any, Union, Tuple

from tmd.exporters.model.base import create_mesh_from_heightmap
from tmd.exporters.model.mesh_utils import (
    calculate_vertex_normals,
    validate_heightmap, 
    ensure_directory_exists
)

# Set up logging
logger = logging.getLogger(__name__)

# Try to import USD libraries
try:
    from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf, Vt
    USD_AVAILABLE = True
except ImportError:
    USD_AVAILABLE = False
    logger.warning("USD libraries not found. Install with: pip install usd-core")

def _create_texture_from_heightmap(height_map: np.ndarray, output_dir: str, name: str = "diffuse") -> str:
    """Generate a texture from a height map and save it to a file."""
    try:
        import cv2
        from matplotlib import cm
        
        # Normalize the height map to [0, 1]
        h_min = np.min(height_map)
        h_max = np.max(height_map)
        
        if h_max > h_min:
            normalized = (height_map - h_min) / (h_max - h_min)
        else:
            normalized = np.zeros_like(height_map)
            
        # Apply a colormap
        colored = cm.terrain(normalized)
        
        # Convert to 8-bit RGB
        rgb = (colored[:, :, :3] * 255).astype(np.uint8)
        
        # Ensure BGR for OpenCV
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        
        # Save to file
        texture_path = os.path.join(output_dir, f"{name}.png")
        cv2.imwrite(texture_path, bgr)
        
        return texture_path
    except ImportError:
        logger.warning("Could not create texture - OpenCV or matplotlib missing")
        return ""

def convert_heightmap_to_usd(
    height_map: np.ndarray,
    filename: str = "output.usda",
    z_scale: float = 1.0,
    base_height: float = 0.0,
    include_normals: bool = True,
    add_texture: bool = False,
    binary: bool = False,
    **kwargs
) -> Optional[str]:
    """
    Convert a height map to USD format.
    
    Args:
        height_map: 2D numpy array of height values
        filename: Output filename
        z_scale: Scale factor for height values
        base_height: Height of solid base to add below the model
        include_normals: Whether to include normal vectors
        add_texture: Whether to add a texture
        binary: Whether to output binary (usdc) instead of ASCII (usda)
        **kwargs: Additional parameters
        
    Returns:
        Path to the created file or None if failed
    """
    if not USD_AVAILABLE:
        logger.error("USD libraries not available. Install with: pip install usd-core")
        return None
        
    # Validate input
    if not validate_heightmap(height_map):
        logger.error("Invalid height map: empty, None, or too small")
        return None
    
    # Ensure correct file extension
    ext = ".usdc" if binary else ".usda"
    if not filename.lower().endswith(ext):
        filename = os.path.splitext(filename)[0] + ext
    
    # Ensure output directory exists
    output_dir = os.path.dirname(os.path.abspath(filename))
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Create the mesh
        vertices, faces = create_mesh_from_heightmap(
            height_map, 
            0, 0, 1, 1, 
            z_scale, 
            base_height
        )
        
        if not vertices or not faces:
            logger.error("Failed to generate mesh from heightmap")
            return None
            
        # Convert to numpy arrays
        vertices_array = np.array(vertices, dtype=np.float32)
        faces_array = np.array(faces, dtype=np.int32)
        
        # Create stage
        stage = Usd.Stage.CreateNew(filename)
        
        # Create mesh
        mesh_path = "/World/terrain"
        mesh = UsdGeom.Mesh.Define(stage, mesh_path)
        
        # Set vertices
        points = Vt.Vec3fArray([Gf.Vec3f(*v) for v in vertices_array])
        mesh.CreatePointsAttr(points)
        
        # Set face vertex counts and indices
        face_vertex_counts = Vt.IntArray([3] * len(faces_array))
        face_vertex_indices = Vt.IntArray(faces_array.flatten())
        
        mesh.CreateFaceVertexCountsAttr(face_vertex_counts)
        mesh.CreateFaceVertexIndicesAttr(face_vertex_indices)
        
        # Add normals if requested
        if include_normals:
            normals = calculate_vertex_normals(vertices_array, faces_array)
            normal_array = Vt.Vec3fArray([Gf.Vec3f(*n) for n in normals])
            mesh.CreateNormalsAttr(normal_array)
            mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        
        # Add texture if requested
        if add_texture:
            texture_path = _create_texture_from_heightmap(height_map, output_dir)
            if texture_path:
                # Create material
                material = UsdShade.Material.Define(stage, f"{mesh_path}/material")
                pbr = UsdShade.Shader.Define(stage, f"{mesh_path}/material/pbr")
                pbr.CreateIdAttr("UsdPreviewSurface")
                
                # Create texture shader
                texture = UsdShade.Shader.Define(stage, f"{mesh_path}/material/diffuseTexture")
                texture.CreateIdAttr("UsdUVTexture")
                texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture_path)
                
                # Connect texture to material
                pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
                    texture.ConnectableAPI(), "rgb")
                
                material.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")
                
                # Bind material to mesh
                UsdShade.MaterialBindingAPI(mesh).Bind(material)
        
        # Set up stage defaults
        root_prim = stage.GetPrimAtPath("/World")
        stage.SetDefaultPrim(root_prim)
        
        # Save stage
        stage.GetRootLayer().Save()
        logger.info(f"Exported USD file to {filename}")
        return filename
    except Exception as e:
        logger.error(f"Error creating USD file: {e}")
        import traceback
        traceback.print_exc()
        return None

def convert_heightmap_to_usdz(
    height_map: np.ndarray,
    filename: str = "output.usdz",
    z_scale: float = 1.0,
    base_height: float = 0.0,
    include_normals: bool = True,
    add_texture: bool = False,
    **kwargs
) -> Optional[str]:
    """
    Convert a height map to USDZ format.
    
    Args:
        height_map: 2D numpy array of height values
        filename: Output filename
        z_scale: Scale factor for height values
        base_height: Height of solid base to add below the model
        include_normals: Whether to include normal vectors
        add_texture: Whether to add a texture
        **kwargs: Additional parameters
        
    Returns:
        Path to the created file or None if failed
    """
    if not USD_AVAILABLE:
        logger.error("USD libraries not available. Install with: pip install usd-core")
        return None
    
    # First create a regular USD file
    if not filename.lower().endswith('.usdz'):
        filename = os.path.splitext(filename)[0] + '.usdz'
    
    # Create temporary USD file
    temp_usd_file = os.path.splitext(filename)[0] + '.usdc'
    
    # Create USD file
    result = convert_heightmap_to_usd(
        height_map=height_map,
        filename=temp_usd_file,
        z_scale=z_scale,
        base_height=base_height,
        include_normals=include_normals,
        add_texture=add_texture,
        binary=True,
        **kwargs
    )
    
    if not result:
        logger.error("Failed to create temporary USD file")
        return None
    
    try:
        # Pack as USDZ
        from pxr import UsdUtils
        
        # Create asset list for packaging
        asset_paths = [temp_usd_file]
        
        # Add textures if necessary
        texture_dir = os.path.dirname(temp_usd_file)
        if add_texture:
            texture_path = os.path.join(texture_dir, "diffuse.png")
            if os.path.exists(texture_path):
                asset_paths.append(texture_path)
        
        # Create USDZ package
        UsdUtils.CreateNewARKitUsdzPackage(
            temp_usd_file,  # Input USD file
            filename        # Output USDZ file
        )
        
        # Check if file was created
        if os.path.exists(filename):
            logger.info(f"Created USDZ file at {filename}")
            
            # Remove temporary USD file
            if os.path.exists(temp_usd_file):
                os.remove(temp_usd_file)
            
            return filename
        else:
            logger.error("USDZ file creation failed")
            return None
        
    except Exception as e:
        logger.error(f"Error creating USDZ file: {e}")
        import traceback
        traceback.print_exc()
        return None
