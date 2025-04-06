"""
USD exporter implementation for TMD.

This module provides the USDExporter class and related functions for exporting
height maps to USD/USDZ format, which is commonly used for AR applications,
particularly on Apple platforms.
"""

import os
import tempfile
import shutil
import numpy as np
import logging
from typing import Optional, Dict, Any, Tuple, Union, List

from ..base import ModelExporter, ExportConfig, MeshData
from ..utils import validate_heightmap, ensure_directory_exists
from ..registry import register_exporter

# Set up logging
logger = logging.getLogger(__name__)


@register_exporter
class USDExporter(ModelExporter):
    """Exporter for Universal Scene Description (USD) format."""
    
    # Class attributes
    format_name = "USD"
    file_extensions = ["usd", "usda", "usdc", "usdz"]
    binary_supported = True  # USDC and USDZ are binary formats
    
    @classmethod
    def export(cls, 
               height_map: np.ndarray, 
               filename: str, 
               config: ExportConfig) -> Optional[str]:
        """
        Export a heightmap to USD format.
        
        Args:
            height_map: 2D numpy array of height values
            filename: Output filename
            config: Export configuration
            
        Returns:
            Path to the created file if successful, None otherwise
        """
        try:
            import usd
        except ImportError:
            logger.error("USD export requires the USD library. Install with: pip install usd-core")
            return None

        # Validate input
        if not validate_heightmap(height_map):
            logger.error("Invalid height map: empty, None, or too small")
            return None
        
        # Get USD-specific parameters from config
        binary = config.binary
        add_texture = config.texture
        texture_resolution = config.texture_resolution
        color_map = config.color_map
        
        # Determine format from extension if binary not specified
        if binary is None:
            ext = os.path.splitext(filename)[1].lower()
            binary = ext in ['.usdc', '.usdz']
        
        # Ensure filename has correct extension based on binary flag
        if binary:
            if filename.lower().endswith('.usda'):
                filename = filename[:-1] + 'c'  # .usda -> .usdc
            elif not any(filename.lower().endswith(ext) for ext in ['.usdc', '.usdz']):
                filename = f"{os.path.splitext(filename)[0]}.usdc"
        else:
            if filename.lower().endswith('.usdc'):
                filename = filename[:-1] + 'a'  # .usdc -> .usda
            elif not filename.lower().endswith('.usda'):
                filename = f"{os.path.splitext(filename)[0]}.usda"
                
        # Handle special case for USDZ (AR-optimized format)
        is_usdz = filename.lower().endswith('.usdz')
        
        # Ensure output directory exists
        if not ensure_directory_exists(filename):
            logger.error(f"Failed to create directory for {filename}")
            return None
        
        try:
            # Create mesh from heightmap
            mesh = cls.create_mesh_from_heightmap(height_map, config)
            
            # Ensure we have normals and UVs
            mesh.ensure_normals()
            mesh.ensure_uvs()
            
            # Export to USD format
            export_result = export_mesh_to_usd(
                mesh=mesh,
                filename=filename,
                binary=binary,
                add_texture=add_texture,
                texture_resolution=texture_resolution,
                color_map=color_map,
                height_map=height_map
            )
            
            # Special post-processing for USDZ format (AR packaging)
            if is_usdz and export_result:
                export_result = convert_to_usdz(export_result)
            
            return export_result
            
        except Exception as e:
            logger.error(f"Error exporting to USD: {e}")
            import traceback
            traceback.print_exc()
            return None


def export_mesh_to_usd(
    mesh: MeshData,
    filename: str,
    binary: bool = True,
    add_texture: bool = True,
    texture_resolution: Optional[Tuple[int, int]] = None,
    color_map: str = 'terrain',
    height_map: Optional[np.ndarray] = None
) -> Optional[str]:
    """
    Export a mesh to USD format.
    
    Args:
        mesh: MeshData object containing the mesh
        filename: Output filename
        binary: Whether to use binary format (USDC)
        add_texture: Whether to add a texture
        texture_resolution: Optional target resolution for texture
        color_map: Colormap name for texture generation
        height_map: Original heightmap for texture generation
        
    Returns:
        Path to the created file or None if failed
    """
    try:
        # Check if USD library is available
        from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf, Vt
    except ImportError:
        logger.error("Cannot export to USD: PixarUSD library not available")
        return None
    
    try:
        # Create a new USD stage
        if binary:
            stage = Usd.Stage.CreateNew(filename)
        else:
            stage = Usd.Stage.CreateNew(filename)
        
        # Set up units for proper scaling
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        
        # Create a mesh primitive at the root
        mesh_prim = UsdGeom.Mesh.Define(stage, '/terrainMesh')
        
        # Set mesh vertices
        point_attr = mesh_prim.CreatePointsAttr()
        points = [(v[0], v[2], v[1]) for v in mesh.vertices]  # Swap Y and Z for USD
        point_attr.Set(points)
        
        # Set mesh faces
        face_vertex_counts = [3] * len(mesh.faces)  # All triangles
        mesh_prim.CreateFaceVertexCountsAttr().Set(face_vertex_counts)
        
        # Set face vertex indices
        face_indices = []
        for face in mesh.faces:
            face_indices.extend(face)
        mesh_prim.CreateFaceVertexIndicesAttr().Set(face_indices)
        
        # Set normals if available
        if mesh.normals is not None:
            normal_attr = mesh_prim.CreateNormalsAttr()
            normals = [(n[0], n[2], n[1]) for n in mesh.normals]  # Swap Y and Z for USD
            normal_attr.Set(normals)
            mesh_prim.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        
        # Set UVs if available
        if mesh.uvs is not None:
            # We need to create a PrimvarsAPI for texture coordinates
            pv = UsdGeom.PrimvarsAPI(mesh_prim)
            
            # Create texture coordinate primvar
            texcoords = [(uv[0], 1.0 - uv[1], 0.0) for uv in mesh.uvs]  # Flip V coordinate
            texcoord_primvar = pv.CreatePrimvar("st", 
                                               Sdf.ValueTypeNames.TexCoord2fArray, 
                                               UsdGeom.Tokens.vertex)
            texcoord_primvar.Set(texcoords)
        
        # Add texture if requested
        if add_texture and height_map is not None:
            # Generate texture from heightmap
            texture_path = _generate_usd_texture(
                height_map=height_map,
                filename=os.path.splitext(filename)[0] + "_texture.png",
                color_map=color_map,
                resolution=texture_resolution
            )
            
            if texture_path:
                # Create material
                material = UsdShade.Material.Define(stage, '/terrainMaterial')
                
                # Create PBR shader
                pbr_shader = UsdShade.Shader.Define(stage, '/terrainMaterial/PBRShader')
                pbr_shader.CreateIdAttr("UsdPreviewSurface")
                
                # Create texture sampler
                tex_sampler = UsdShade.Shader.Define(stage, '/terrainMaterial/diffuseTexture')
                tex_sampler.CreateIdAttr('UsdUVTexture')
                
                # Set texture path
                tex_file = tex_sampler.CreateInput('file', Sdf.ValueTypeNames.Asset)
                tex_file.Set(texture_path)
                
                # Connect texture to PBR shader
                diffuse_input = pbr_shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f)
                tex_sampler.CreateOutput('rgb', Sdf.ValueTypeNames.Float3).Connect(diffuse_input)
                
                # Create material output
                material.CreateSurfaceOutput().ConnectToSource(pbr_shader, 'surface')
                
                # Bind material to mesh
                UsdShade.MaterialBindingAPI(mesh_prim).Bind(material)
        
        # Save the stage
        stage.GetRootLayer().Save()
        
        logger.info(f"Exported USD file to {filename}")
        return filename
    
    except Exception as e:
        logger.error(f"Error during USD export: {e}")
        import traceback
        traceback.print_exc()
        return None


def _generate_usd_texture(
    height_map: np.ndarray,
    filename: str,
    color_map: str = 'terrain',
    resolution: Optional[Tuple[int, int]] = None
) -> Optional[str]:
    """
    Generate a texture file for USD from heightmap data.
    
    Args:
        height_map: 2D heightmap array
        filename: Output texture filename
        color_map: Name of colormap to use
        resolution: Optional target resolution for texture
    
    Returns:
        Path to the created texture file or None if failed
    """
    try:
        from ..utils.heightmap import generate_heightmap_texture
        
        # Generate texture
        rgb_image = generate_heightmap_texture(height_map, color_map, resolution)
        
        # Save texture
        try:
            from PIL import Image
            image = Image.fromarray(rgb_image)
            image.save(filename)
            logger.info(f"Created texture at {filename}")
            return filename
        except ImportError:
            logger.warning("PIL not available, texture generation skipped")
            return None
        
    except Exception as e:
        logger.error(f"Error generating texture: {e}")
        return None


def convert_to_usdz(usd_filename: str) -> Optional[str]:
    """
    Convert a USD file to USDZ format (for AR applications).
    
    Args:
        usd_filename: Path to the USD file to convert
        
    Returns:
        Path to the created USDZ file or None if failed
    """
    try:
        # Check if USD library is available
        from pxr import UsdUtils
    except ImportError:
        logger.error("Cannot convert to USDZ: PixarUSD library not available")
        return None
    
    try:
        # Ensure the input file exists
        if not os.path.exists(usd_filename):
            logger.error(f"Input file not found: {usd_filename}")
            return None
        
        # Create output filename
        usdz_filename = os.path.splitext(usd_filename)[0] + ".usdz"
        
        # Create a temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Find assets that need to be included (like textures)
            asset_dir = os.path.dirname(usd_filename)
            
            # Create the USDZ file
            result = UsdUtils.CreateNewARKitUsdzPackage(
                usd_filename,  # Input USD file
                usdz_filename,  # Output USDZ file
                [asset_dir]    # Asset search paths
            )
            
            if result:
                logger.info(f"Created USDZ file at {usdz_filename}")
                return usdz_filename
            else:
                logger.error("Failed to create USDZ file")
                return None
                
    except Exception as e:
        logger.error(f"Error converting to USDZ: {e}")
        import traceback
        traceback.print_exc()
        return None