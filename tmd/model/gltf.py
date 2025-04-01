"""
glTF exporter implementation for TMD.

This module provides the GLTFExporter class and related functions for converting
height maps to glTF and GLB files, which are widely used for 3D model exchange 
with web and AR/VR applications.
"""

import os
import json
import base64
import struct
import numpy as np
import logging
from typing import Optional, Dict, List, Tuple, Any, Union, Callable

from .base import ModelExporter, create_mesh_from_heightmap
from .mesh_utils import (
    calculate_vertex_normals,
    validate_heightmap,
    ensure_directory_exists,
    generate_uv_coordinates,
    optimize_mesh
)

# Set up logging
logger = logging.getLogger(__name__)


class GLTFExporter(ModelExporter):
    """Exporter for glTF (GL Transmission Format) and GLB (Binary glTF)."""
    
    @classmethod
    def get_extension(cls) -> str:
        """Get file extension."""
        return "gltf"
    
    @classmethod
    def get_format_name(cls) -> str:
        """Get format name."""
        return "GL Transmission Format (glTF)"
    
    @classmethod
    def supports_binary(cls) -> bool:
        """Check if binary format is supported."""
        return True  # Supports both JSON and binary formats
    
    @classmethod
    def export(cls, 
               height_map: np.ndarray, 
               filename: str, 
               x_offset: float = 0.0,
               y_offset: float = 0.0,
               x_length: float = 1.0,
               y_length: float = 1.0,
               z_scale: float = 1.0,
               base_height: float = 0.0,
               binary: bool = False,
               embed_textures: bool = True,
               add_texture: bool = False,
               optimize: bool = True,
               **kwargs) -> Optional[str]:
        """
        Export a heightmap to glTF/GLB format.
        
        Args:
            height_map: 2D numpy array of height values
            filename: Output filename
            x_offset: X-axis offset for the model
            y_offset: Y-axis offset for the model
            x_length: Physical length in X direction
            y_length: Physical length in Y direction
            z_scale: Scale factor for Z-axis values
            base_height: Height of solid base to add below the model
            binary: Whether to use binary format (GLB instead of glTF)
            embed_textures: Whether to embed textures in the file
            add_texture: Whether to add a texture based on height values
            optimize: Whether to optimize the mesh by merging vertices
            **kwargs: Additional parameters
            
        Returns:
            Path to the created file if successful, None otherwise
        """
        # Determine extension based on binary flag
        ext = 'glb' if binary else 'gltf'
        
        # Set the appropriate texture format options
        texture_format = kwargs.get('texture_format', 'terrain')
        texture_resolution = kwargs.get('texture_resolution', None)
        
        return convert_heightmap_to_gltf(
            height_map=height_map,
            filename=filename,
            x_offset=x_offset,
            y_offset=y_offset,
            x_length=x_length,
            y_length=y_length,
            z_scale=z_scale,
            base_height=base_height,
            generate_binary=binary,
            embed_textures=embed_textures,
            add_texture=add_texture,
            texture_format=texture_format,
            texture_resolution=texture_resolution,
            optimize=optimize,
            **kwargs
        )


def convert_heightmap_to_gltf(
    height_map: np.ndarray,
    filename: str = "output.gltf",
    x_offset: float = 0,
    y_offset: float = 0,
    x_length: float = 1,
    y_length: float = 1,
    z_scale: float = 1,
    base_height: float = 0.0,
    generate_binary: bool = False,
    embed_textures: bool = True,
    add_texture: bool = False,
    texture_format: str = 'terrain',
    texture_resolution: Optional[Tuple[int, int]] = None,
    optimize: bool = True,
    **kwargs
) -> Optional[str]:
    """
    Convert a height map to glTF/GLB format.
    
    Args:
        height_map: 2D numpy array of height values
        filename: Output filename (should end with .gltf or .glb)
        x_offset: X-axis offset for the model
        y_offset: Y-axis offset for the model
        x_length: Physical length in X direction
        y_length: Physical length in Y direction
        z_scale: Scale factor for Z-axis values
        base_height: Height of solid base to add below the model
        generate_binary: Whether to generate GLB (binary glTF)
        embed_textures: Whether to embed textures in the file
        add_texture: Whether to add a texture based on height values
        texture_format: Colormap name for texture generation
        texture_resolution: Optional (width, height) for texture resolution
        optimize: Whether to optimize the mesh by merging vertices
        **kwargs: Additional options
        
    Returns:
        Path to the created file or None if failed
    """
    # Validate input
    if not validate_heightmap(height_map):
        logger.error("Invalid height map: empty, None, or too small")
        return None

    # Ensure filename has correct extension
    if generate_binary:
        if not filename.lower().endswith('.glb'):
            filename = f"{os.path.splitext(filename)[0]}.glb"
    else:
        if not filename.lower().endswith('.gltf'):
            filename = f"{os.path.splitext(filename)[0]}.gltf"
            
    # Ensure output directory exists
    if not ensure_directory_exists(filename):
        logger.error(f"Failed to create directory for {filename}")
        return None

    try:
        # Create mesh from heightmap
        logger.info(f"Creating mesh from heightmap with dimensions: {height_map.shape}")
        vertices, faces = create_mesh_from_heightmap(
            height_map, 
            x_offset, 
            y_offset, 
            x_length, 
            y_length, 
            z_scale, 
            base_height
        )
        
        if not vertices or not faces:
            logger.error("Failed to generate mesh from heightmap")
            return None
            
        # Convert to numpy arrays for easier processing
        vertices_array = np.array(vertices, dtype=np.float32)
        faces_array = np.array(faces, dtype=np.uint32)
        
        # Optimize mesh if requested
        if optimize:
            logger.info("Optimizing mesh by merging vertices")
            vertices_array, faces_array = optimize_mesh(vertices_array, faces_array)
            logger.info(f"Optimized mesh: {len(vertices_array)} vertices, {len(faces_array)} faces")
        
        # Calculate vertex normals
        normals = calculate_vertex_normals(vertices_array, faces_array)
        
        # Generate texture coordinates (use spherical mapping if terrain is not flat)
        height_range = np.max(height_map) - np.min(height_map)
        if height_range > 0.1 * np.mean(height_map.shape) and "uv_method" not in kwargs:
            # For terrain with significant height variation, spherical might look better
            logger.info("Using cylindrical UV mapping for non-flat terrain")
            uvs = generate_uv_coordinates(vertices_array, method='cylindrical')
        else:
            # Use specified or default planar mapping
            uv_method = kwargs.get("uv_method", "planar")
            logger.info(f"Using {uv_method} UV mapping")
            uvs = generate_uv_coordinates(vertices_array, method=uv_method)
        
        # Create glTF structure
        gltf = _create_gltf_structure(
            vertices=vertices_array,
            faces=faces_array,
            normals=normals,
            uvs=uvs,
            height_map=height_map,
            generate_binary=generate_binary,
            embed_textures=embed_textures,
            add_texture=add_texture,
            texture_format=texture_format,
            texture_resolution=texture_resolution
        )
        
        # Store output filename for potential external textures
        gltf["output_filename"] = filename
        
        # Write to file
        if generate_binary:
            # Binary GLB format
            _write_binary_glb(gltf, filename)
        else:
            # JSON-based glTF format
            _write_json_gltf(gltf, filename)
            
        logger.info(f"Exported {'GLB' if generate_binary else 'glTF'} file to {filename}")
        return filename
        
    except Exception as e:
        logger.error(f"Error exporting to glTF: {e}")
        import traceback
        traceback.print_exc()
        return None


def _create_gltf_structure(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    uvs: np.ndarray,
    height_map: np.ndarray,
    generate_binary: bool = False,
    embed_textures: bool = True,
    add_texture: bool = False,
    texture_format: str = 'terrain',
    texture_resolution: Optional[Tuple[int, int]] = None
) -> Dict:
    """
    Create the glTF JSON structure with buffers.
    
    Args:
        vertices: Vertex positions (Nx3 float array)
        faces: Triangular faces (Mx3 uint32 array)
        normals: Vertex normals (Nx3 float array)
        uvs: Texture coordinates (Nx2 float array)
        height_map: Original height map for texture generation
        generate_binary: Whether to create binary GLB format
        embed_textures: Whether to embed textures in the file
        add_texture: Whether to add a texture based on height values
        texture_format: Colormap name for texture generation
        texture_resolution: Optional (width, height) for texture resolution
        
    Returns:
        glTF structure as a dictionary
    """
    # 1. Create buffer views for vertex positions, normals, UVs, and indices
    buffer_data = bytearray()
    buffer_views = []
    accessors = []
    
    # Helper function to add buffer view
    def add_buffer_view(data, component_type, type_str, count, normalized=False):
        buffer_view = {
            "buffer": 0,
            "byteOffset": len(buffer_data),
            "byteLength": len(data),
            "target": 34962  # ARRAY_BUFFER
        }
        
        if type_str == "SCALAR" and component_type == 5125:  # UNSIGNED_INT for indices
            buffer_view["target"] = 34963  # ELEMENT_ARRAY_BUFFER
            
        buffer_views.append(buffer_view)
        
        # Add accessor
        accessor = {
            "bufferView": len(buffer_views) - 1,
            "componentType": component_type,
            "count": count,
            "type": type_str,
            "normalized": normalized
        }
        
        # Calculate min/max values
        if type_str == "VEC3" and data:
            # Reshape to Nx3 array for calculating min/max
            reshaped = np.frombuffer(data, dtype=np.float32).reshape(-1, 3)
            accessor["min"] = reshaped.min(axis=0).tolist()
            accessor["max"] = reshaped.max(axis=0).tolist()
        elif type_str == "VEC2" and data:
            # Reshape to Nx2 array for calculating min/max
            reshaped = np.frombuffer(data, dtype=np.float32).reshape(-1, 2)
            accessor["min"] = reshaped.min(axis=0).tolist()
            accessor["max"] = reshaped.max(axis=0).tolist()
            
        accessors.append(accessor)
        
        # Extend buffer data
        buffer_data.extend(data)
        
        # Add padding to ensure 4-byte alignment
        while len(buffer_data) % 4 != 0:
            buffer_data.append(0)
            
        return len(accessors) - 1
    
    # Vertex positions - FLOAT (5126), VEC3
    position_accessor = add_buffer_view(
        vertices.astype(np.float32).tobytes(),
        5126,  # GL.FLOAT
        "VEC3",
        len(vertices)
    )
    
    # Vertex normals - FLOAT (5126), VEC3
    normal_accessor = add_buffer_view(
        normals.astype(np.float32).tobytes(),
        5126,  # GL.FLOAT
        "VEC3",
        len(normals)
    )
    
    # Texture coordinates - FLOAT (5126), VEC2
    texcoord_accessor = add_buffer_view(
        uvs.astype(np.float32).tobytes(),
        5126,  # GL.FLOAT
        "VEC2",
        len(uvs)
    )
    
    # Indices - UNSIGNED_INT (5125), SCALAR
    index_accessor = add_buffer_view(
        faces.flatten().astype(np.uint32).tobytes(),
        5125,  # GL.UNSIGNED_INT
        "SCALAR",
        len(faces.flatten())
    )
    
    # 2. Create the base glTF structure
    gltf = {
        "asset": {
            "version": "2.0",
            "generator": "TMD glTF Exporter",
            "copyright": "Copyright © TMD"
        },
        "scene": 0,
        "scenes": [
            {"nodes": [0]}
        ],
        "nodes": [
            {
                "mesh": 0,
                "name": "HeightMapTerrain"
            }
        ],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": position_accessor,
                            "NORMAL": normal_accessor,
                            "TEXCOORD_0": texcoord_accessor
                        },
                        "indices": index_accessor,
                        "mode": 4  # TRIANGLES
                    }
                ],
                "name": "terrain_mesh"
            }
        ],
        "buffers": [
            {"byteLength": len(buffer_data)}
        ],
        "bufferViews": buffer_views,
        "accessors": accessors
    }
    
    # 3. Add materials and textures if required
    if add_texture:
        # Generate a texture from the height map
        texture_data = _generate_texture_from_heightmap(
            height_map, 
            colormap=texture_format,
            resolution=texture_resolution
        )
        
        material_idx = _add_material(gltf, texture_data, generate_binary, embed_textures)
        
        # Set material on primitive
        gltf["meshes"][0]["primitives"][0]["material"] = material_idx
    
    # 4. Add buffer data
    if generate_binary:
        # For GLB, store buffer data to be written later
        gltf["binary_buffer"] = buffer_data
    else:
        # For glTF, encode as Base64
        gltf["buffers"][0]["uri"] = "data:application/octet-stream;base64," + base64.b64encode(buffer_data).decode('ascii')
    
    return gltf


def _add_material(
    gltf: Dict,
    texture_data: bytes,
    binary: bool = False,
    embed_textures: bool = True
) -> int:
    """
    Add material and texture to glTF structure.
    
    Args:
        gltf: glTF structure to add to
        texture_data: PNG texture data as bytes
        binary: Whether we're generating binary GLB
        embed_textures: Whether to embed textures
        
    Returns:
        Index of the added material
    """
    # Initialize arrays if they don't exist
    if "materials" not in gltf:
        gltf["materials"] = []
        
    if "textures" not in gltf:
        gltf["textures"] = []
        
    if "images" not in gltf:
        gltf["images"] = []
    
    # Add image
    image_idx = len(gltf["images"])
    image = {}
    
    if binary or embed_textures:
        # For GLB or embedded texture, store for buffer view
        buffer_view_idx = len(gltf.get("bufferViews", []))
        
        # Add buffer view for image
        gltf["bufferViews"].append({
            "buffer": 0,
            "byteOffset": len(gltf["binary_buffer"]) if binary else len(base64.b64decode(gltf["buffers"][0]["uri"].split(",")[1])),
            "byteLength": len(texture_data)
        })
        
        # Update buffer
        if binary:
            # Add texture to binary buffer with padding
            gltf["binary_buffer"].extend(texture_data)
            while len(gltf["binary_buffer"]) % 4 != 0:
                gltf["binary_buffer"].append(0)
                
            # Update total buffer length
            gltf["buffers"][0]["byteLength"] = len(gltf["binary_buffer"])
            
            # Reference buffer view in image
            image["bufferView"] = buffer_view_idx
            image["mimeType"] = "image/png"
        else:
            # For embedded textures in JSON, encode as base64
            uri = "data:image/png;base64," + base64.b64encode(texture_data).decode('ascii')
            image["uri"] = uri
    else:
        # For external texture
        image_filename = os.path.splitext(os.path.basename(gltf["output_filename"]))[0] + "_texture.png"
        image["uri"] = image_filename
        gltf["external_textures"] = gltf.get("external_textures", []) + [(image_filename, texture_data)]
    
    gltf["images"].append(image)
    
    # Add texture referencing the image
    texture_idx = len(gltf["textures"])
    gltf["textures"].append({
        "sampler": 0,
        "source": image_idx
    })
    
    # Add sampler if not present
    if "samplers" not in gltf:
        gltf["samplers"] = [{
            "magFilter": 9729,  # LINEAR
            "minFilter": 9987,  # LINEAR_MIPMAP_LINEAR
            "wrapS": 10497,     # REPEAT
            "wrapT": 10497      # REPEAT
        }]
    
    # Add material with PBR (Physically Based Rendering) properties
    material_idx = len(gltf["materials"])
    gltf["materials"].append({
        "pbrMetallicRoughness": {
            "baseColorTexture": {
                "index": texture_idx
            },
            "metallicFactor": 0.0,
            "roughnessFactor": 0.9
        },
        "name": "TerrainMaterial",
        "doubleSided": False,
        "alphaMode": "OPAQUE"
    })
    
    return material_idx


def _generate_texture_from_heightmap(
    height_map: np.ndarray, 
    colormap: str = 'terrain', 
    resolution: Optional[Tuple[int, int]] = None
) -> bytes:
    """
    Generate a PNG texture from a height map.
    
    Args:
        height_map: 2D height map array
        colormap: Name of the colormap to use
        resolution: Optional target resolution (width, height)
        
    Returns:
        PNG texture data as bytes
    """
    try:
        import cv2
        from matplotlib import cm
        
        # Normalize to 0-1
        h_min = np.min(height_map)
        h_max = np.max(height_map)
        
        if h_max > h_min:
            normalized_map = (height_map - h_min) / (h_max - h_min)
        else:
            normalized_map = np.zeros_like(height_map)
        
        # Resize if resolution is specified
        if resolution:
            normalized_map = cv2.resize(normalized_map, resolution, interpolation=cv2.INTER_CUBIC)
        
        # Apply colormap
        try:
            cmap = cm.get_cmap(colormap)
        except ValueError:
            logger.warning(f"Colormap '{colormap}' not found, using 'terrain' instead")
            cmap = cm.get_cmap('terrain')
            
        colored = cmap(normalized_map)
        
        # Convert to 8-bit RGB
        rgb_image = (colored[:, :, :3] * 255).astype(np.uint8)
        
        # Convert to BGR for OpenCV
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        
        # Encode as PNG
        _, png_data = cv2.imencode('.png', bgr_image)
        
        logger.info(f"Generated texture with colormap '{colormap}' and shape {bgr_image.shape}")
        return png_data.tobytes()
        
    except ImportError:
        # Fallback if matplotlib not available
        logger.warning("Matplotlib not available for texture generation. Trying OpenCV grayscale.")
        
        try:
            import cv2
            
            # Normalize to 0-255
            if np.max(height_map) > np.min(height_map):
                normalized = ((height_map - np.min(height_map)) / 
                           (np.max(height_map) - np.min(height_map)) * 255).astype(np.uint8)
            else:
                normalized = np.zeros_like(height_map, dtype=np.uint8)
            
            # Resize if resolution is specified
            if resolution:
                normalized = cv2.resize(normalized, resolution, interpolation=cv2.INTER_CUBIC)
            
            # Apply colormap (OpenCV has some basic colormaps)
            try:
                colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
            except Exception:
                # If colormap fails, convert to RGB by duplicating channels
                colored = np.stack([normalized, normalized, normalized], axis=2)
            
            # Encode as PNG
            _, png_data = cv2.imencode('.png', colored)
            
            logger.info(f"Generated grayscale texture with OpenCV, shape {colored.shape}")
            return png_data.tobytes()
            
        except ImportError:
            # If even OpenCV is not available, return empty texture
            logger.error("Cannot generate texture: Neither matplotlib nor OpenCV is available")
            return b''


def _write_json_gltf(gltf: Dict, filename: str) -> None:
    """
    Write glTF structure as JSON to file.
    
    Args:
        gltf: glTF structure
        filename: Output filename
    """
    # Remove binary buffer data before saving as JSON
    if "binary_buffer" in gltf:
        del gltf["binary_buffer"]
    
    # Write external textures if any
    if "external_textures" in gltf:
        for texture_filename, texture_data in gltf["external_textures"]:
            texture_path = os.path.join(os.path.dirname(filename), texture_filename)
            with open(texture_path, 'wb') as f:
                f.write(texture_data)
            logger.info(f"Wrote external texture to {texture_path}")
        del gltf["external_textures"]
    
    # Save output_filename in case it's needed elsewhere
    if "output_filename" in gltf:
        del gltf["output_filename"]
    
    # Write JSON to file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(gltf, f, indent=2)
    
    logger.info(f"Wrote JSON glTF to {filename}")


def _write_binary_glb(gltf: Dict, filename: str) -> None:
    """
    Write glTF data as binary GLB file.
    
    Args:
        gltf: glTF structure with binary buffer data
        filename: Output filename
    """
    # Extract binary buffer
    binary_buffer = gltf.get("binary_buffer", bytearray())
    if "binary_buffer" in gltf:
        del gltf["binary_buffer"]
    
    # Clean up any other temporary fields
    if "external_textures" in gltf:
        del gltf["external_textures"]
    if "output_filename" in gltf:
        del gltf["output_filename"]
    
    # Convert to JSON string
    json_data = json.dumps(gltf).encode('utf-8')
    
    # Pad JSON to 4-byte boundary
    json_pad = (4 - (len(json_data) % 4)) % 4
    json_data += b' ' * json_pad
    
    # Pad binary buffer to 4-byte boundary
    bin_pad = (4 - (len(binary_buffer) % 4)) % 4
    if bin_pad > 0:
        binary_buffer.extend(b'\x00' * bin_pad)
    
    # GLB header (magic, version, file length)
    length = 12 + 8 + len(json_data) + (8 + len(binary_buffer) if binary_buffer else 0)
    header = struct.pack('<III', 0x46546C67, 2, length)  # 'glTF' magic, version 2, file length
    
    # JSON chunk header (length, type)
    json_header = struct.pack('<II', len(json_data), 0x4E4F534A)  # 'JSON' type
    
    # Write GLB file
    with open(filename, 'wb') as f:
        # Write GLB header
        f.write(header)
        
        # Write JSON chunk
        f.write(json_header)
        f.write(json_data)
        
        # Write BIN chunk if it exists
        if binary_buffer:
            # Binary chunk header (length, type)
            bin_header = struct.pack('<II', len(binary_buffer), 0x004E4942)  # 'BIN\0' type
            f.write(bin_header)
            f.write(binary_buffer)
    
    logger.info(f"Wrote binary GLB to {filename}, total size: {length} bytes")


def convert_heightmap_to_glb(
    height_map: np.ndarray,
    filename: str = "output.glb",
    x_offset: float = 0,
    y_offset: float = 0,
    x_length: float = 1,
    y_length: float = 1,
    z_scale: float = 1,
    base_height: float = 0.0,
    **kwargs
) -> Optional[str]:
    """
    Convert a height map to GLB format (binary glTF).
    
    This is a convenience wrapper around convert_heightmap_to_gltf with generate_binary=True.
    
    Args:
        height_map: 2D numpy array of height values
        filename: Output filename
        x_offset, y_offset: Position offset
        x_length, y_length: Dimensions
        z_scale: Height scaling factor
        base_height: Height of solid base
        **kwargs: Additional options passed to convert_heightmap_to_gltf
        
    Returns:
        Path to the created file or None if failed
    """
    # Ensure .glb extension
    if not filename.lower().endswith('.glb'):
        filename = f"{os.path.splitext(filename)[0]}.glb"
    
    return convert_heightmap_to_gltf(
        height_map=height_map,
        filename=filename,
        x_offset=x_offset,
        y_offset=y_offset,
        x_length=x_length,
        y_length=y_length,
        z_scale=z_scale,
        base_height=base_height,
        generate_binary=True,
        **kwargs
    )