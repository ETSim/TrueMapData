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

from ..base import ModelExporter, ExportConfig, MeshData
from ..utils import validate_heightmap, ensure_directory_exists
from ..registry import register_exporter

# Set up logging
logger = logging.getLogger(__name__)


@register_exporter
class GLTFExporter(ModelExporter):
    """Exporter for glTF (GL Transmission Format)."""
    
    # Class attributes
    format_name = "GL Transmission Format (glTF)"
    file_extensions = ["gltf", "glb"]
    binary_supported = True
    
    @classmethod
    def export(cls, 
               height_map: np.ndarray, 
               filename: str, 
               config: ExportConfig) -> Optional[str]:
        """
        Export a heightmap to glTF/GLB format.
        
        Args:
            height_map: 2D numpy array of height values
            filename: Output filename
            config: Export configuration
            
        Returns:
            Path to the created file if successful, None otherwise
        """
        # Validate input
        if not validate_heightmap(height_map):
            logger.error("Invalid height map: empty, None, or too small")
            return None

        # Determine if we're using binary format (GLB)
        is_binary = config.binary
        if is_binary is None:
            # Auto-detect from extension
            is_binary = filename.lower().endswith('.glb')
        
        # Ensure filename has correct extension
        if is_binary:
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
            mesh = cls.create_mesh_from_heightmap(height_map, config)
            
            # Ensure we have normals and UVs
            mesh.ensure_normals()
            mesh.ensure_uvs(method=config.extra.get('uv_method', 'planar'))
            
            # Add texture if requested
            texture_data = None
            if config.texture:
                texture_data = _generate_texture_from_heightmap(
                    height_map,
                    colormap=config.color_map,
                    resolution=config.texture_resolution
                )
            
            # Create glTF structure
            gltf = _create_gltf_structure(
                mesh=mesh,
                texture_data=texture_data,
                generate_binary=is_binary,
                embed_textures=config.extra.get('embed_textures', True)
            )
            
            # Store output filename for potential external textures
            gltf["output_filename"] = filename
            
            # Write to file
            if is_binary:
                # Binary GLB format
                _write_binary_glb(gltf, filename)
            else:
                # JSON-based glTF format
                _write_json_gltf(gltf, filename)
                
            logger.info(f"Exported {'GLB' if is_binary else 'glTF'} file to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error exporting to glTF: {e}")
            import traceback
            traceback.print_exc()
            return None


def _create_gltf_structure(
    mesh: MeshData,
    texture_data: Optional[bytes] = None,
    generate_binary: bool = False,
    embed_textures: bool = True
) -> Dict:
    """
    Create the glTF JSON structure with buffers.
    
    Args:
        mesh: MeshData object containing the mesh
        texture_data: Optional PNG texture data as bytes
        generate_binary: Whether to create binary GLB format
        embed_textures: Whether to embed textures in the file
        
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
        mesh.vertices.astype(np.float32).tobytes(),
        5126,  # GL.FLOAT
        "VEC3",
        len(mesh.vertices)
    )
    
    # Vertex normals - FLOAT (5126), VEC3
    normal_accessor = add_buffer_view(
        mesh.normals.astype(np.float32).tobytes(),
        5126,  # GL.FLOAT
        "VEC3",
        len(mesh.normals)
    )
    
    # Texture coordinates - FLOAT (5126), VEC2
    texcoord_accessor = add_buffer_view(
        mesh.uvs.astype(np.float32).tobytes(),
        5126,  # GL.FLOAT
        "VEC2",
        len(mesh.uvs)
    )
    
    # Indices - UNSIGNED_INT (5125), SCALAR
    index_accessor = add_buffer_view(
        mesh.faces.flatten().astype(np.uint32).tobytes(),
        5125,  # GL.UNSIGNED_INT
        "SCALAR",
        len(mesh.faces.flatten())
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
    if texture_data:
        # Generate a texture from the height map
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
    from ..utils.heightmap import generate_heightmap_texture
    
    try:
        # Generate texture
        rgb_image = generate_heightmap_texture(height_map, colormap, resolution)
        
        # Encode as PNG
        import io
        from PIL import Image
        
        image = Image.fromarray(rgb_image)
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        
        return buffer.getvalue()
        
    except ImportError:
        # Fall back to grayscale if Pillow or matplotlib not available
        logger.warning("Error generating texture. PIL or matplotlib may be missing. Creating grayscale texture.")
        
        import io
        
        # Normalize to 0-255
        normalized = ((height_map - np.min(height_map)) / 
                     (np.max(height_map) - np.min(height_map)) * 255).astype(np.uint8)
        
        # Resize if needed
        if resolution:
            from ..utils.heightmap import resample_heightmap
            normalized = resample_heightmap(normalized, resolution, 'bilinear')
        
        # Convert to RGB
        rgb = np.stack([normalized, normalized, normalized], axis=2)
        
        # Create PNG
        try:
            from PIL import Image
            image = Image.fromarray(rgb)
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            return buffer.getvalue()
        except ImportError:
            logger.error("Could not generate texture: PIL not available")
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