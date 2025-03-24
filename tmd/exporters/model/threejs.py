""".

Three.js exporter module for height maps.

This module provides functions for converting height maps to Three.js compatible formats
for web-based 3D visualization.
"""

import os
import json
import numpy as np
import logging
import zlib
import base64
import tempfile
from PIL import Image
from typing import Dict, Any, Optional, List, Tuple, Union

from .base import _create_mesh_from_heightmap, _calculate_vertex_normals
from .gltf import convert_heightmap_to_gltf

# Set up logging
logger = logging.getLogger(__name__)

def convert_heightmap_to_threejs(
    height_map,
    filename=None,
    z_scale=1.0,
    add_wireframe=False,
    include_texture=False,
    texture_map=None,
    compress=False,
    use_gltf=False,
    **kwargs
):
    """
    Convert a height map to Three.js JSON format.
    
    Args:
        height_map: 2D numpy array of height values
        filename: Output filename (should end with .json)
        z_scale: Scale factor for Z-axis values
        add_wireframe: Whether to add wireframe representation
        include_texture: Whether to include texture data
        texture_map: Optional texture map to use
        compress: Whether to compress the output
        use_gltf: Whether to use GLTF format instead of JSON
        **kwargs: Additional options for export
    
    Returns:
        Path to the created file or None if failed
    """
    try:
        # Process options
        kwargs['z_scale'] = z_scale
        kwargs['add_wireframe'] = add_wireframe
        kwargs['include_texture'] = include_texture
        kwargs['compress'] = compress
        
        # Use GLTF if requested
        if use_gltf:
            from .gltf import convert_heightmap_to_gltf
            # Adjust filename to .gltf if needed
            if filename and not filename.lower().endswith(('.gltf', '.glb')):
                filename = os.path.splitext(filename)[0] + '.gltf'
            return convert_heightmap_to_gltf(
                height_map=height_map,
                filename=filename,
                add_texture=include_texture,
                texture_map=texture_map,
                **kwargs
            )
        
        # Process texture
        if include_texture:
            if texture_map is None:
                # Generate a simple texture based on height
                from PIL import Image
                import numpy as np
                
                texture_filename = os.path.splitext(filename)[0] + "_texture.png"
                
                # Normalize height map to 0-255
                norm_height = (height_map - np.min(height_map)) / (np.max(height_map) - np.min(height_map) + 1e-8)
                img_data = (norm_height * 255).astype(np.uint8)
                
                # Create RGB texture
                rgb_img = np.stack([img_data, img_data, img_data], axis=2)
                
                # Save texture
                Image.fromarray(rgb_img).save(texture_filename)
                kwargs['texture_file'] = texture_filename
            else:
                # Use provided texture
                kwargs['texture_file'] = texture_map
        
        # Use the backend exporter
        from .backends import ThreeJSExporter
        exporter = ThreeJSExporter()
        return exporter.export(height_map, filename, **kwargs)
        
    except Exception as e:
        logger.error(f"Error exporting Three.js: {e}")
        import traceback
        traceback.print_exc()
        return None

def _create_threejs_json(
    vertices: List[List[float]],
    faces: List[List[int]],
    normals: List[List[float]],
    add_wireframe: bool = False,
    add_texture: bool = False,
    height_map: Optional[np.ndarray] = None,
    texture_resolution: Optional[Tuple[int, int]] = None
) -> Dict[str, Any]:
    """.

    Create a Three.js JSON structure from mesh data.
    
    Args:
        vertices: List of vertex positions [x, y, z]
        faces: List of face indices [a, b, c]
        normals: List of vertex normals [nx, ny, nz]
        add_wireframe: Whether to add a wireframe geometry
        add_texture: Whether to add a texture
        height_map: Original height map (needed for texture)
        texture_resolution: Optional resolution for texture
        
    Returns:
        Three.js JSON structure
    """
    # Flatten arrays for Three.js format
    vertices_flat = [coord for vertex in vertices for coord in vertex]
    faces_flat = [index for face in faces for index in face]
    normals_flat = [coord for normal in normals for coord in normal]
    
    # Generate UVs if texture is requested
    uvs = []
    if add_texture:
        assert height_map is not None, "Height map is required for texturing"
        uvs = _generate_uvs(vertices, height_map.shape)
    
    # Create geometries
    geometries = [{
        "uuid": "heightmap-mesh",
        "type": "BufferGeometry",
        "data": {
            "attributes": {
                "position": {
                    "itemSize": 3,
                    "type": "Float32Array",
                    "array": vertices_flat
                },
                "normal": {
                    "itemSize": 3,
                    "type": "Float32Array",
                    "array": normals_flat
                }
            },
            "index": {
                "type": "Uint32Array",
                "array": faces_flat
            }
        }
    }]
    
    # Add UVs if needed
    if uvs:
        geometries[0]["data"]["attributes"]["uv"] = {
            "itemSize": 2,
            "type": "Float32Array",
            "array": uvs
        }
    
    # Add wireframe geometry if requested
    if add_wireframe:
        # Extract edges from faces
        edges = set()
        for face in faces:
            # For each face, add its 3 edges
            if len(face) >= 3:  # Safety check
                edges.add(tuple(sorted([face[0], face[1]])))
                edges.add(tuple(sorted([face[1], face[2]])))
                edges.add(tuple(sorted([face[2], face[0]])))
        
        # Create wireframe geometry
        wire_vertices = []
        wire_indices = []
        for i, (v1, v2) in enumerate(edges):
            wire_vertices.extend(vertices[v1])
            wire_vertices.extend(vertices[v2])
            wire_indices.extend([i*2, i*2+1])
        
        geometries.append({
            "uuid": "wireframe",
            "type": "BufferGeometry",
            "data": {
                "attributes": {
                    "position": {
                        "itemSize": 3,
                        "type": "Float32Array",
                        "array": wire_vertices
                    }
                },
                "index": {
                    "type": "Uint32Array",
                    "array": wire_indices
                }
            }
        })
    
    # Create materials
    materials = [{
        "uuid": "heightmap-material",
        "type": "MeshPhongMaterial",
        "color": 0xcccccc,
        "specular": 0x111111,
        "shininess": 30,
        "side": 2  # DoubleSide
    }]
    
    # Add wireframe material if needed
    if add_wireframe:
        materials.append({
            "uuid": "wireframe-material",
            "type": "LineBasicMaterial",
            "color": 0x000000,
            "linewidth": 1
        })
    
    # Add texture if requested
    textures = []
    images = []
    if add_texture and height_map is not None:
        # Generate texture data
        texture_data, image_uuid = _generate_threejs_texture(height_map, texture_resolution)
        textures.append(texture_data)
        
        # Generate height map image
        # Convert height map to grayscale image
        normalized = (height_map - np.min(height_map)) / (np.max(height_map) - np.min(height_map))
        img_data = (normalized * 255).astype(np.uint8)
        
        # Resize if specified
        if texture_resolution:
            from PIL import Image
            # Convert to PIL Image for resizing
            pil_img = Image.fromarray(img_data, mode='L')
            # Use proper resize parameters - correction to fix TypeError
            pil_img = pil_img.resize(size=texture_resolution, resample=Image.Resampling.LANCZOS)
            # Convert back to numpy array
            img_data = np.array(pil_img)
        
        # Convert to base64 for embedding
        with tempfile.NamedTemporaryFile(suffix='.png') as tmp:
            Image.fromarray(img_data).save(tmp.name)
            with open(tmp.name, 'rb') as f:
                img_data_base64 = base64.b64encode(f.read()).decode('ascii')
        
        # Add image data
        images.append({
            "uuid": image_uuid,
            "url": f"data:image/png;base64,{img_data_base64}"
        })
        
        # Update material to use texture
        materials[0]["map"] = image_uuid
    
    # Create objects
    objects = {
        "type": "Group",
        "uuid": "heightmap-group",
        "children": [
            {
                "type": "Mesh",
                "uuid": "heightmap-object",
                "geometry": "heightmap-mesh",
                "material": "heightmap-material"
            }
        ]
    }
    
    # Add wireframe object if needed
    if add_wireframe:
        objects["children"].append({
            "type": "LineSegments",
            "uuid": "wireframe-object",
            "geometry": "wireframe",
            "material": "wireframe-material"
        })
    
    # Create final Three.js JSON structure
    threejs_json = {
        "metadata": {
            "version": 4.5,
            "type": "Object",
            "generator": "TMD Library"
        },
        "geometries": geometries,
        "materials": materials,
        "object": objects
    }
    
    # Add textures and images if present
    if textures:
        threejs_json["textures"] = textures
    if images:
        threejs_json["images"] = images
    
    return threejs_json

def _generate_uvs(vertices: List[List[float]], shape: Tuple[int, int]) -> List[float]:
    """.

    Generate UV coordinates for vertices.
    
    Args:
        vertices: List of vertex positions [x, y, z]
        shape: Dimensions of the height map (rows, cols)
        
    Returns:
        Flattened list of UV coordinates
    """
    rows, cols = shape
    uvs = []
    
    # Find min/max x and y coordinates
    min_x = min(v[0] for v in vertices)
    max_x = max(v[0] for v in vertices)
    min_y = min(v[1] for v in vertices)
    max_y = max(v[1] for v in vertices)
    
    x_range = max_x - min_x
    y_range = max_y - min_y
    
    # Generate UV coordinates for each vertex
    for vertex in vertices:
        u = (vertex[0] - min_x) / x_range if x_range > 0 else 0.5
        v = (vertex[1] - min_y) / y_range if y_range > 0 else 0.5
        uvs.extend([u, v])
    
    return uvs

def _generate_threejs_texture(
    height_map: np.ndarray, 
    resolution: Optional[Tuple[int, int]] = None
) -> Tuple[Dict[str, Any], str]:
    """.

    Generate Three.js texture data.
    
    Args:
        height_map: Height map to use for texture
        resolution: Optional (width, height) to resize the texture
        
    Returns:
        Tuple of (texture data dict, image UUID)
    """
    # Generate UUID for texture and image
    texture_uuid = "heightmap-texture"
    image_uuid = "heightmap-image"
    
    # Create texture data
    texture_data = {
        "uuid": texture_uuid,
        "name": "HeightMapTexture",
        "mapping": 300,  # UVMapping
        "repeat": [1, 1],
        "offset": [0, 0],
        "wrap": [1000, 1000],  # RepeatWrapping
        "format": 1023,  # RGBFormat
        "minFilter": 1008,  # LinearFilter
        "magFilter": 1006,  # LinearFilter
        "anisotropy": 1,
        "image": image_uuid
    }
    
    return texture_data, image_uuid
