"""
Mesh converter utilities for the TMD2Model application.

This module provides unified conversion functions to handle different 3D model formats
and reduce code duplication in the main application.
"""

import os
import logging
import time
import numpy as np
import struct
from typing import Optional, Dict, Any, Callable, Union, Tuple, List
from pathlib import Path

# Exporters for image formats
from ..exporters.image.image_io import load_image, ImageType
from ..exporters.image.bump_map import convert_heightmap_to_bump_map
from ..exporters.image.normal_map import export_normal_map
from ..exporters.image.ao_map import export_ambient_occlusion, convert_heightmap_to_ao_map
from ..exporters.image.displacement_map import export_displacement_map
from ..exporters.image.heightmap import convert_heightmap_to_heightmap
from ..exporters.image.hillshade import generate_hillshade
from ..exporters.image.material_set import generate_material_set

# Exporters for 3D model formats
from ..exporters.model import (
    convert_heightmap_to_stl,
    convert_heightmap_to_obj,
    convert_heightmap_to_gltf,
    convert_heightmap_to_glb,
    convert_heightmap_to_ply,
    convert_heightmap_to_usd,
    convert_heightmap_to_usdz
)
from ..exporters.model.adaptive_mesh import convert_heightmap_to_adaptive_mesh

# Set up logging
logger = logging.getLogger(__name__)

# Define mapping for export functions
FORMAT_EXPORTERS = {
    # Image formats
    "normal_map": {
        "function": export_normal_map,
        "params": {
            "output_path": "output_file", 
            "z_scale": "normal_map_z_scale"
        },
        "defaults": {
            "normalize": True
        }
    },
    "bump_map": {
        "function": convert_heightmap_to_bump_map,
        "params": {
            "filename": "output_file",
            "strength": "bump_map_strength",
            "blur_radius": "bump_map_blur"
        }
    },
    "ao_map": {
        "function": convert_heightmap_to_ao_map,
        "params": {
            "filename": "output_file", 
            "strength": "ao_strength",
            "samples": "ao_samples",
            "intensity": "ao_strength"
        }
    },
    "displacement_map": {
        "function": export_displacement_map,
        "params": {
            "output_file": "output_file",
            "bit_depth": "bit_depth"
        }
    },
    "heightmap": {
        "function": convert_heightmap_to_heightmap,
        "params": {
            "output_file": "output_file",
            "bit_depth": "bit_depth"
        },
        "defaults": {
            "normalize": True
        }
    },
    "hillshade": {
        "function": generate_hillshade,
        "params": {
            "output_file": "output_file",
            "azimuth": "hillshade_azimuth", 
            "altitude": "hillshade_altitude",
            "z_factor": "hillshade_z_factor"
        }
    },
    "material_set": {
        "function": generate_material_set,
        "params": {
            "output_dir": "output_file",
            "base_name": "material_base_name",
            "z_scale": "normal_map_z_scale",
            "ao_strength": "ao_strength",
            "ao_samples": "ao_samples"
        }
    },
    # 3D model formats
    "stl": {
        "function": convert_heightmap_to_stl,
        "params": {
            "filename": "output_file",
            "x_offset": "x_offset",
            "y_offset": "y_offset",
            "x_length": "x_length",
            "y_length": "y_length",
            "z_scale": "z_scale",
            "base_height": "base_height",
            "ascii_format": "ascii_format"
        }
    },
    "obj": {
        "function": convert_heightmap_to_obj,
        "params": {
            "filename": "output_file",
            "x_offset": "x_offset",
            "y_offset": "y_offset",
            "x_length": "x_length",
            "y_length": "y_length",
            "z_scale": "z_scale",
            "base_height": "base_height",
            "include_materials": "include_materials"
        }
    },
    "ply": {
        "function": convert_heightmap_to_ply,
        "params": {
            "filename": "output_file",
            "x_offset": "x_offset",
            "y_offset": "y_offset",
            "x_length": "x_length",
            "y_length": "y_length",
            "z_scale": "z_scale",
            "base_height": "base_height",
            "binary": "binary",
            "add_color": "add_color"
        }
    },
    "gltf": {
        "function": convert_heightmap_to_gltf,
        "params": {
            "filename": "output_file",
            "x_offset": "x_offset",
            "y_offset": "y_offset",
            "x_length": "x_length", 
            "y_length": "y_length",
            "z_scale": "z_scale",
            "base_height": "base_height",
            "add_texture": "add_texture"
        },
        "defaults": {
            "generate_binary": False
        }
    },
    "glb": {
        "function": convert_heightmap_to_glb,
        "params": {
            "filename": "output_file",
            "x_offset": "x_offset",
            "y_offset": "y_offset",
            "x_length": "x_length",
            "y_length": "y_length",
            "z_scale": "z_scale",
            "base_height": "base_height",
            "add_texture": "add_texture"
        }
    },
    "usd": {
        "function": convert_heightmap_to_usd,
        "params": {
            "filename": "output_file",
            "z_scale": "z_scale",
            "base_height": "base_height",
            "add_texture": "add_texture"
        }
    },
    "usdz": {
        "function": convert_heightmap_to_usdz,
        "params": {
            "filename": "output_file",
            "z_scale": "z_scale",
            "base_height": "base_height",
            "add_texture": "add_texture"
        }
    }
}


def convert_heightmap(
    height_map: np.ndarray,
    output_file: str,
    format_type: str,
    **kwargs
) -> Optional[Union[str, Tuple]]:
    """
    Convert a heightmap to the specified format using the appropriate exporter.
    
    Args:
        height_map: The heightmap to convert
        output_file: Path to save the output file
        format_type: Output format type (stl, obj, ply, gltf, etc.)
        **kwargs: Additional parameters for the exporter
        
    Returns:
        Path to the created file or None if failed
    """
    # Progress callback for formats that support it
    progress_callback = kwargs.pop('progress_callback', None)
    
    try:
        # Record start time for performance measurement
        start_time = time.time()
        
        # Calculate dimensions to preserve aspect ratio
        height, width = height_map.shape
        aspect_ratio = width / height if height > 0 else 1.0
        
        # By default, match x_length to aspect ratio if not specified directly
        if 'x_length' not in kwargs and 'y_length' in kwargs:
            kwargs['x_length'] = kwargs['y_length'] * aspect_ratio
        elif 'y_length' not in kwargs and 'x_length' in kwargs:
            kwargs['y_length'] = kwargs['x_length'] / aspect_ratio
        # If neither is specified, use the aspect ratio with default size 1.0
        elif 'x_length' not in kwargs and 'y_length' not in kwargs:
            kwargs['x_length'] = aspect_ratio
            kwargs['y_length'] = 1.0
        
        # Special case for adaptive STL which uses a different function
        if format_type == "stl" and kwargs.pop('adaptive', False):
            result = convert_heightmap_to_adaptive_mesh(
                height_map=height_map,
                output_file=output_file,
                z_scale=kwargs.pop('z_scale', 1.0),
                base_height=kwargs.pop('base_height', 0.0),
                x_scale=kwargs.pop('x_scale', 1.0),
                y_scale=kwargs.pop('y_scale', 1.0),
                max_subdivisions=kwargs.pop('max_subdivisions', 8),
                error_threshold=kwargs.pop('max_error', 0.01),
                max_triangles=kwargs.pop('max_triangles', None),
                progress_callback=progress_callback,
                ascii=not kwargs.pop('binary', True),
                coordinate_system=kwargs.pop('coordinate_system', "right-handed"),
                origin_at_zero=kwargs.pop('origin_at_zero', True),
                invert_base=kwargs.pop('invert_base', False)
            )
        else:
            # For regular format exporters, use the mapping to call the right function
            if format_type not in FORMAT_EXPORTERS:
                logger.error(f"Unsupported format type: {format_type}")
                return None
            
            exporter_info = FORMAT_EXPORTERS[format_type]
            exporter_func = exporter_info["function"]
            
            # Prepare function parameters
            func_params = {"height_map": height_map}
            
            # Binary parameter special case for STL
            if format_type == "stl":
                kwargs["ascii_format"] = not kwargs.pop('binary', True)
            
            # Map kwargs parameters to function parameters based on the mapping
            if "params" in exporter_info:
                for func_param, kwarg_name in exporter_info["params"].items():
                    if kwarg_name in kwargs:
                        func_params[func_param] = kwargs.pop(kwarg_name)
                    elif func_param in kwargs:  # Direct parameter match
                        func_params[func_param] = kwargs.pop(func_param)
            
            # Add output_file parameter if not already mapped
            if "filename" not in func_params and "output_file" not in func_params and "output_path" not in func_params:
                # Check function signature to determine correct parameter name
                param_name = None
                for potential_name in ["output_file", "filename", "output_path"]:
                    if potential_name in exporter_func.__code__.co_varnames:
                        param_name = potential_name
                        break
                
                # Default to output_file if no match found
                if param_name is None:
                    param_name = "output_file"
                    
                func_params[param_name] = output_file
                
            # Add any default parameters
            if "defaults" in exporter_info:
                for param, value in exporter_info["defaults"].items():
                    if param not in func_params:
                        func_params[param] = value

            # For debugging
            logger.debug(f"Calling {exporter_func.__name__} with parameters: {func_params}")
            
            # Call the exporter function
            result = exporter_func(**func_params)
        
        # Calculate and log elapsed time
        elapsed = time.time() - start_time
        logger.info(f"{format_type.upper()} conversion completed in {elapsed:.2f}s")
        
        return result
            
    except Exception as e:
        logger.error(f"Error converting heightmap to {format_type}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def get_file_extension(format_type: str) -> str:
    """
    Get the file extension for a given format.
    
    Args:
        format_type: The format type string
        
    Returns:
        The appropriate file extension with leading dot
    """
    image_formats = {"normal_map", "bump_map", "heightmap", "ao_map", "displacement_map"}
    
    if format_type in image_formats:
        return ".png"
    else:
        return f".{format_type}"


def print_conversion_stats(output_file: str, format_type: str) -> Dict[str, Any]:
    """
    Get statistics about the converted file.
    
    Args:
        output_file: Path to the output file
        format_type: The format type of the output file
        
    Returns:
        Dictionary containing statistics about the converted file
    """
    # Handle tuple return values from convert_heightmap_to_adaptive_mesh
    if isinstance(output_file, tuple) and len(output_file) > 2 and isinstance(output_file[2], str):
        output_file = output_file[2]
    
    # Handle PIL Image objects (returned by some export functions)
    if hasattr(output_file, 'filename'):
        # If PIL Image with filename attribute
        output_file = output_file.filename
    elif not isinstance(output_file, str) and not isinstance(output_file, Path):
        # For any other non-string/path output (like PIL Image without filename)
        return {
            "format": format_type.upper(),
            "file_size_str": "Unknown (in-memory object)",
            "note": "File was generated but stats are not available"
        }
    
    if not os.path.exists(output_file):
        return {}
        
    stats = {
        "format": format_type.upper(),
        "file_path": output_file,
        "file_size": os.path.getsize(output_file)
    }
    
    # Format file size
    size_bytes = stats["file_size"]
    if size_bytes < 1024 * 1024:
        stats["file_size_str"] = f"{size_bytes / 1024:.1f} KB"
    else:
        stats["file_size_str"] = f"{size_bytes / (1024 * 1024):.2f} MB"
    
    # Get file creation time
    try:
        stats["creation_time"] = os.path.getctime(output_file)
        stats["creation_time_str"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stats["creation_time"]))
    except:
        pass
    
    # Try to get triangle count for STL models
    if format_type == "stl":
        try:
            # For binary STL, read triangle count from header
            with open(output_file, 'rb') as f:
                # Check for 80-byte header followed by 4-byte triangle count
                if size_bytes >= 84:
                    f.seek(80)  # Skip header
                    stats["triangle_count"] = int.from_bytes(f.read(4), byteorder='little')
        except Exception:
            pass
            
    # For OBJ files, estimate the mesh complexity
    elif format_type == "obj":
        try:
            with open(output_file, 'r') as f:
                content = f.read()
                stats["vertex_count"] = content.count('\nv ')
                stats["face_count"] = content.count('\nf ')
        except Exception:
            pass
    
    # For PLY files, read the header to get vertex/face counts
    elif format_type == "ply":
        try:
            vertex_count = 0
            face_count = 0
            with open(output_file, 'rb') as f:
                # Check first bytes to determine if binary or ASCII
                header = f.readline().decode('ascii', errors='ignore').strip()
                if not header.startswith("ply"):
                    return stats
                
                # Read header for element counts
                line = f.readline().decode('ascii', errors='ignore').strip()
                while line != "end_header":
                    if line.startswith("element vertex"):
                        vertex_count = int(line.split()[2])
                    elif line.startswith("element face"):
                        face_count = int(line.split()[2])
                    line = f.readline().decode('ascii', errors='ignore').strip()
                    
                stats["vertex_count"] = vertex_count
                stats["face_count"] = face_count
        except Exception:
            pass
    
    # For GLTF/GLB files, estimate complexity from file size
    elif format_type in ["gltf", "glb"]:
        stats["estimated_triangles"] = f"~{int(size_bytes / 50):,}"
    
    return stats


def is_large_heightmap(height_map: np.ndarray, threshold: int = 1000000) -> bool:
    """
    Check if a heightmap is considered large (exceeds threshold of total pixels).
    
    Args:
        height_map: The heightmap to check
        threshold: Number of pixels to consider large (default: 1 million)
        
    Returns:
        bool: True if the heightmap is large
    """
    return height_map.size > threshold


def get_heightmap_stats(height_map: np.ndarray) -> Dict[str, Any]:
    """
    Calculate comprehensive statistics for a heightmap.
    
    Args:
        height_map: The heightmap to analyze
        
    Returns:
        Dictionary containing heightmap statistics
    """
    if height_map is None:
        return {}
        
    stats = {
        "dimensions": height_map.shape,
        "width": height_map.shape[1],
        "height": height_map.shape[0],
        "total_pixels": height_map.size,
        "min_height": float(np.min(height_map)),
        "max_height": float(np.max(height_map)),
        "mean_height": float(np.mean(height_map)),
        "median_height": float(np.median(height_map)),
        "std_dev": float(np.std(height_map)),
        "is_large": is_large_heightmap(height_map)
    }
    
    # Calculate peak-to-valley height
    stats["peak_to_valley"] = stats["max_height"] - stats["min_height"]
    
    # Calculate roughness (RMS)
    try:
        # Remove mean plane first
        leveled = height_map - stats["mean_height"]
        stats["rms_roughness"] = float(np.sqrt(np.mean(np.square(leveled))))
    except:
        stats["rms_roughness"] = 0.0
        
    # Calculate aspect ratio
    stats["aspect_ratio"] = float(height_map.shape[1] / height_map.shape[0]) if height_map.shape[0] > 0 else 1.0
    
    return stats


def prepare_conversion_info(
    input_file: str, 
    height_map: np.ndarray, 
    original_shape: Tuple[int, int],
    format_type: str,
    output_file: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Prepare a dictionary with all conversion parameters for reporting.
    
    Args:
        input_file: Path to input file
        height_map: Heightmap being converted
        original_shape: Original dimensions before any resizing
        format_type: Output format type
        output_file: Path to output file
        **kwargs: Additional conversion parameters
        
    Returns:
        Dictionary with conversion parameters
    """
    # Calculate aspect ratio
    aspect_ratio = height_map.shape[1] / height_map.shape[0] if height_map.shape[0] > 0 else 1.0
    x_length = kwargs.get('x_length', aspect_ratio)
    y_length = kwargs.get('y_length', 1.0)
    
    info = {
        "input_file": input_file,
        "output_file": output_file,
        "format": format_type,
        "dimensions": {
            "original": original_shape,
            "processing": height_map.shape,
            "total_pixels": height_map.size,
            "aspect_ratio": aspect_ratio,
            "model_x_length": x_length,
            "model_y_length": y_length
        },
        "parameters": {}
    }
    
    # Add relevant parameters based on format type
    for key, value in kwargs.items():
        if key in ["z_scale", "base_height", "mirror_x", "rotate", "adaptive",
                  "max_error", "coordinate_system", "binary", "origin_at_zero"]:
            info["parameters"][key] = value
    
    # Add heightmap statistics
    info["heightmap_stats"] = get_heightmap_stats(height_map)
    
    return info


def display_conversion_stats(stats: Dict[str, Any]) -> None:
    """
    Format and display conversion statistics.
    
    Args:
        stats: Dictionary of conversion statistics
    """
    try:
        from rich.console import Console
        from rich.table import Table
        
        console = Console()
        summary_table = Table(title="Conversion Result")
        summary_table.add_column("Property", style="cyan")
        summary_table.add_column("Value", style="green")
        
        # Define the display order for better readability
        display_order = [
            "format", "file_path", "file_size_str", "triangle_count", 
            "face_count", "vertex_count", "estimated_triangles",
            "elapsed_time", "creation_time_str"
        ]
        
        # First add items in preferred order
        for key in display_order:
            if key in stats:
                if key == "file_path":
                    summary_table.add_row("Output File", str(stats[key]))
                elif key == "format":
                    summary_table.add_row("Output Format", str(stats[key]))
                elif key == "file_size_str":
                    summary_table.add_row("File Size", str(stats[key]))
                elif key == "triangle_count" or key == "face_count":
                    summary_table.add_row("Triangle Count", f"{stats[key]:,}")
                elif key == "vertex_count":
                    summary_table.add_row("Vertex Count", f"{stats[key]:,}")
                elif key == "estimated_triangles":
                    summary_table.add_row("Estimated Triangles", str(stats[key]))
                elif key == "elapsed_time":
                    summary_table.add_row("Processing Time", str(stats[key]))
                elif key == "creation_time_str":
                    summary_table.add_row("Creation Time", str(stats[key]))
        
        # Then add any remaining items
        for key, value in stats.items():
            if key not in display_order and key not in ["file_size", "creation_time"]:
                summary_table.add_row(key.replace("_", " ").title(), str(value))
                
        console.print(summary_table)
    except ImportError:
        # Fallback if rich is not available
        for key, value in stats.items():
            if key not in ["file_size", "creation_time"]:
                print(f"{key}: {value}")
