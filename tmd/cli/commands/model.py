#!/usr/bin/env python3
"""
Model generation commands for TMD CLI.

This module provides functions for creating 3D models from TMD files
using different triangulation methods and mesh generation algorithms.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Callable

# Import CLI utilities
from tmd.cli.core import (
    console,
    print_warning,
    print_error,
    print_success,
    load_config,
    save_config,
    load_tmd_file
)

def generate_model_command(
    tmd_file: Path,
    output_file: Optional[Path] = None,
    z_scale: float = 1.0,
    base_height: float = 0.0,
    max_triangles: Optional[int] = None,
    error_threshold: float = 0.01,
    coordinate_system: str = "right-handed",
    origin_at_zero: bool = True,
    invert_base: bool = False,
    progress_callback: Optional[Callable[[float], None]] = None
) -> bool:
    """
    Generate a 3D model from a TMD file.
    
    Args:
        tmd_file: Path to TMD file
        output_file: Path to output file (STL, OBJ, or PLY)
        z_scale: Scaling factor for height values
        base_height: Height of solid base below the model
        max_triangles: Maximum number of triangles
        error_threshold: Error threshold for adaptive subdivision
        coordinate_system: Coordinate system orientation
        origin_at_zero: Place origin at center of model
        invert_base: Invert base to create mold/negative
        progress_callback: Optional callback function for progress updates
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load TMD file
        data = load_tmd_file(tmd_file, with_console_status=True)
        if not data:
            print_error(f"Failed to load TMD file: {tmd_file}")
            return False
        
        # Get height map
        height_map = data.height_map()
        
        # Determine output filename if not specified
        if output_file is None:
            # Load config for default format
            config = load_config()
            default_format = config.get("model", {}).get("default_format", "stl")
            output_format = default_format.lower()
            
            # Create output filename
            from tmd.cli.core.io import create_output_dir
            output_dir = create_output_dir(subdir="models")
            output_file = output_dir / f"{tmd_file.stem}.{output_format}"
        
        # Import model generation functionality
        from tmd.model.adaptive_mesh import convert_heightmap_to_adaptive_mesh
        
        # Generate the mesh
        with console.status(f"Generating 3D model from {tmd_file.name}..."):
            result = convert_heightmap_to_adaptive_mesh(
                height_map=height_map,
                output_file=str(output_file),
                z_scale=z_scale,
                base_height=base_height,
                error_threshold=error_threshold,
                max_triangles=max_triangles,
                progress_callback=progress_callback,
                coordinate_system=coordinate_system,
                origin_at_zero=origin_at_zero,
                invert_base=invert_base
            )
            
            if result is None:
                print_error(f"Failed to generate 3D model")
                return False
            
            vertices, faces = result
            
            print_success(f"Generated 3D model with {len(vertices)} vertices and {len(faces)} triangles")
            print_success(f"Model saved to {output_file}")
            
            return True
            
    except ImportError:
        print_error("3D model generation functionality not available")
        print_warning("Make sure SciPy, NumPy, and OpenCV are installed")
        return False
        
    except Exception as e:
        print_error(f"Error generating 3D model: {e}")
        return False
