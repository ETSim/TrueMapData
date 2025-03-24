""".

Material map set generation module.

This module provides functions for creating complete sets of material maps from heightmaps.
"""

import os
import json
import logging
import numpy as np
from PIL import Image
import cv2

from .displacement_map import convert_heightmap_to_displacement_map
from .normal_map import convert_heightmap_to_normal_map
from .bump_map import convert_heightmap_to_bump_map
from .hillshade import generate_hillshade
from .utils import generate_roughness_map, create_orm_map, generate_edge_map

from tmd.utils.filters import calculate_rms_roughness, calculate_surface_gradient

logger = logging.getLogger(__name__)

def generate_maps_from_tmd(height_map, tmd_metadata, output_dir="."):
    """.

    Generate a complete set of texture maps using heightmap data and TMD metadata.

    Args:
        height_map: 2D numpy array representing height data.
        tmd_metadata: Dictionary containing metadata parameters.
        output_dir: Directory to save output maps.

    Returns:
        Dictionary of generated maps.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Extract parameters from metadata
    params = {
        "normal_strength": tmd_metadata.get("normal_strength", 1.0),
        "bump_strength": tmd_metadata.get("bump_strength", 1.0),
        "bump_blur_radius": tmd_metadata.get("bump_blur_radius", 1.0),
        "roughness_scale": tmd_metadata.get("roughness_scale", 1.0),
        "edge_threshold1": tmd_metadata.get("edge_threshold1", 50),
        "edge_threshold2": tmd_metadata.get("edge_threshold2", 150),
    }

    # Physical dimensions
    units = tmd_metadata.get("units", "µm")
    x_length = tmd_metadata.get("x_length", 10.0)
    y_length = tmd_metadata.get("y_length", 10.0)

    # Calculate pixel size
    rows, cols = height_map.shape
    pixel_size_x = x_length / cols if cols > 0 else 1.0
    pixel_size_y = y_length / rows if rows > 0 else 1.0

    logger.info(
        f"Dimensions: {x_length}x{y_length} {units}, pixel size: {pixel_size_x:.4f}x{pixel_size_y:.4f} {units}/pixel"
    )

    maps = {}

    # Basic maps
    disp_filename = os.path.join(output_dir, "displacement.png")
    maps["displacement"] = convert_heightmap_to_displacement_map(
        height_map, filename=disp_filename, units=units
    )

    norm_filename = os.path.join(output_dir, "normal.png")
    maps["normal"] = convert_heightmap_to_normal_map(
        height_map, filename=norm_filename, strength=params["normal_strength"]
    )

    bump_filename = os.path.join(output_dir, "bump.png")
    maps["bump"] = convert_heightmap_to_bump_map(
        height_map,
        filename=bump_filename,
        strength=params["bump_strength"],
        blur_radius=params["bump_blur_radius"],
    )

    # Roughness map
    roughness_map = generate_roughness_map(height_map, scale=params["roughness_scale"])
    rms_roughness = calculate_rms_roughness(height_map)
    maps["roughness"] = roughness_map

    roughness_filename = os.path.join(output_dir, f"roughness_RMS_{rms_roughness:.2f}{units}.png")
    Image.fromarray(roughness_map).save(roughness_filename)

    # Derived maps
    ao_map = 255 - np.array(maps["displacement"])
    maps["ambient_occlusion"] = ao_map
    Image.fromarray(ao_map).save(os.path.join(output_dir, "ao.png"))

    base_color = height_map.copy().astype(np.uint8)
    maps["base_color"] = base_color
    Image.fromarray(base_color).save(os.path.join(output_dir, "base_color.png"))

    # Slope map
    grad_x, grad_y = calculate_surface_gradient(height_map, dx=pixel_size_x, dy=pixel_size_y)
    slope_map = np.sqrt(grad_x**2 + grad_y**2)
    slope_min, slope_max = slope_map.min(), slope_map.max()
    normalized_slope = (
        ((slope_map - slope_min) / (slope_max - slope_min) * 255).astype(np.uint8)
        if slope_max > slope_min
        else np.zeros_like(slope_map, dtype=np.uint8)
    )
    maps["slope"] = normalized_slope

    slope_filename = os.path.join(output_dir, f"slope_max_{slope_max:.2f}deg.png")
    Image.fromarray(normalized_slope).save(slope_filename)

    # Combined maps
    orm_map = create_orm_map(ao_map, roughness_map, base_color)
    maps["orm"] = orm_map
    cv2.imwrite(os.path.join(output_dir, "orm.png"), orm_map)

    edge_map = generate_edge_map(
        np.array(maps["displacement"]),
        threshold1=params["edge_threshold1"],
        threshold2=params["edge_threshold2"],
    )
    maps["edge"] = edge_map
    cv2.imwrite(os.path.join(output_dir, "edge.png"), edge_map)

    # Smoothness (inverse roughness)
    smoothness_map = 255 - roughness_map
    maps["smoothness"] = smoothness_map
    cv2.imwrite(os.path.join(output_dir, "smoothness.png"), smoothness_map)

    # Save metadata as JSON
    map_metadata = {
        "physical_dimensions": {
            "width": x_length,
            "height": y_length,
            "units": units,
            "pixel_size_x": pixel_size_x,
            "pixel_size_y": pixel_size_y,
        },
        "roughness": {"rms": float(rms_roughness)},
        "slope": {"max_angle": float(np.arctan(slope_max) * 180 / np.pi) if slope_max > 0 else 0.0},
        "generated_maps": list(maps.keys()),
    }

    with open(os.path.join(output_dir, "map_metadata.json"), "w") as f:
        json.dump(map_metadata, f, indent=2)

    return maps


def generate_all_maps(height_map, output_dir="."):
    """.

    Generate a suite of maps from a height map using default parameters.

    Args:
        height_map: 2D numpy array representing height data.
        output_dir: Directory to save the output maps.

    Returns:
        Dictionary of generated maps.
    """
    default_metadata = {
        "normal_strength": 1.0,
        "bump_strength": 1.0,
        "bump_blur_radius": 1.0,
        "roughness_scale": 1.0,
        "edge_threshold1": 50,
        "edge_threshold2": 150,
        "material_channel_type": "rgbe",
        "units": "µm",
        "x_length": 10.0,
        "y_length": 10.0,
    }

    maps = generate_maps_from_tmd(height_map, default_metadata, output_dir)

    # Add hillshade to the generated maps
    hillshade_file = os.path.join(output_dir, "hillshade.png")
    maps["hillshade"] = generate_hillshade(
        height_map, filename=hillshade_file, altitude=45, azimuth=0
    )

    return maps
