# Image Exporter

The Image Exporter module provides functions to convert height maps into various image formats useful for 3D graphics, material rendering, and visual analysis.

## Overview

This module converts TMD height maps into specialized image formats like displacement maps, normal maps, bump maps, and multi-channel maps. These images are particularly useful for:

- 3D rendering in graphics software
- Material creation in game engines
- Physical surface visualization
- Surface property analysis

## Main Functions

::: tmd.exporters.image.convert_heightmap_to_displacement_map

::: tmd.exporters.image.convert_heightmap_to_normal_map

::: tmd.exporters.image.convert_heightmap_to_bump_map

::: tmd.exporters.image.convert_heightmap_to_multi_channel_map

## Utility Functions

::: tmd.exporters.image.generate_roughness_map

::: tmd.exporters.image.create_terrain_type_map

::: tmd.exporters.image.generate_maps_from_tmd

## Examples

### Basic Usage

```python
from tmd.processor import TMDProcessor
from tmd.exporters.image import convert_heightmap_to_displacement_map, convert_heightmap_to_normal_map

# Process a TMD file
processor = TMDProcessor("example.tmd")
processor.process()
height_map = processor.get_height_map()

# Generate displacement map (grayscale)
displacement_img = convert_heightmap_to_displacement_map(
    height_map,
    filename="displacement.png",
    units="µm"
)

# Generate normal map (RGB)
normal_img = convert_heightmap_to_normal_map(
    height_map,
    filename="normal.png",
    strength=1.5
)
```

### Complete Material Set

```python
from tmd.exporters.image import generate_maps_from_tmd

# Configure metadata
metadata = {
    "normal_strength": 2.0,
    "bump_strength": 1.5,
    "roughness_scale": 1.2,
    "terrain_type": "mountain",
    "units": "µm",
    "x_length": 15.0,
    "y_length": 15.0
}

# Generate all maps in one function call
maps = generate_maps_from_tmd(
    height_map,
    tmd_metadata=metadata,
    output_dir="output/materials"
)

# The 'maps' dictionary contains all generated images
displacement = maps["displacement"]
normal = maps["normal"]
roughness = maps["roughness"]
orm = maps["orm"]  # Combined Occlusion-Roughness-Metallic map
```

## Terrain Type Maps

The `create_terrain_type_map()` function can generate specialized texture maps with different characteristics based on terrain type:

- **Mountain**: Emphasizes slopes and elevations
- **Desert**: Adds fine noise patterns for sand-like textures
- **Forest**: Generates organic patterns suitable for vegetation
- **Generic**: Standard height-based coloration

## Output File Types

All functions in this module generate standard PNG image files that can be used directly in:

- 3D modeling software like Blender
- Game engines like Unity and Unreal
- Texture painting tools like Substance Painter
- Image editing software like Photoshop
