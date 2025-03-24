# Image Exporters

The TMD image exporters module provides tools for converting heightmaps to various image formats that are useful for visualization, texture mapping, and material creation.

## Supported Maps

The module can generate the following types of maps from height data:

- **Displacement Maps**: Grayscale images representing height values
- **Normal Maps**: RGB images encoding surface normals for dynamic lighting
- **Bump Maps**: Grayscale images for simple height-based lighting
- **Roughness Maps**: Grayscale images representing surface texture variation
- **Ambient Occlusion Maps**: Grayscale images representing surface occlusion
- **Multi-Channel Maps**: Combined maps for PBR materials
- **Hillshade Maps**: Grayscale images simulating terrain illumination

## Functions

### Displacement Maps

::: tmd.exporters.image.convert_heightmap_to_displacement_map

### Normal Maps

::: tmd.exporters.image.convert_heightmap_to_normal_map

### Bump Maps

::: tmd.exporters.image.convert_heightmap_to_bump_map

### Hillshade Maps

::: tmd.exporters.image.generate_hillshade

### Material Maps

::: tmd.exporters.image.convert_heightmap_to_multi_channel_map

### Utility Functions

::: tmd.exporters.image.generate_roughness_map

::: tmd.exporters.image.generate_all_maps

## Usage Examples

### Basic Map Generation

```python
from tmd.exporters.image import convert_heightmap_to_displacement_map, convert_heightmap_to_normal_map

# Generate a displacement map
displacement_map = convert_heightmap_to_displacement_map(
    height_map,
    filename="displacement.png",
    units="mm"
)

# Generate a normal map
normal_map = convert_heightmap_to_normal_map(
    height_map,
    filename="normal.png",
    strength=2.0  # Enhance the effect
)
```

### Creating a Hillshade Visualization

```python
from tmd.exporters.image import generate_hillshade

# Create a hillshade with default lighting (45° altitude, 0° azimuth)
hillshade = generate_hillshade(
    height_map,
    filename="hillshade_default.png"
)

# Create hillshades with different lighting angles
hillshade_afternoon = generate_hillshade(
    height_map,
    filename="hillshade_afternoon.png",
    altitude=30,
    azimuth=225,  # Southwest light direction
    z_factor=2.0  # Exaggerate vertical features
)
```

### Creating a Complete Material Set

```python
from tmd.exporters.image import generate_all_maps

# Generate all map types in one operation
maps = generate_all_maps(height_map, output_dir="textures")

# Access individual maps from the result
displacement = maps["displacement"]
normal = maps["normal"]
hillshade = maps["hillshade"]
```

## Working with Map Sets

When developing materials for games or 3D applications, you'll often need a complete set of texture maps. The `generate_all_maps` function creates all necessary maps with consistent parameters.

For terrain visualization, use the hillshade function with different angles to highlight various features:

- Low altitude (15-30°) with varied azimuths: Shows subtle terrain details
- Higher altitude (45-60°): Shows overall terrain structure
- Adjusting z_factor: Controls the emphasis of height differences
