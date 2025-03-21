# 3D Model Exporter

The 3D Model Exporter module provides functions to convert height maps to various 3D model formats for visualization, CAD integration, or 3D printing.

## Overview

This module enables exporting height maps to common 3D model formats:

- **STL** (STereoLithography): The most widely used format for 3D printing
- **OBJ** (Wavefront Object): Common in 3D graphics and visualization
- **PLY** (Polygon File Format): Supports additional vertex properties

## Export Functions

::: tmd.exporters.model.convert_heightmap_to_stl

::: tmd.exporters.model.convert_heightmap_to_obj

::: tmd.exporters.model.convert_heightmap_to_ply

## Implementation Options

The module provides two implementation options for each export format:

1. **Custom implementations**: Direct implementations with fine-grained control
2. **Meshio-based implementations**: Using the meshio library for broader format support

```python
# Custom implementation
convert_heightmap_to_stl(height_map, "output.stl")

# Meshio-based implementation
convert_heightmap_to_stl_meshio(height_map, "output.stl") 
```

## Examples

### Basic STL Export

```python
from tmd.exporters.model import convert_heightmap_to_stl

# Export to STL with default parameters
convert_heightmap_to_stl(
    height_map,
    filename="surface.stl",
    z_scale=1.0
)

# Export with ASCII format
convert_heightmap_to_stl(
    height_map,
    filename="surface_ascii.stl",
    z_scale=1.0,
    ascii=True
)
```

### Custom Physical Dimensions

```python
from tmd.exporters.model import convert_heightmap_to_stl, convert_heightmap_to_obj

# Export with specific physical dimensions (in mm)
convert_heightmap_to_stl(
    height_map,
    filename="surface_10mm.stl",
    x_length=10.0,  # 10mm width
    y_length=10.0,  # 10mm length 
    z_scale=2.0,    # Exaggerate height by 2x
    ascii=False
)

# Export the same surface to OBJ format
convert_heightmap_to_obj(
    height_map,
    filename="surface_10mm.obj",
    x_length=10.0,
    y_length=10.0,
    z_scale=2.0
)
```

### Adding a Solid Base for 3D Printing

```python
from tmd.exporters.model import convert_heightmap_to_stl

# Export with a solid base for better 3D printing stability
convert_heightmap_to_stl(
    height_map,
    filename="surface_with_base.stl",
    x_length=50.0,   # 50mm width
    y_length=50.0,   # 50mm length
    z_scale=1.0,     
    base_height=2.0  # 2mm thick solid base
)
```

### Using Meshio-Based Exporters

```python
from tmd.exporters.model import convert_heightmap_to_ply_meshio

# Export using meshio implementation
convert_heightmap_to_ply_meshio(
    height_map,
    filename="surface.ply",
    x_length=10.0,
    y_length=10.0,
    z_scale=2.0
)
```

### Workflow: 3D Printing

For 3D printing applications, follow these steps:

1. **Prepare the height map**:
   ```python
   # Apply Gaussian filter to reduce noise
   from tmd.utils.filter import apply_gaussian_filter
   smoothed_map = apply_gaussian_filter(height_map, sigma=1.0)
   
   # Threshold to remove outliers
   from tmd.utils.processing import threshold_height_map
   prepared_map = threshold_height_map(
       smoothed_map, 
       min_height=smoothed_map.min() * 1.05,
       max_height=smoothed_map.max() * 0.95
   )
   ```

2. **Export with appropriate settings**:
   ```python
   from tmd.exporters.model import convert_heightmap_to_stl
   
   # For 3D printing, use binary STL with height exaggeration and a solid base
   convert_heightmap_to_stl(
       prepared_map,
       filename="print_ready.stl",
       x_length=50.0,    # 50mm width
       y_length=50.0,    # 50mm length
       z_scale=10.0,     # Exaggerate height by 10x for visibility
       base_height=3.0,  # 3mm solid base for stability
       ascii=False       # Binary format for smaller file size
   )
   ```

3. **Load into slicer software** (Cura, PrusaSlicer, etc.)

## Coordinate Systems

The model exporters use the following coordinate system:

- **X-axis**: Left to right in the height map
- **Y-axis**: Top to bottom in the height map
- **Z-axis**: Height value (normal to the surface)

This matches standard 3D coordinate systems with +Z pointing upward.
