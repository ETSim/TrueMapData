# STL Exporter

The STL exporter module provides functions to convert height maps to STL files for 3D printing or visualization in CAD software.

## Overview

STL (STereoLithography) is a file format that represents 3D surfaces as triangular meshes. This module provides functions to convert height maps into STL files, allowing you to physically produce your surface data through 3D printing.

## Functions

::: tmd.exporters.model.convert_heightmap_to_stl

## Examples

### Basic Export

```python
from tmd.processor import TMDProcessor
from tmd.exporters.model import convert_heightmap_to_stl

# Process a TMD file
processor = TMDProcessor("example.tmd")
processor.process()
height_map = processor.get_height_map()

# Export to STL
convert_heightmap_to_stl(
    height_map=height_map,
    filename="surface.stl",
    z_scale=1.0
)
```

### Customized Export

```python
# Export with customized parameters
convert_heightmap_to_stl(
    height_map=height_map,
    filename="enhanced_surface.stl",
    x_offset=5.0,         # Shift model in X direction
    y_offset=10.0,        # Shift model in Y direction
    x_length=100.0,       # Physical X dimension in mm
    y_length=100.0,       # Physical Y dimension in mm
    z_scale=5.0,          # Exaggerate height by 5x
    ascii=True            # Use ASCII STL format instead of binary
)
```

### Export for 3D Printing

When exporting for 3D printing, you may need to adjust parameters to get good results:

=== "Small Model (< 10cm)"

    ```python
    convert_heightmap_to_stl(
        height_map=height_map,
        filename="small_model.stl",
        x_length=50.0,     # 50mm width
        y_length=50.0,     # 50mm length
        z_scale=10.0,      # Exaggerate height for visibility
        ascii=False        # Use binary format for smaller file size
    )
    ```

=== "Large Model (> 10cm)"

    ```python
    convert_heightmap_to_stl(
        height_map=height_map,
        filename="large_model.stl",
        x_length=150.0,    # 150mm width
        y_length=150.0,    # 150mm length
        z_scale=5.0,       # Less exaggeration for larger model
        ascii=False        # Binary format is essential for large models
    )
    ```

## Tips for 3D Printing

- **Base Addition**: Consider adding a base to your model for stability
- **Z-Scale**: Adjust the z_scale parameter to make features visible
- **Resolution**: For large height maps, consider downsampling to reduce file size
- **Orientation**: Print the model flat on the build plate for best results
