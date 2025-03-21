# SDF Exporter

The SDF (Signed Distance Field) exporter allows you to convert height maps to SDF files that can be used in various 3D applications, physics simulations, and rendering engines.

## What is an SDF?

A Signed Distance Field (SDF) is a spatial representation where each point contains the distance to the closest surface. The "signed" aspect means that distances are negative inside the object and positive outside the object. SDFs are widely used in:

- Real-time ray marching rendering
- Collision detection in physics simulations
- 3D modeling operations (e.g., CSG boolean operations)
- Font rendering

## File Format

The TMD library implements a basic SDF file format with the following structure:

- **Magic number**: `SDF1` (4 bytes)
- **Version**: 1 (int, 4 bytes)
- **Width**: Number of columns (int, 4 bytes)
- **Height**: Number of rows (int, 4 bytes)
- **Data**: Height values as 32-bit floats, row by row

## Usage

```python
from tmd.exporters.sdf import export_heightmap_to_sdf

# Export a height map to SDF format
result = export_heightmap_to_sdf(
    heightmap,        # 2D numpy array of height values
    'output.sdf',     # Output file path
    scale=1.0,        # Optional scaling factor (default: 1.0)
    offset=0.0        # Optional height offset (default: 0.0)
)

# Check if export was successful
if result:
    print("Export successful")
else:
    print("Export failed")
```

## Parameters

- **heightmap**: 2D numpy array containing height values
- **output_file**: Path where the SDF file will be saved
- **scale**: Optional scaling factor for height values (default: 1.0)
- **offset**: Optional offset added to all height values (default: 0.0)

## Example Applications

### Using SDFs in Rendering

SDFs are commonly used in shaders for real-time rendering of complex shapes and surfaces. The generated SDF can be loaded into a graphics engine to achieve effects like:

- Ray-marched rendering of the height field
- Soft shadows based on distance fields
- Ambient occlusion

### Using SDFs in Physics Simulations

In physics engines, SDFs can be used for efficient collision detection:

1. The SDF file is loaded into memory
2. Collision queries use the distance field to determine proximity to surfaces
3. Contact resolution is performed based on gradient information in the SDF

## Limitations

The current implementation treats the height map as a distance field directly. For true SDF functionality in 3D applications, you may need to post-process the exported file to generate proper signed distances throughout a volume.
