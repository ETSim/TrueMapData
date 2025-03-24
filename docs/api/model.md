# Model Export API

The TMD model export API provides functions for converting heightmaps to various 3D model formats suitable for visualization, 3D printing, simulation, and other applications.

## Basic Usage

```python
from tmd.exporters.model import convert_heightmap_to_stl

# Convert a heightmap to STL
stl_file = convert_heightmap_to_stl(
    height_map,         # NumPy array of height values
    filename="output.stl",
    z_scale=10.0,       # Exaggerate heights by 10x
    base_height=1.0     # Add a 1-unit thick base
)
```

## Supported Formats

The model export API supports the following formats:

| Format | Function | Description |
|--------|----------|-------------|
| STL | `convert_heightmap_to_stl` | Standard Triangle Language - Common for 3D printing |
| OBJ | `convert_heightmap_to_obj` | Wavefront OBJ - With vertex normals |
| PLY | `convert_heightmap_to_ply` | Stanford Triangle Format - ASCII or binary |
| glTF/GLB | `convert_heightmap_to_gltf/glb` | GL Transmission Format - For web applications |
| Three.js | `convert_heightmap_to_threejs` | Three.js JSON format - For web-based 3D |
| USDZ | `convert_heightmap_to_usdz` | Universal Scene Description - For Apple's AR |
| SDF | `export_heightmap_to_sdf` | Signed Distance Field - For simulations |
| NVBD | `export_heightmap_to_nvbd` | NVIDIA Blast Destructible - For physics simulations |

## STL Export

The Standard Triangle Language (STL) format is widely used for 3D printing and is supported by most CAD software.

```python
from tmd.exporters.model import convert_heightmap_to_stl

# Basic STL export
convert_heightmap_to_stl(
    height_map,
    filename="model.stl",
    z_scale=1.0
)

# STL with adaptive mesh for optimized triangle count
convert_heightmap_to_stl(
    height_map,
    filename="adaptive_model.stl",
    z_scale=1.0,
    adaptive=True,
    error_threshold=0.01,
    max_subdivisions=8
)

# ASCII STL (larger files but human-readable)
convert_heightmap_to_stl(
    height_map,
    filename="ascii_model.stl",
    ascii=True
)
```

## OBJ Export

The Wavefront OBJ format includes vertex normals which can improve rendering quality.

```python
from tmd.exporters.model import convert_heightmap_to_obj

# Basic OBJ export
convert_heightmap_to_obj(
    height_map,
    filename="model.obj",
    z_scale=1.0
)

# OBJ with physical dimensions (e.g., 100mm x 100mm)
convert_heightmap_to_obj(
    height_map,
    filename="sized_model.obj",
    x_length=100.0,
    y_length=100.0,
    z_scale=10.0
)
```

## Web-Ready Formats

For web-based 3D visualization, the API provides several options:

```python
from tmd.exporters.model import convert_heightmap_to_gltf, convert_heightmap_to_threejs

# glTF with separate JSON and binary files
convert_heightmap_to_gltf(
    height_map,
    filename="model.gltf",
    z_scale=1.0
)

# GLB (binary glTF) - single file format
convert_heightmap_to_glb(
    height_map,
    filename="model.glb",
    z_scale=1.0
)

# Three.js JSON format
convert_heightmap_to_threejs(
    height_map,
    filename="model.json",
    z_scale=1.0
)
```

## Specialized Formats

For physics simulations and other specialized applications:

```python
from tmd.exporters.model import export_heightmap_to_sdf, export_heightmap_to_nvbd

# Export as Signed Distance Field
export_heightmap_to_sdf(
    height_map,
    filename="terrain.sdf",
    scale=1.0
)

# Export as NVIDIA Blast Destructible
export_heightmap_to_nvbd(
    height_map,
    filename="destructible.nvbd",
    scale=1.0,
    chunk_size=16
)
```

## Adaptive Mesh Generation

For efficient mesh generation with triangles distributed according to detail level:

```python
from tmd.exporters.model import convert_heightmap_to_adaptive_mesh

# Generate an adaptive mesh with more triangles in areas of high detail
vertices, faces = convert_heightmap_to_adaptive_mesh(
    height_map,
    error_threshold=0.01,
    max_subdivisions=8
)

# The returned vertices and faces can be exported to any format
```

## Advanced Usage

### Custom Base

Adding a solid base to the model can improve 3D printing results:

```python
convert_heightmap_to_stl(
    height_map,
    filename="with_base.stl",
    z_scale=1.0,
    base_height=2.0  # Add a 2mm thick base
)
```

### Physical Dimensions

To specify the physical size of the resulting model:

```python
convert_heightmap_to_stl(
    height_map,
    filename="physical_model.stl",
    x_offset=-50,    # Center model around origin
    y_offset=-50,
    x_length=100.0,  # 100mm wide
    y_length=100.0,  # 100mm deep
    z_scale=10.0     # 10mm per unit of height
)
```

### Using Different Backends

For better performance or compatibility with specific toolchains:

```python
from tmd.exporters.model.backends import ModelBackend

# Use optimized backend
convert_heightmap_to_stl(
    height_map,
    filename="fast_model.stl",
    backend=ModelBackend.NUMPY_STL
)
```
