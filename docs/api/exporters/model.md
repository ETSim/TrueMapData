# Model Exporters

The TMD model exporters module provides tools for converting heightmaps to various 3D model formats suitable for visualization, 3D printing, and other applications.

## Supported Formats

The module can generate the following formats from height data:

- **STL**: Standard Tessellation Language files for 3D printing
- **OBJ**: Wavefront OBJ format with vertex normals
- **PLY**: Stanford Triangle Format (both binary and ASCII)
- **GLTF/GLB**: Modern 3D format for web and mobile applications
- **USDZ**: Format for Apple's AR platform
- **Three.js JSON**: Format for web-based 3D visualization

## Backend Options

TMD supports multiple backends for 3D model generation, each with different performance characteristics and dependencies:

| Backend | Description | Installation |
|---------|-------------|--------------|
| `adaptive_mesh` | TMD's adaptive mesh (default) | Built-in |
| `standard_mesh` | TMD's regular mesh | Built-in |
| `numpy_stl` | numpy-stl library | `pip install numpy-stl` |
| `meshio` | meshio library | `pip install meshio` |
| `trimesh` | trimesh library | `pip install trimesh` |
| `stl_reader` | stl_reader library | `pip install stl_reader` |
| `openstl` | OpenSTL library | `pip install openstl` |

## Functions

### STL Export

::: tmd.exporters.model.convert_heightmap_to_stl

### OBJ Export

::: tmd.exporters.model.convert_heightmap_to_obj

### PLY Export

::: tmd.exporters.model.convert_heightmap_to_ply

### GLTF/GLB Export

::: tmd.exporters.model.convert_heightmap_to_gltf
::: tmd.exporters.model.convert_heightmap_to_glb

### Three.js Export

::: tmd.exporters.model.convert_heightmap_to_threejs

### USDZ Export

::: tmd.exporters.model.convert_heightmap_to_usdz

### Adaptive Mesh Generation

::: tmd.exporters.model.convert_heightmap_to_adaptive_mesh

## Usage Examples

### Basic STL Export

```python
from tmd.exporters.model import convert_heightmap_to_stl

# Generate a basic STL model
convert_heightmap_to_stl(
    height_map,
    filename="model.stl",
    z_scale=10.0,  # Exaggerate height by 10x
    base_height=1.0  # Add a 1-unit thick base
)
```

### Using Different Backends

```python
from tmd.exporters.model import convert_heightmap_to_stl
from tmd.exporters.model.backends import ModelBackend

# Using OpenSTL backend for potentially faster processing
convert_heightmap_to_stl(
    height_map,
    filename="model_openstl.stl",
    z_scale=10.0,
    backend=ModelBackend.OPENSTL
)

# Using Trimesh backend for additional features
convert_heightmap_to_stl(
    height_map,
    filename="model_trimesh.stl",
    z_scale=10.0,
    backend=ModelBackend.TRIMESH
)
```

### Adaptive vs Standard Mesh Generation

```python
from tmd.exporters.model import convert_heightmap_to_stl

# Adaptive mesh (fewer triangles in flat areas)
convert_heightmap_to_stl(
    height_map,
    filename="adaptive.stl",
    adaptive=True,
    error_threshold=0.01,  # Controls level of detail
    max_subdivisions=10    # Controls maximum resolution
)

# Standard mesh (uniform grid of triangles)
convert_heightmap_to_stl(
    height_map,
    filename="standard.stl",
    adaptive=False
)
```

### Exporting for Web Visualization

```python
from tmd.exporters.model import convert_heightmap_to_gltf, convert_heightmap_to_threejs

# Export as glTF for modern web frameworks
convert_heightmap_to_gltf(
    height_map,
    filename="model.gltf",
    z_scale=5.0
)

# Export for Three.js applications
convert_heightmap_to_threejs(
    height_map,
    filename="model.json",
    z_scale=5.0
)
```