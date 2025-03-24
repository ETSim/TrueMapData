# Supported Export Formats

TMD supports exporting height maps to a variety of 3D model formats.

## Common Formats

### OBJ
Standard 3D object format supported by most 3D software.

```python
from tmd.exporters.model.obj import export_heightmap_to_obj

export_heightmap_to_obj(
    height_map=height_map,
    filename="output.obj",
    x_offset=0, y_offset=0,
    x_length=1, y_length=1,
    z_scale=1.0,
    base_height=0.0
)
```

### STL
Common format for 3D printing and CAD.

```python
from tmd.exporters.model.stl import export_heightmap_to_stl

export_heightmap_to_stl(
    height_map=height_map,
    filename="output.stl",
    x_offset=0, y_offset=0,
    x_length=1, y_length=1,
    z_scale=1.0,
    base_height=0.0,
    binary=True  # Set to False for ASCII STL
)
```

### PLY
Polygon File Format with support for colors and other properties.

```python
from tmd.exporters.model.ply import export_heightmap_to_ply

export_heightmap_to_ply(
    height_map=height_map,
    filename="output.ply",
    x_offset=0, y_offset=0,
    x_length=1, y_length=1,
    z_scale=1.0,
    base_height=0.0,
    binary=True  # Set to False for ASCII PLY
)
```

### glTF / GLB
Modern 3D format optimized for web and real-time applications.

```python
from tmd.exporters.model.gltf import convert_heightmap_to_gltf, convert_heightmap_to_glb

# Export as glTF
convert_heightmap_to_gltf(
    height_map=height_map,
    filename="output.gltf",
    x_offset=0, y_offset=0,
    x_length=1, y_length=1,
    z_scale=1.0,
    base_height=0.0,
    add_texture=True
)

# Export as binary GLB
convert_heightmap_to_glb(
    height_map=height_map,
    filename="output.glb",
    x_offset=0, y_offset=0,
    x_length=1, y_length=1,
    z_scale=1.0,
    base_height=0.0,
    add_texture=True
)
```

### Three.js JSON
Format for direct use with the Three.js JavaScript library.

```python
from tmd.exporters.model.threejs import convert_heightmap_to_threejs

convert_heightmap_to_threejs(
    height_map=height_map,
    filename="output.json",
    x_offset=0, y_offset=0,
    x_length=1, y_length=1,
    z_scale=1.0,
    base_height=0.0,
    add_texture=True,
    compress=False,
    add_wireframe=False
)
```

### USD / USDZ
Pixar's Universal Scene Description format, with USDZ support for AR/Apple platforms.

```python
from tmd.exporters.model.usd import export_heightmap_to_usd, export_heightmap_to_usdz

# Export as USD
export_heightmap_to_usd(
    height_map=height_map,
    filename="output.usda",  # or .usd/.usdc
    x_offset=0, y_offset=0,
    x_length=1, y_length=1,
    z_scale=1.0,
    base_height=0.0,
    add_texture=True,
    up_axis="Y"  # or "Z"
)

# Export as USDZ (for AR/Apple platforms)
export_heightmap_to_usdz(
    height_map=height_map,
    filename="output.usdz",
    x_offset=0, y_offset=0,
    x_length=1, y_length=1,
    z_scale=1.0,
    base_height=0.0,
    add_texture=True
)
```

#### USDZ on Ubuntu
Exporting to USDZ format requires Pixar's USD tools, which can be built from source on Ubuntu:

1. Install dependencies:
   ```bash
   sudo apt update
   sudo apt install -y build-essential cmake python3-dev libboost-all-dev git opensubdiv-dev
   ```

2. Clone and build USD:
   ```bash
   git clone https://github.com/PixarAnimationStudios/USD.git
   cd USD
   git checkout v23.05  # Or another stable release
   mkdir build && cd build
   cmake .. -DCMAKE_INSTALL_PREFIX=../install -DPXR_ENABLE_USDVIEW=OFF
   make -j$(nproc)
   make install
   ```

3. Add USD tools to your path or specify full path to usdzip tool.

## Special Formats

### SDF
Signed Distance Field format for specialized applications.

```python
from tmd.exporters.model.sdf import export_heightmap_to_sdf

export_heightmap_to_sdf(
    height_map=height_map,
    filename="output.sdf",
    scale=1.0,
    offset=0.0,
    grid_size=(1.0, 1.0, 1.0)
)
```

### NVBD
NVIDIA Binary Data format for NVIDIA applications.

```python
from tmd.exporters.model.nvbd import export_heightmap_to_nvbd

export_heightmap_to_nvbd(
    height_map=height_map,
    filename="output.nvbd",
    scale=1.0,
    offset=0.0,
    chunk_size=16,
    include_normals=True
)
```
