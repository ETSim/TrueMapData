# Converting EXR Heightmaps to STL Models

This guide explains how to convert EXR (OpenEXR) heightmap files to STL 3D models using TMD. EXR files are commonly used in computer graphics and visual effects because they support high dynamic range and floating-point precision, making them ideal for storing detailed height information.

## Command Line Usage

TMD provides a dedicated command for EXR to STL conversion:

```bash
tmd2model exr2stl heightmap.exr --z-scale 10.0 --base-height 0.5
```

This command will:
1. Load the EXR file as a heightmap
2. Apply adaptive triangulation to create an efficient 3D mesh
3. Export the result as a binary STL file

### Command Options

| Option | Description |
|--------|-------------|
| `--output`, `-o` | Output STL file path (default: based on input file) |
| `--z-scale`, `-z` | Z-axis scaling factor (default: 1.0) |
| `--base-height`, `-b` | Height of solid base below model (default: 0.0) |
| `--max-error`, `-e` | Maximum error for adaptive triangulation (default: 0.01) |
| `--max-subdivisions`, `-m` | Maximum quad tree subdivisions for adaptive algorithm (default: 8) |
| `--max-triangles`, `-n` | Maximum triangle count (default: unlimited) |

## Python API Usage

You can also perform the conversion from Python code:

```python
import os
from tmd.utils.heightmap_io import load_heightmap
from tmd.exporters.model import convert_heightmap_to_stl

# Enable OpenEXR support in OpenCV
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

# Load EXR heightmap
heightmap = load_heightmap("heightmap.exr", normalize=True)

# Convert to STL
convert_heightmap_to_stl(
    height_map=heightmap,
    filename="output.stl",
    z_scale=10.0,
    base_height=0.5,
    adaptive=True,
    max_subdivisions=8,
    error_threshold=0.01
)
```

## Using Adaptive Mesh Generation

When working with EXR files, which often contain detailed terrain information, the adaptive mesh generation is particularly useful. This technique:

1. Applies more triangles to areas with complex features
2. Uses fewer triangles in flat areas
3. Optimizes memory usage for large heightmaps
4. Produces more efficient 3D models

You can control the level of detail with the `--max-error` parameter. Lower values (e.g., 0.005) produce more detailed models with more triangles, while higher values (e.g., 0.05) create simpler models with fewer triangles.

## Tips for Best Results

- **Z-scaling**: Set an appropriate `--z-scale` value to ensure your height features are visible
- **Base Height**: Adding a base with `--base-height` provides stability for 3D printing
- **Memory Usage**: For very large EXR files, reduce `--max-subdivisions` to save memory
- **Quality vs. Size**: Balance `--max-error` to get good quality without excessive file size
- **Color Information**: Note that only height information is used, color channels are ignored

## Common Issues

- **Memory Errors**: If you encounter memory issues, reduce `--max-subdivisions` or set a `--max-triangles` limit
- **Missing OpenEXR Support**: Ensure OpenCV is installed with OpenEXR support
