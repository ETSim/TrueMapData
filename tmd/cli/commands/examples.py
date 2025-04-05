#!/usr/bin/env python3
"""
Example commands module for TMD CLI.

This module provides example usage patterns for the TMD CLI tools.
"""

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax

from tmd.cli.core import print_success

console = Console()

EXAMPLES_MD = """
# TMD Command-Line Tool Examples

## Basic Information

```bash
# Display help
python tmd_cli.py --help

# Show information about a TMD file
python tmd_cli.py info Dime.tmd

# Show version information
python tmd_cli.py version
```

## Compression

```bash
# Downsample a TMD file to 50% of its original size
python tmd_cli.py compress downsample Dime.tmd --scale 0.5

# Quantize height values to 256 levels
python tmd_cli.py compress quantize Dime.tmd --levels 256

# Combine downsampling and quantization
python tmd_cli.py compress combined Dime.tmd --scale 0.5 --levels 256

# Batch compression of multiple files
python tmd_cli.py compress batch tmd_files/ --mode downsample --scale 0.5 --recursive
```

## Visualization

```bash
# Basic 2D visualization with default settings
python tmd_cli.py visualize basic Dime.tmd

# 2D visualization with custom colormap
python tmd_cli.py visualize basic Dime.tmd --colormap viridis

# 3D visualization with plotly
python tmd_cli.py visualize 3d Dime.tmd --z-scale 2.0 --plotter plotly

# 3D visualization with matplotlib
python tmd_cli.py visualize 3d Dime.tmd --z-scale 1.5 --plotter matplotlib

# Height profile visualization 
python tmd_cli.py visualize profile Dime.tmd --row 50

# Height profile with seaborn
python tmd_cli.py visualize profile Dime.tmd --row 75 --plotter seaborn

# Interactive 3D visualization with Polyscope
python tmd_cli.py visualize ps-3d Dime.tmd --z-scale 2.0

# Point cloud visualization with Polyscope
python tmd_cli.py visualize ps-pointcloud Dime.tmd --sample-rate 2 --point-size 3.0

# Triangle mesh visualization with Polyscope
python tmd_cli.py visualize ps-mesh Dime.tmd --wireframe --smooth

# Create animation from multiple TMD files
python tmd_cli.py visualize polyscope-animate tmd_sequence/*.tmd --fps 30

# Check available visualization backends
python tmd_cli.py visualize plotters

# Check Polyscope installation
python tmd_cli.py visualize check-polyscope
```

## Cache Management

```bash
# Get cache information
python tmd_cli.py cache info

# Clear expired cache entries
python tmd_cli.py cache clear

# Clear the entire cache
python tmd_cli.py cache clear-all
```

## Configuration

```bash
# Show current configuration
python tmd_cli.py config show

# Set default plotter
python tmd_cli.py config set default_plotter matplotlib

# Reset configuration to defaults
python tmd_cli.py config reset
```

## Map Export Examples

```bash
# List available map types
python tmd_cli.py maps list

# Export individual maps
python tmd_cli.py maps ao input.tmd --samples 32 --strength 1.5
python tmd_cli.py maps bump input.tmd --scale 2.0
python tmd_cli.py maps roughness input.tmd --kernel-size 5
python tmd_cli.py maps metallic input.tmd --threshold 0.5
python tmd_cli.py maps displacement input.tmd --scale 1.0
python tmd_cli.py maps height input.tmd --normalize
python tmd_cli.py maps hillshade input.tmd --azimuth 315 --altitude 45

# Export all maps
python tmd_cli.py maps all input.tmd --output-dir textures/

# Export specific maps
python tmd_cli.py maps all input.tmd --output-dir textures/ --types ao bump roughness

# Export with options
python tmd_cli.py maps all input.tmd --output-dir textures/ \\
    --ao-samples 32 \\
    --ao-strength 1.5 \\
    --bump-scale 2.0 \\
    --roughness-kernel 5 \\
    --metallic-threshold 0.5 \\
    --displacement-scale 1.0 \\
    --height-normalize true \\
    --hillshade-azimuth 315
```

## Mesh Export Commands

### List Available Formats
```bash
# Show all supported mesh export formats
python tmd_cli.py mesh list
```

### STL Export (Stereolithography)
```bash
# Export as binary STL (default, recommended for most use cases)
python tmd_cli.py mesh stl terrain.tmd --scale 1.0

# Export as ASCII STL (larger file, human-readable)
python tmd_cli.py mesh stl terrain.tmd --binary false

# Export with custom scaling and output location
python tmd_cli.py mesh stl terrain.tmd --scale 2.5 --output-file models/terrain.stl
```

### OBJ Export (Wavefront)
```bash
# Basic OBJ export (includes materials and textures)
python tmd_cli.py mesh obj terrain.tmd

# Export with increased detail and custom scale
python tmd_cli.py mesh obj terrain.tmd --scale 1.5 --output-file terrain_detailed.obj

# Export with automatic texture generation
python tmd_cli.py mesh obj terrain.tmd --texture true --texture-resolution 2048,2048
```

### PLY Export (Stanford Triangle Format)
```bash
# Export as binary PLY (default, compact file size)
python tmd_cli.py mesh ply terrain.tmd

# Export as ASCII PLY with vertex colors
python tmd_cli.py mesh ply terrain.tmd --binary false --vertex-colors true

# Export with normal vectors and custom scale
python tmd_cli.py mesh ply terrain.tmd --scale 2.0 --normals true
```

### GLTF Export (GL Transmission Format)
```bash
# Export as binary GLB (default, recommended for web/mobile)
python tmd_cli.py mesh gltf terrain.tmd

# Export as JSON-based GLTF with external resources
python tmd_cli.py mesh gltf terrain.tmd --binary false

# Export with PBR materials and high-resolution textures
python tmd_cli.py mesh gltf terrain.tmd --pbr true --texture-resolution 4096,4096
```

### USD Export (Universal Scene Description)
```bash
# Export as binary USDC (default)
python tmd_cli.py mesh usd terrain.tmd

# Export as ASCII USDA (for debugging/editing)
python tmd_cli.py mesh usd terrain.tmd --binary false

# Export as USDZ (for AR Quick Look on iOS)
python tmd_cli.py mesh usd terrain.tmd --output-file terrain.usdz

# Export with high-quality materials for visualization
python tmd_cli.py mesh usd terrain.tmd --pbr true --texture true
```

### Advanced Export Options

#### Scale and Resolution Control
```bash
# Export with vertical exaggeration
python tmd_cli.py mesh stl terrain.tmd --scale 2.5

# Export with high detail preservation
python tmd_cli.py mesh obj terrain.tmd --error-threshold 0.001

# Export with reduced polygon count
python tmd_cli.py mesh ply terrain.tmd --max-triangles 100000
```

#### Material and Texture Options
```bash
# Generate PBR materials with all maps
python tmd_cli.py mesh gltf terrain.tmd --pbr true --maps ao,normal,roughness

# Custom texture resolution
python tmd_cli.py mesh obj terrain.tmd --texture-resolution 2048,2048

# Specific color scheme for textures
python tmd_cli.py mesh usd terrain.tmd --color-map terrain
```

#### Coordinate System Options
```bash
# Export with origin at center
python tmd_cli.py mesh stl terrain.tmd --origin-at-zero true

# Export with specific coordinate system
python tmd_cli.py mesh obj terrain.tmd --coordinate-system right-handed

# Export with custom base height
python tmd_cli.py mesh ply terrain.tmd --base-height 10
```

For more details on each format and its options, use:
```bash
python tmd_cli.py mesh <format> --help
```

## Compression Commands

```bash
# Downsample a TMD file to 50% of its original size
python tmd_cli.py compress downsample Dime.tmd --scale 0.5

# Quantize height values to 256 levels
python tmd_cli.py compress quantize Dime.tmd --levels 256

# Combine downsampling and quantization
python tmd_cli.py compress combined Dime.tmd --scale 0.5 --levels 256

# Batch compression of multiple files
python tmd_cli.py compress batch tmd_files/ --mode downsample --scale 0.5 --recursive
```
"""

def show_examples():
    """Display comprehensive usage examples for TMD CLI."""
    print_success("TMD Command-Line Tool Examples:")
    
    md = Markdown(EXAMPLES_MD)
    console.print(md)
    
    print_success("\nSee the documentation for more detailed information and examples.")
