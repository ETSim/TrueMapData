#!/usr/bin/env python3
"""
Example commands module for TMD CLI.

This module provides example usage patterns for the TMD CLI tools.
"""

from rich.console import Console
from rich.markdown import Markdown

from tmd.cli.core import print_success

console = Console()

EXAMPLES_MD = """
# TMD Command-Line Tool Examples

## Basic Information

```bash
# Display help
python tmd_cli.py --help

# Show information about a TMD file
python tmd_cli.py info path/to/file.tmd

# Show version information
python tmd_cli.py version
```

## Compression

```bash
# Downsample a TMD file to 50% of its original size
python tmd_cli.py compress downsample path/to/file.tmd --scale 0.5

# Quantize height values to 256 levels
python tmd_cli.py compress quantize path/to/file.tmd --levels 256

# Combine downsampling and quantization
python tmd_cli.py compress combined path/to/file.tmd --scale 0.5 --levels 256

# Batch compression of multiple files
python tmd_cli.py compress batch tmd_files/ --mode downsample --scale 0.5 --recursive
```

## Visualization

```bash
# Basic 2D visualization with default settings
python tmd_cli.py visualize basic path/to/file.tmd

# 2D visualization with custom colormap
python tmd_cli.py visualize basic path/to/file.tmd --colormap viridis

# 3D visualization with plotly
python tmd_cli.py visualize 3d path/to/file.tmd --z-scale 2.0 --plotter plotly

# 3D visualization with matplotlib
python tmd_cli.py visualize 3d path/to/file.tmd --z-scale 1.5 --plotter matplotlib

# Height profile visualization 
python tmd_cli.py visualize profile path/to/file.tmd --row 50

# Height profile with seaborn
python tmd_cli.py visualize profile path/to/file.tmd --row 75 --plotter seaborn

# Interactive 3D visualization with Polyscope
python tmd_cli.py visualize ps-3d path/to/file.tmd --z-scale 2.0

# Point cloud visualization with Polyscope
python tmd_cli.py visualize ps-pointcloud path/to/file.tmd --sample-rate 2 --point-size 3.0

# Triangle mesh visualization with Polyscope
python tmd_cli.py visualize ps-mesh path/to/file.tmd --wireframe --smooth

# Check available visualization backends
python tmd_cli.py visualize backends

# Full markdown examples (all major CLI areas)
python tmd_cli.py visualize examples
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

## Mesh export

Use `tmd-process` instead of `python tmd_cli.py` if you installed the package from PyPI.

```bash
# Supported destination formats
python tmd_cli.py mesh formats

# Convenience exporters (quality preset controls error_threshold / max_triangles)
python tmd_cli.py mesh stl path/to/terrain.tmd --scale 5.0 --quality high
python tmd_cli.py mesh stl path/to/terrain.tmd --binary false
python tmd_cli.py mesh obj path/to/terrain.tmd --error-threshold 0.02 --max-triangles 80000
python tmd_cli.py mesh ply path/to/terrain.tmd --binary true
python tmd_cli.py mesh gltf path/to/terrain.tmd --binary true
python tmd_cli.py mesh usd path/to/terrain.tmd --binary true

# Full `mesh generate` (method: adaptive or quadtree; extra knobs on CLI)
python tmd_cli.py mesh generate path/to/terrain.tmd --format stl --method adaptive --quality high
python tmd_cli.py mesh generate path/to/terrain.tmd --format stl --method quadtree --max-subdivisions 12 --max-triangles 120000

python tmd_cli.py mesh stl --help
python tmd_cli.py mesh generate --help
```
"""

def show_examples():
    """Display comprehensive usage examples for TMD CLI."""
    print_success("TMD Command-Line Tool Examples:")
    
    md = Markdown(EXAMPLES_MD)
    console.print(md)
    
    print_success("\nSee the documentation for more detailed information and examples.")
