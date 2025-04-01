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
python tmd_cli.py visualize polyscope Dime.tmd --wireframe

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
"""

def show_examples():
    """Display comprehensive usage examples for TMD CLI."""
    print_success("TMD Command-Line Tool Examples:")
    
    md = Markdown(EXAMPLES_MD)
    console.print(md)
    
    print_success("\nSee the documentation for more detailed information and examples.")
