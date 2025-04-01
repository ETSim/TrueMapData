# TMD Visualization Guide

This guide provides examples and instructions for using the visualization capabilities of the TMD CLI. 
The TMD tool offers multiple visualization backends (matplotlib, plotly, seaborn, polyscope) and various 
visualization modes.

## Available Visualization Commands

The TMD CLI provides the following visualization subcommands:

- `basic`: Create a basic 2D visualization of a TMD file
- `3d`: Create a 3D surface visualization of a TMD file
- `profile`: Create a profile (cross-section) visualization of a TMD file
- `contour`: Create a contour plot visualization of a TMD file
- `fancy`: Create an enhanced statistical visualization (requires seaborn)
- `compare`: Compare two TMD files or multiple profiles from the same file
- `backends`: Display information about available visualization backends
- `examples`: Show example commands for all visualization types

## Basic Usage Examples

### 2D Heatmap Visualization

```bash
# Basic 2D visualization with default backend (matplotlib)
python tmd_cli.py visualize basic Dime.tmd

# 2D visualization with plotly backend and custom colormap
python tmd_cli.py visualize basic Dime.tmd --plotter plotly --colormap plasma
```

### 3D Surface Visualization

```bash
# 3D visualization with default settings
python tmd_cli.py visualize 3d Dime.tmd

# 3D visualization with increased z-scale and plotly backend
python tmd_cli.py visualize 3d Dime.tmd --z-scale 2.5 --plotter plotly

# 3D visualization with wireframe option
python tmd_cli.py visualize 3d Dime.tmd --wireframe
```

### Profile Visualization

```bash
# Profile visualization at the middle row (default)
python tmd_cli.py visualize profile Dime.tmd

# Profile at row 50 with seaborn backend
python tmd_cli.py visualize profile Dime.tmd --row 50 --plotter seaborn

# Profile with no markers and custom colormap
python tmd_cli.py visualize profile Dime.tmd --no-show-markers --colormap coolwarm
```

### Contour Visualization

```bash
# Contour plot with default settings
python tmd_cli.py visualize contour Dime.tmd

# Contour plot with custom levels
python tmd_cli.py visualize contour Dime.tmd --levels 15

# Contour plot with plotly backend
python tmd_cli.py visualize contour Dime.tmd --plotter plotly --colormap terrain
```

### Enhanced Statistical Visualizations

```bash
# Enhanced heatmap visualization
python tmd_cli.py visualize fancy Dime.tmd --plotter seaborn

# Height distribution visualization
python tmd_cli.py visualize fancy Dime.tmd --mode distribution --plotter seaborn

# Joint distribution visualization
python tmd_cli.py visualize fancy Dime.tmd --mode joint --plotter seaborn
```

### Comparison Visualizations

```bash
# Compare multiple profiles within the same file
python tmd_cli.py visualize compare Dime.tmd

# Compare two different TMD files
python tmd_cli.py visualize compare Dime.tmd --second-file Quarter.tmd

# Compare with seaborn backend
python tmd_cli.py visualize compare Dime.tmd --plotter seaborn
```

## Interactive 3D Visualization with Polyscope

The TMD CLI offers interactive 3D visualization capabilities using Polyscope, which provides real-time interaction with your height map data.

### Polyscope Surface Visualization

```bash
# Basic 3D surface with Polyscope
python tmd_cli.py visualize ps-3d Dime.tmd

# Increase Z-scale factor
python tmd_cli.py visualize ps-3d Dime.tmd --z-scale 3.0

# Show as wireframe
python tmd_cli.py visualize ps-3d Dime.tmd --wireframe
```

### Polyscope Point Cloud Visualization

```bash
# Create a point cloud visualization
python tmd_cli.py visualize ps-pointcloud Dime.tmd

# Adjust point size and sampling rate
python tmd_cli.py visualize ps-pointcloud Dime.tmd --point-size 4.0 --sample-rate 2

# Save a screenshot without showing interactive window
python tmd_cli.py visualize ps-pointcloud Dime.tmd --output dime_points.png --no-interactive
```

### Polyscope Mesh Visualization

```bash
# Create a triangle mesh visualization
python tmd_cli.py visualize ps-mesh Dime.tmd

# With wireframe and flat shading
python tmd_cli.py visualize ps-mesh Dime.tmd --wireframe --no-smooth
```

### Tips for Polyscope Visualizations

- Use your mouse to rotate, pan, and zoom in the Polyscope window
- Right-click for additional options in the viewport
- Use the GUI controls to adjust visualization settings
- Press 'a' to reset the camera view
- Press 'space' to show/hide the settings panel
- Press 'p' to take a screenshot

## Backends and Output Options

### Available Backends

- `matplotlib`: Widely compatible, supports PNG/PDF output
- `plotly`: Interactive HTML visualizations
- `seaborn`: Statistical visualization with enhanced aesthetics
- `polyscope`: 3D mesh visualization (when available)

Check available backends:

```bash
python tmd_cli.py visualize backends
```

### Output Options

Save visualizations to a file:

```bash
python tmd_cli.py visualize basic Dime.tmd --output dime_visualization.png
python tmd_cli.py visualize 3d Dime.tmd --plotter plotly --output dime_3d.html
```

Automatically open the output file:

```bash
python tmd_cli.py visualize profile Dime.tmd --output profile.png --auto-open
```

## Advanced Visualization Techniques

### Multiple Profiles

The compare command can visualize multiple profiles in the same plot:

```bash
python tmd_cli.py visualize compare Dime.tmd
```

### Distribution Analysis

Statistical visualization of height distribution:

```bash
python tmd_cli.py visualize fancy Dime.tmd --mode distribution --plotter seaborn
```

### Joint Distributions

Visualize the joint distribution of height values:

```bash
python tmd_cli.py visualize fancy Dime.tmd --mode joint --plotter seaborn
```

## Tips

- Use `--plotter auto` to automatically select the best available backend
- When using plotly, save with `.html` extension for interactive visualizations
- Seaborn provides the best statistical visualizations 
- For large datasets, consider using matplotlib for better performance
