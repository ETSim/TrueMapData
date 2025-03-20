# Visualization Modules

The TMD Library provides multiple visualization options through separate modules, each offering different visualization capabilities through popular Python plotting libraries.

## Overview

The visualization modules include:

- **Matplotlib Plotter**: Static 2D/3D plots (available in all installations)
- **Plotly Plotter**: Interactive 3D visualizations (requires plotly)
- **Seaborn Plotter**: Statistical visualizations (requires seaborn)

## Matplotlib Plotter

The most widely-compatible visualization module, providing static plots with matplotlib.

::: tmd.plotters.matplotlib.plot_height_map_matplotlib

::: tmd.plotters.matplotlib.plot_2d_heatmap_matplotlib 

::: tmd.plotters.matplotlib.plot_x_profile_matplotlib

### Examples

```python
from tmd.plotters.matplotlib import (
    plot_height_map_matplotlib,
    plot_2d_heatmap_matplotlib,
    plot_x_profile_matplotlib
)

# 3D surface plot
plot_height_map_matplotlib(
    height_map,
    colorbar_label="Height (µm)",
    filename="3d_surface.png"
)

# 2D heatmap
plot_2d_heatmap_matplotlib(
    height_map,
    colorbar_label="Height (µm)",
    filename="heatmap.png"
)

# Cross-section profile plot
metadata = {
    'height_map': height_map,
    'width': height_map.shape[1],
    'x_offset': 0.0,
    'x_length': 10.0
}
x_coords, x_heights, fig = plot_x_profile_matplotlib(
    metadata,
    profile_row=height_map.shape[0] // 2,
    filename="profile.png"
)
```

## Plotly Plotter

Interactive 3D visualizations that can be viewed in web browsers and Jupyter notebooks.

::: tmd.plotters.plotly.plot_height_map_3d

::: tmd.plotters.plotly.plot_cross_section_plotly

### Examples

```python
from tmd.plotters.plotly import plot_height_map_3d, plot_cross_section_plotly

# Interactive 3D surface
plot_height_map_3d(
    height_map,
    title="Surface Topography",
    colorscale="Viridis",
    filename="interactive_surface.html"
)

# Interactive cross-section
plot_cross_section_plotly(
    x_positions,  # From extract_cross_section function
    heights,      # From extract_cross_section function
    title="Surface Profile",
    filename="interactive_profile.html"
)
```

## Comparing Visualization Options

| Feature | Matplotlib | Plotly | 
|---------|------------|--------|
| **Type** | Static | Interactive |
| **Output** | PNG, PDF, SVG | HTML, Jupyter |
| **3D Support** | Basic | Advanced |
| **Interactivity** | Limited | Full (zoom, rotate, etc.) |
| **Dependencies** | Minimal | Additional |
| **Sharing** | Image files | Interactive web pages |

## Visualization Workflows

### Basic Analysis Workflow

```python
from tmd.processor import TMDProcessor
from tmd.plotters.matplotlib import plot_height_map_matplotlib, plot_2d_heatmap_matplotlib
from tmd.utils.processing import extract_cross_section
import matplotlib.pyplot as plt
import os

def analyze_tmd(file_path, output_dir="."):
    """Basic analysis workflow for TMD files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Process file
    processor = TMDProcessor(file_path)
    data = processor.process()
    height_map = data['height_map']
    
    # 3D visualization
    plot_height_map_matplotlib(
        height_map,
        colorbar_label="Height (µm)",
        filename=os.path.join(output_dir, "3d_surface.png")
    )
    
    # 2D heatmap
    plot_2d_heatmap_matplotlib(
        height_map,
        colorbar_label="Height (µm)",
        filename=os.path.join(output_dir, "heatmap.png")
    )
    
    # Extract and plot cross-sections
    row_pos = height_map.shape[0] // 2
    col_pos = height_map.shape[1] // 2
    
    x_pos, x_heights = extract_cross_section(height_map, data, axis='x', position=row_pos)
    y_pos, y_heights = extract_cross_section(height_map, data, axis='y', position=col_pos)
    
    # Create cross-section plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(x_pos, x_heights)
    ax1.set_title(f"X Cross-section at row {row_pos}")
    ax1.set_xlabel("X Position")
    ax1.set_ylabel("Height")
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(y_pos, y_heights)
    ax2.set_title(f"Y Cross-section at column {col_pos}")
    ax2.set_xlabel("Y Position")
    ax2.set_ylabel("Height")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cross_sections.png"), dpi=300)
    plt.close()
    
    print(f"Analysis complete. Results saved to {output_dir}")
    return data

# Example usage
analyze_tmd("sample.tmd", "output/analysis")
```

### Web-Ready Interactive Visualization

```python
from tmd.processor import TMDProcessor
from tmd.plotters.plotly import plot_height_map_3d
import os

def create_web_visualization(tmd_files, output_dir):
    """Create web-ready visualizations for multiple TMD files."""
    os.makedirs(output_dir, exist_ok=True)
    
    for file_path in tmd_files:
        try:
            # Extract filename
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            
            # Process TMD file
            processor = TMDProcessor(file_path)
            data = processor.process()
            
            if data is None:
                print(f"Failed to process {file_path}")
                continue
                
            height_map = data['height_map']
            
            # Create interactive visualization
            html_path = os.path.join(output_dir, f"{file_name}.html")
            plot_height_map_3d(
                height_map,
                title=f"Surface: {file_name}",
                colorscale="Viridis",
                filename=html_path
            )
            
            print(f"Created interactive visualization: {html_path}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    # Create index.html to link all visualizations
    with open(os.path.join(output_dir, "index.html"), "w") as f:
        f.write("<html><head><title>TMD Visualizations</title></head><body>\n")
        f.write("<h1>TMD Surface Visualizations</h1>\n<ul>\n")
        
        for file_path in tmd_files:
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            f.write(f'  <li><a href="{file_name}.html">{file_name}</a></li>\n')
            
        f.write("</ul></body></html>")
    
    print(f"Created visualization index at {os.path.join(output_dir, 'index.html')}")

# Example usage
tmd_files = [
    "examples/v2/Dime.tmd",
    "examples/v2/StepHeight.tmd",
    "examples/v2/Surface.tmd"
]
create_web_visualization(tmd_files, "output/web_visualizations")
```
