# TMD Library: TrueMap Data Processing & Visualization

The TMD Library is a comprehensive Python package for processing, analyzing, and visualizing height map data stored in TrueMap Data (TMD) files. It provides a complete toolkit for working with surface topography data across various scientific and engineering applications.

## Key Features

- **TMD File Processing**: Read and parse both TrueMap v6 and GelSight TMD file formats
- **Height Map Manipulation**: Crop, rotate, threshold, and filter height maps
- **Surface Analysis**: Calculate roughness, waviness, slope, and other surface parameters
- **Rich Visualizations**: 2D and 3D plots, cross-sections, and interactive visualizations
- **Multi-format Export**: Convert height maps to image maps, 3D models, and NumPy formats
- **Advanced Materials**: Generate complete material sets for 3D rendering and game development

## Getting Started

### Installation

```bash
pip install truemapdata
```

Or install from source:

```bash
git clone https://github.com/ETSTribology/TrueMapData
cd tmd
pip install -e .
```

### Basic Usage

```python
from tmd.processor import TMDProcessor

# Process a TMD file
processor = TMDProcessor("sample.tmd")
data = processor.process()

# Access the height map and metadata
height_map = data['height_map']
metadata = processor.get_metadata()

# Print basic statistics
stats = processor.get_stats()
print(f"Height range: {stats['min']} to {stats['max']}")
print(f"Mean height: {stats['mean']}")
```

## Core Modules

The library is organized into several key modules:

### TMD Processor
The central component for reading and processing TMD files.

```python
from tmd.processor import TMDProcessor

processor = TMDProcessor("sample.tmd")
data = processor.process()
```

### Height Map Processing
Tools for manipulating height maps:

```python
from tmd.utils.processing import crop_height_map, rotate_height_map, threshold_height_map

# Crop to region of interest
cropped = crop_height_map(height_map, region=(10, 60, 10, 60))

# Rotate by 45 degrees
rotated = rotate_height_map(height_map, angle=45)

# Threshold to remove outliers
filtered = threshold_height_map(height_map, min_height=0.1, max_height=0.9)
```

### Filtering & Analysis
Functions for surface analysis and filtering:

```python
from tmd.utils.filter import apply_gaussian_filter, calculate_rms_roughness

# Apply Gaussian smoothing
smoothed = apply_gaussian_filter(height_map, sigma=1.0)

# Calculate roughness parameters
roughness = calculate_rms_roughness(height_map)
```

### Visualization
Multiple plotting options for different needs:

```python
from tmd.plotters.matplotlib import plot_height_map_matplotlib
from tmd.plotters.plotly import plot_height_map_3d

# Create static visualization
plot_height_map_matplotlib(height_map, filename="height_map.png")

# Create interactive 3D visualization
plot_height_map_3d(height_map, filename="height_map_3d.html")
```

### Export Options
Convert height maps to various formats:

```python
# Image maps for 3D rendering
from tmd.exporters.image import convert_heightmap_to_normal_map, generate_all_maps

# Generate a normal map
normal_map = convert_heightmap_to_normal_map(height_map, filename="normal.png")

# Generate complete material set
maps = generate_all_maps(height_map, output_dir="material_maps")

# 3D models for printing or CAD
from tmd.exporters.model import convert_heightmap_to_stl

# Export as STL for 3D printing
convert_heightmap_to_stl(height_map, filename="surface.stl", z_scale=2.0)

# Export to NumPy formats
from tmd.exporters.compression import export_to_npy, export_to_npz

# Save height map as NumPy array
export_to_npy(height_map, "height_data.npy")
```

## Example Applications

- **Surface Metrology**: Analyze surface roughness and features
- **Materials Science**: Study surface topography and properties
- **Game Development**: Generate PBR material maps from real-world scans
- **3D Printing**: Convert surface scans to printable 3D models
- **Data Visualization**: Create compelling visualizations of surface data

## Documentation

For detailed API documentation and tutorials:

- [User Guide](user-guide/getting-started.md)
- [API Reference](api/exporters/image.md)
- [Architecture Overview](architecture/overview.md)

## License

This project is licensed under the MIT License. See the LICENSE file for details.
