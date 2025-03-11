# TMD - TrueMap Data Library

[![Python Version](https://img.shields.io/pypi/pyversions/tmd.svg)](https://pypi.org/project/tmd/)
[![License](https://img.shields.io/github/license/yourusername/tmd)](https://github.com/yourusername/tmd/blob/main/LICENSE)
[![Documentation Status](https://img.shields.io/readthedocs/tmd)](https://tmd.readthedocs.io/)

A Python library for processing and analyzing TrueMap Data (TMD) files used in surface topography analysis.

## Features

- **File Processing**: Read and write TMD files (v1 and v2 formats)
- **Surface Analysis**: Calculate roughness, waviness, gradients, and slopes
- **Statistical Analysis**: Extract statistical metrics from height maps
- **Visualization**: Create 2D and 3D visualizations using matplotlib
- **Export Options**: Export to STL, NumPy arrays, and image formats

## Installation

```bash
pip install tmd
```

## Quick Start

```python
from tmd.processor import TMDProcessor

# Load and process a TMD file
processor = TMDProcessor("example.tmd")
processor.process()

# Get the height map data
height_map = processor.get_height_map()

# Get statistical information
stats = processor.get_stats()
print(f"Mean height: {stats['mean']}")
print(f"Standard deviation: {stats['std']}")

# Calculate surface properties
from tmd.utils.filter import calculate_rms_roughness, calculate_rms_waviness
roughness = calculate_rms_roughness(height_map, sigma=1.0)
waviness = calculate_rms_waviness(height_map, sigma=1.0)
```

## Documentation

For comprehensive documentation including API reference and examples, visit our [Documentation Site](https://tmd.readthedocs.io/).

## Examples

### Processing a TMD File

```python
from tmd.processor import TMDProcessor

# Initialize with debug mode
processor = TMDProcessor("example.tmd").set_debug(True)

# Process the file
result = processor.process()
if result:
    # Export metadata to a text file
    processor.export_metadata("metadata.txt")
```

### Creating a 3D Visualization

```python
from tmd.plotters.matplotlib import plot_height_map_matplotlib

# Get height map from processor
height_map = processor.get_height_map()

# Create 3D plot
plot_height_map_matplotlib(
    height_map,
    colorbar_label="Height (μm)",
    filename="surface_plot.png"
)
```

### Exporting to STL for 3D Printing

```python
from tmd.exporters.stl import convert_heightmap_to_stl

# Export as STL with vertical scaling
convert_heightmap_to_stl(
    height_map,
    filename="surface_model.stl",
    z_scale=5.0  # Exaggerate height by 5x
)
```

## Contributing

Contributions are welcome! Please check out our [contribution guidelines](CONTRIBUTING.md) for more information.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
