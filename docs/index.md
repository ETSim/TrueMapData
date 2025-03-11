# TMD - TrueMap Data Library

The TMD library is a Python package for processing and analyzing TrueMap Data files (TMD format), a proprietary format for storing surface topography data.

## Features

- **File Processing**: Read and write TMD files (both v1 and v2 formats)
- **Data Analysis**: Extract statistical information from height maps
- **Filtering**: Apply Gaussian filters to separate waviness and roughness components
- **Surface Analysis**: Calculate gradients, slopes, and RMS values
- **Visualization**: Create 2D and 3D visualizations of height maps
- **Export**: Export data to STL for 3D printing, NumPy arrays, or image formats

## Quick Start

```python
from tmd.processor import TMDProcessor

# Initialize processor with a TMD file
processor = TMDProcessor("path/to/file.tmd")

# Process the file
processor.process()

# Get statistical information
stats = processor.get_stats()
print(f"Min height: {stats['min']}, Max height: {stats['max']}")

# Get the height map as a NumPy array
height_map = processor.get_height_map()

# Analyze surface characteristics
from tmd.utils.filter import calculate_rms_roughness
roughness = calculate_rms_roughness(height_map, sigma=1.0)
print(f"RMS roughness: {roughness}")
```

## Installation

```bash
pip install tmd
```

For more details, see the [Installation Guide](installation.md).
