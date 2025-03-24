# Installation Guide

This guide provides detailed instructions for installing the TMD library with all its dependencies.

## Requirements

### Minimum Requirements

- Python 3.8 or higher
- NumPy
- Matplotlib
- Pillow (PIL)
- OpenCV (cv2)

### Recommended Setup

For the full functionality, the following additional libraries are recommended:

- Plotly (for interactive visualizations)
- Meshio (for advanced 3D model exports)
- SciPy (for advanced filtering)

## Installation Methods

### Method 1: PyPI Installation (Recommended)

The simplest way to install the TMD library is using pip:

```bash
# Basic installation
pip install truemapdata

# With visualization dependencies
pip install truemapdata[viz]

# Complete installation with all optional dependencies
pip install truemapdata[full]
```

### Method 2: From Source

To install from source (e.g., for development):

```bash
# Clone the repository
git clone https://github.com/ETSTribology/TrueMapData
cd tmd

# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt
```

## Environment Setup

### Using Virtual Environments

It's recommended to use a virtual environment:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install the library
pip install truemapdata
```

## Verifying Installation

To verify your installation:

```python
import tmd
from tmd.processor import TMDProcessor

# Should print the version number
print(tmd.__version__)

# Test core functionality
processor = TMDProcessor("path/to/example.tmd")
# If no errors, installation is successful
```

## Optional Dependencies

### Interactive Visualization

For interactive 3D visualizations:

```bash
pip install plotly
```

### 3D Model Export

For advanced 3D model export capabilities:

```bash
pip install meshio
```

### Documentation

To build the documentation:

```bash
pip install -r requirements-docs.txt
mkdocs build
```

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure all dependencies are installed correctly

   ```bash
   pip install --upgrade truemapdata[full]
   ```

2. **OpenCV installation issues**: On some systems, you may need to install OpenCV separately:

   ```bash
   # On Debian/Ubuntu
   sudo apt-get install python3-opencv

   # Or with pip
   pip install opencv-python
   ```

3. **File permission errors**: When saving files, ensure the output directory is writable:

   ```python
   import os
   os.makedirs("output", exist_ok=True)
   ```

### Getting Help

If you encounter issues:

1. Check the documentation at <https://yourusername.github.io/tmd/>
2. Open an issue on GitHub
3. Contact the maintainers

## Upgrading

To upgrade to the latest version:

```bash
pip install --upgrade truemapdata
```

For development installations:

```bash
git pull
pip install -e .
```
