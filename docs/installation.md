# Installation

The TMD library can be installed using pip or directly from the source code.

## Prerequisites

- Python 3.8 or higher
- NumPy
- SciPy
- Matplotlib (for visualization)

## Install via pip

```bash
pip install tmd
```

## Install from Source

Clone the repository and install the package:

```bash
git clone https://github.com/ETSTribology/TrueMapData
cd tmd
pip install -e .
```

## Development Installation

For development, install the package with development dependencies:

```bash
pip install -e ".[dev]"
```

## Verify Installation

You can verify that the installation was successful by running:

```python
import tmd
print(tmd.__version__)
```
