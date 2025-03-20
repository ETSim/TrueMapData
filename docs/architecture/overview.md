# TMD Library Architecture Overview

This document provides a high-level overview of the TMD library's architecture, explaining the core design principles and components.

## Design Philosophy

The TMD library is designed with the following principles in mind:

1. **Modularity**: Separate components with clear responsibilities
2. **Extensibility**: Easy to add new functionality without modifying existing code
3. **Usability**: Simple, intuitive API for common operations
4. **Performance**: Efficient handling of large height map data

## Core Components

The TMD library is organized into several core components:

### 1. TMD Processor

The central component that handles loading and parsing TMD files. It serves as the entry point for most operations and manages the extraction of height maps and metadata.

```python
from tmd.processor import TMDProcessor

processor = TMDProcessor("example.tmd")
data = processor.process()
```

### 2. Utility Modules

A collection of utility modules that provide core functionality:

- **Processing**: Functions for manipulating height maps (crop, rotate, threshold)
- **Filter**: Functions for filtering and analyzing height maps
- **Metadata**: Tools for handling metadata extraction and storage

### 3. Exporters

Modules for exporting height maps to various formats:

- **Image Exporter**: Converts to image formats (PNG, normal maps, displacement maps)
- **3D Model Exporter**: Exports to 3D model formats (STL, OBJ, PLY)
- **Compression Exporter**: Saves to NumPy formats (NPY, NPZ)

### 4. Visualizers

Components for creating visualizations:

- **Matplotlib Plotter**: Static 2D/3D visualizations
- **Plotly Plotter**: Interactive 3D visualizations in web browsers
- **Seaborn Plotter**: Statistical visualizations

## Data Flow

The typical data flow in the TMD library:

1. A TMD file is loaded by the `TMDProcessor`
2. The processor extracts the height map and metadata
3. The height map can be processed using utility functions
4. The processed data can be visualized or exported

## Integration Points

The library is designed to integrate well with:

- **NumPy/SciPy** ecosystem for scientific computing
- **3D modeling** software via STL, OBJ exports
- **Game engines** via material map generation
- **Jupyter notebooks** for interactive analysis

## Extensibility

New functionality can be added by:

1. Creating new processing functions in the utility modules
2. Adding new exporters for additional file formats
3. Implementing new visualization methods
4. Extending the processor to handle additional file formats

For more detailed information about the components and their relationships, see the [Component Diagram](component-diagram.md) and [Data Flow](data-flow.md) documentation.
