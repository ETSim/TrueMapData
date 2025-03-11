# Getting Started with TMD

This guide will help you get started with processing TMD files and analyzing height map data.

## Basic Concepts

TMD (TrueMap Data) files contain height map information for surface topography analysis. The key components are:

- **Height Map**: A 2D array of height values
- **Metadata**: Information about the dimensions, scale, and origin of the height map

## Processing Your First TMD File

=== "Using the Processor"

    ```python
    from tmd.processor import TMDProcessor
    
    # Initialize processor with your TMD file
    processor = TMDProcessor("path/to/your/file.tmd")
    
    # Process the file
    processor.process()
    
    # Print basic information
    print(processor.get_metadata())
    ```

=== "Using Low-level Functions"

    ```python
    from tmd.utils.utils import process_tmd_file
    
    # Process the file directly
    metadata, height_map = process_tmd_file("path/to/your/file.tmd")
    
    # Print metadata
    print(metadata)
    ```

## Basic Analysis

Once you have processed a TMD file, you can analyze the height map:

```python
# Get the height map
height_map = processor.get_height_map()

# Get basic statistics
stats = processor.get_stats()
print(f"Min height: {stats['min']}")
print(f"Max height: {stats['max']}")
print(f"Mean height: {stats['mean']}")
```

## Visualizing the Height Map

You can visualize the height map using the built-in plotting functions:

```python
from tmd.plotters.matplotlib import plot_height_map_matplotlib

# Create a 3D surface plot
plot_height_map_matplotlib(
    height_map, 
    colorbar_label="Height (μm)",
    filename="height_map.png"
)

# Create a 2D heatmap
from tmd.plotters.matplotlib import plot_2d_heatmap_matplotlib
plot_2d_heatmap_matplotlib(
    height_map,
    filename="heatmap.png"
)
```

## Next Steps

- Learn about [filtering](../api/filter.md) to separate waviness from roughness
- Explore different [export options](exporting.md) for your data
- Check out the [examples](../examples/basic-processing.md) for more advanced usage
