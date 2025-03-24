# Compression Exporters

The TMD compression exporters module provides utilities for exporting heightmap data and metadata to efficient binary formats for storage or exchange.

## Supported Formats

The module supports the following formats:

- **NPY**: NumPy's native binary format for single arrays
- **NPZ**: NumPy's compressed archive format for multiple arrays with metadata

## Functions

### NPY Format

::: tmd.exporters.compression.export_to_npy
::: tmd.exporters.compression.load_from_npy

### NPZ Format

::: tmd.exporters.compression.export_to_npz
::: tmd.exporters.compression.load_from_npz

## Usage Examples

### Basic NPY Export

```python
from tmd.exporters.compression import export_to_npy, load_from_npy

# Export just the height map array
export_to_npy(
    height_map,
    output_path="heightmap.npy"
)

# Load the height map
loaded_height_map = load_from_npy("heightmap.npy")
```

### Compressed NPZ Export with Metadata

```python
from tmd.exporters.compression import export_to_npz, load_from_npz

# Create data dictionary with height map and metadata
data = {
    "height_map": height_map,
    "units": "mm",
    "x_length": 10.0,
    "y_length": 10.0,
    "source": "laser_scan",
    "timestamp": "2023-06-15T14:30:00"
}

# Export the complete data (height map + metadata)
export_to_npz(
    data,
    output_path="terrain_with_metadata.npz",
    compress=True  # Use compression to reduce file size
)

# Load the complete data
loaded_data = load_from_npz("terrain_with_metadata.npz")

# Access individual components
height_map = loaded_data["height_map"]
units = loaded_data["units"]
```

## Performance Considerations

- NPY format is faster to load/save but stores only the height map without metadata
- NPZ format is slightly slower but can store both height map and metadata
- Compression reduces file size but adds a small performance overhead
- For large datasets (>1000x1000), the compressed NPZ format offers the best balance of size and loading speed

## Implementation Notes

The module converts complex metadata types to JSON-compatible formats under the hood, ensuring that:

1. All metadata can be reliably recovered when loading
2. Binary files remain portable across different systems
3. NumPy data types are preserved exactly
