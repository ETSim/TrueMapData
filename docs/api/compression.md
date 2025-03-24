# Compression & NumPy Export

The Compression module provides functions for saving height maps and TMD data to various NumPy-based file formats.

## Overview

This module enables exporting height maps and metadata to NumPy's efficient binary formats:

- **NPY**: Simple format for saving single NumPy arrays
- **NPZ**: Compressed format for saving multiple arrays and data structures

## Export Functions

::: tmd.exporters.compression.export_to_npy

::: tmd.exporters.compression.export_to_npz

## Examples

### Basic NPY Export

```python
from tmd.exporters.compression import export_to_npy

# Export height map to NPY file
filepath = export_to_npy(height_map, "height_data.npy")
print(f"Height map saved to {filepath}")

# Load the saved data
import numpy as np
loaded_height_map = np.load("height_data.npy")
```

### NPZ Export with Metadata

```python
from tmd.exporters.compression import export_to_npz

# Export complete TMD data including metadata
tmd_data = {
    'height_map': height_map,
    'width': height_map.shape[1],
    'height': height_map.shape[0],
    'x_length': 10.0,
    'y_length': 10.0,
    'comment': 'Sample height map'
}

filepath = export_to_npz(tmd_data, "complete_data.npz")
print(f"TMD data saved to {filepath}")

# Load the saved data
import numpy as np
loaded_data = np.load("complete_data.npz")

# Access individual components
loaded_height_map = loaded_data['height_map']
x_length = loaded_data['x_length']
```

### Selective NPZ Export

```python
from tmd.exporters.compression import export_to_npz

# Export only specific components
selected_data = {
    'height_map': height_map,
    'roughness_map': roughness_map,
    'normal_map': normal_map
}

filepath = export_to_npz(selected_data, "maps_collection.npz")
```

## Integration with Scientific Workflows

The NumPy formats are ideal for scientific workflows and further processing:

```python
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from tmd.exporters.compression import export_to_npy, export_to_npz

# Process height map
smoothed = ndimage.gaussian_filter(height_map, sigma=1.0)

# Save processed data
export_to_npy(smoothed, "processed_height_map.npy")

# Load for analysis in another script
data = np.load("processed_height_map.npy")

# Visualization
plt.figure(figsize=(10, 8))
plt.imshow(data, cmap='viridis')
plt.colorbar(label='Height')
plt.title('Processed Height Map')
plt.savefig('analysis_result.png', dpi=300)
```

## File Size Comparison

| Format | Typical Size | Features |
|--------|--------------|----------|
| NPY    | Medium       | Fast access, single array |
| NPZ    | Small        | Compressed, multiple arrays |
| TMD    | Medium       | Original format with header |
| STL    | Large        | 3D model format |

## Workflows: Data Archiving

For long-term data storage or dataset creation:

1. **Process and prepare data**:

   ```python
   from tmd.processor import TMDProcessor
   from tmd.utils.filter import apply_gaussian_filter

   processor = TMDProcessor("sample.tmd")
   data = processor.process()
   height_map = data['height_map']

   # Process if needed
   filtered_map = apply_gaussian_filter(height_map, sigma=0.5)
   ```

2. **Save complete dataset**:

   ```python
   from tmd.exporters.compression import export_to_npz

   # Combine original and processed data
   archive_data = {
       'original': height_map,
       'filtered': filtered_map,
       'metadata': {k: v for k, v in data.items() if k != 'height_map'}
   }

   # Save in compressed format
   export_to_npz(archive_data, "dataset_001.npz")
   ```

3. **Document the archive**:

   ```python
   # Create a simple readme
   with open("dataset_001_readme.txt", "w") as f:
       f.write("Dataset: Sample 001\n")
       f.write("Contents:\n")
       f.write("  - original: Raw height map\n")
       f.write("  - filtered: Gaussian filtered (sigma=0.5)\n")
       f.write("  - metadata: Original TMD metadata\n")
       f.write(f"Date: {datetime.now().strftime('%Y-%m-%d')}\n")
   ```
