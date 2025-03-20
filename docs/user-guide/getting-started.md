# Getting Started with TMD

This guide will walk you through your first steps with the TMD library, from processing a TMD file to basic analysis and visualization.

## Quick Start

Let's start with a simple example to process a TMD file and visualize the height map:

```python
from tmd.processor import TMDProcessor
import matplotlib.pyplot as plt

# Process a TMD file
processor = TMDProcessor("examples/v2/Dime.tmd")
data = processor.process()

# Extract the height map
height_map = data['height_map']

# Visualize the height map
plt.figure(figsize=(10, 8))
plt.imshow(height_map, cmap='viridis')
plt.colorbar(label='Height')
plt.title('TMD Height Map')
plt.show()

# Print basic statistics
stats = processor.get_stats()
print(f"Height range: {stats['min']} to {stats['max']}")
print(f"Mean height: {stats['mean']}")
```

## Core Workflow

The typical TMD workflow follows these steps:

1. **Process the TMD file** to extract the height map and metadata
2. **Analyze and manipulate** the height map as needed
3. **Export or visualize** the results

### Step 1: Process a TMD File

```python
from tmd.processor import TMDProcessor

# Initialize with file path
processor = TMDProcessor("path/to/your/file.tmd")

# Process the file
data = processor.process()

# Access the height map and metadata
height_map = data['height_map']
metadata = {k: v for k, v in data.items() if k != 'height_map'}

# Print metadata
for key, value in metadata.items():
    print(f"{key}: {value}")
```

### Step 2: Analyze and Manipulate

```python
from tmd.utils.filter import apply_gaussian_filter, calculate_rms_roughness
from tmd.utils.processing import crop_height_map, rotate_height_map

# Apply Gaussian filter for smoothing
smoothed_map = apply_gaussian_filter(height_map, sigma=1.0)

# Calculate roughness
roughness = calculate_rms_roughness(height_map)
print(f"RMS Roughness: {roughness}")

# Crop a region of interest (row_start, row_end, col_start, col_end)
region = (50, 150, 50, 150)
cropped_map = crop_height_map(height_map, region)

# Rotate the height map
rotated_map = rotate_height_map(height_map, angle=45)
```

### Step 3: Export or Visualize

```python
from tmd.exporters.image import generate_all_maps
from tmd.exporters.model import convert_heightmap_to_stl
import os

# Create output directory
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Generate image maps
maps = generate_all_maps(height_map, output_dir=output_dir)

# Export to STL for 3D printing
convert_heightmap_to_stl(
    height_map,
    filename=os.path.join(output_dir, "model.stl"),
    z_scale=2.0  # Exaggerate height for better visibility
)

# Create a visualization
from tmd.plotters.matplotlib import plot_height_map_matplotlib

plot_height_map_matplotlib(
    height_map,
    colorbar_label="Height (µm)",
    filename=os.path.join(output_dir, "3d_surface.png")
)
```

## Working with TMD Files

### Supported File Formats

The TMD library supports both v1 and v2 TMD file formats:

```python
from tmd.processor import TMDProcessor
from tmd.utils.utils import detect_tmd_version

# Detect file version
version = detect_tmd_version("path/to/file.tmd")
print(f"File is TMD version {version}")

# Process based on version
processor = TMDProcessor("path/to/file.tmd")
processor.set_debug(True)  # Enable debug output
data = processor.process()
```

### Creating TMD Files

You can also create synthetic TMD files:

```python
from tmd.utils.utils import generate_synthetic_tmd, create_sample_height_map

# Generate a sample height map
height_map = create_sample_height_map(
    width=200,
    height=200,
    pattern="combined",  # Options: "waves", "peak", "dome", "ramp", "combined"
    noise_level=0.05
)

# Create a TMD file
tmd_file = generate_synthetic_tmd(
    output_path="synthetic.tmd",
    width=200,
    height=200,
    pattern="combined",
    comment="Synthetic TMD",
    version=2
)

print(f"Generated TMD file: {tmd_file}")
```

## Complete Example: Surface Analysis

```python
from tmd.processor import TMDProcessor
from tmd.utils.filter import apply_gaussian_filter, calculate_rms_roughness
from tmd.utils.processing import extract_cross_section
from tmd.exporters.image import generate_all_maps
import matplotlib.pyplot as plt
import os
import numpy as np

# Create output directory
output_dir = "analysis_output"
os.makedirs(output_dir, exist_ok=True)

# Process TMD file
processor = TMDProcessor("examples/v2/Surface.tmd")
data = processor.process()
height_map = data['height_map']

# Export metadata
processor.export_metadata(os.path.join(output_dir, "metadata.txt"))

# Apply Gaussian filter
filtered_map = apply_gaussian_filter(height_map, sigma=1.0)

# Calculate roughness before and after filtering
original_roughness = calculate_rms_roughness(height_map)
filtered_roughness = calculate_rms_roughness(filtered_map)

print(f"Original RMS Roughness: {original_roughness:.4f}")
print(f"Filtered RMS Roughness: {filtered_roughness:.4f}")
print(f"Roughness reduction: {100 * (original_roughness - filtered_roughness) / original_roughness:.2f}%")

# Extract cross-section
mid_row = height_map.shape[0] // 2
x_pos, x_heights = extract_cross_section(height_map, data, axis='x', position=mid_row)
x_pos_f, x_heights_f = extract_cross_section(filtered_map, data, axis='x', position=mid_row)

# Plot cross-sections
plt.figure(figsize=(10, 6))
plt.plot(x_pos, x_heights, 'b-', label='Original')
plt.plot(x_pos_f, x_heights_f, 'r-', label='Filtered')
plt.title(f'Cross-section at Row {mid_row}')
plt.xlabel('X Position')
plt.ylabel('Height')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(output_dir, "cross_section.png"), dpi=300)

# Generate image maps for both original and filtered
original_dir = os.path.join(output_dir, "original")
filtered_dir = os.path.join(output_dir, "filtered")
os.makedirs(original_dir, exist_ok=True)
os.makedirs(filtered_dir, exist_ok=True)

original_maps = generate_all_maps(height_map, output_dir=original_dir)
filtered_maps = generate_all_maps(filtered_map, output_dir=filtered_dir)

print(f"Analysis complete. Results saved to {output_dir}")
```

## Next Steps

Now that you're familiar with the basics:

1. Explore the [API documentation](../api/processor.md) for more details on each module
2. Check the [examples directory](https://github.com/yourusername/tmd/examples) for more use cases
3. Try the [advanced tutorials](../tutorials/advanced.md) for specific applications
