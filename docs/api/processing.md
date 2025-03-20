# Processing Module

The Processing module provides functions for manipulating height maps, including cropping, rotation, thresholding, and extracting cross-sections and profiles.

## Overview

This module focuses on basic height map manipulations that are commonly needed when working with surface data:

- Cropping regions of interest
- Rotating to align features
- Thresholding to remove outliers or focus on specific height ranges
- Extracting cross-sections for 2D analysis
- Extracting profiles at specific locations

## Manipulation Functions

::: tmd.utils.processing.crop_height_map

::: tmd.utils.processing.rotate_height_map

::: tmd.utils.processing.flip_height_map

::: tmd.utils.processing.threshold_height_map

## Cross-Section Functions

::: tmd.utils.processing.extract_cross_section

::: tmd.utils.processing.extract_profile_at_percentage

## Examples

### Basic Manipulations

```python
from tmd.utils.processing import crop_height_map, rotate_height_map, threshold_height_map
import matplotlib.pyplot as plt

# Crop to region of interest
region = (50, 150, 75, 175)  # (row_start, row_end, col_start, col_end)
cropped_map = crop_height_map(height_map, region)

# Rotate by 45 degrees
rotated_map = rotate_height_map(height_map, angle=45, reshape=True)

# Apply threshold to remove outliers
h_min, h_max = height_map.min(), height_map.max()
h_range = h_max - h_min
# Keep central 80% of height values
thresholded_map = threshold_height_map(
    height_map,
    min_height=h_min + 0.1 * h_range,
    max_height=h_max - 0.1 * h_range
)

# Visualize results
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].imshow(height_map, cmap='viridis')
axes[0, 0].set_title("Original")

axes[0, 1].imshow(cropped_map, cmap='viridis')
axes[0, 1].set_title("Cropped")

axes[1, 0].imshow(rotated_map, cmap='viridis')
axes[1, 0].set_title("Rotated (45°)")

axes[1, 1].imshow(thresholded_map, cmap='viridis')
axes[1, 1].set_title("Thresholded")

plt.tight_layout()
plt.show()
```

### Cross-Section Extraction

```python
from tmd.utils.processing import extract_cross_section
import matplotlib.pyplot as plt

# Extract cross-section in X direction at the middle row
metadata = {
    'x_offset': 0.0,
    'x_length': 10.0,
    'y_offset': 0.0,
    'y_length': 10.0
}
position = height_map.shape[0] // 2  # Middle row
x_positions, x_heights = extract_cross_section(
    height_map,
    metadata,
    axis='x',
    position=position
)

# Plot the cross-section
plt.figure(figsize=(10, 6))
plt.plot(x_positions, x_heights, 'b-', linewidth=2)
plt.fill_between(x_positions, 0, x_heights, alpha=0.2)
plt.title(f'Cross-Section at Row {position}')
plt.xlabel('X Position (mm)')
plt.ylabel('Height')
plt.grid(True, alpha=0.3)
plt.show()
```

### Profile Extraction

```python
from tmd.utils.processing import extract_profile_at_percentage
import matplotlib.pyplot as plt
import numpy as np

# Extract profiles at different percentages
metadata = {
    'x_offset': 0.0,
    'x_length': 10.0,
    'y_offset': 0.0,
    'y_length': 10.0
}

# Create a figure to show multiple profiles
plt.figure(figsize=(12, 8))

# Extract and plot profiles at 25%, 50%, and 75% positions
colors = ['r', 'g', 'b']
percentages = [25, 50, 75]

for i, percentage in enumerate(percentages):
    profile = extract_profile_at_percentage(
        height_map,
        metadata,
        axis='x',
        percentage=percentage
    )
    
    # Get x axis positions
    x_values = np.linspace(0, metadata['x_length'], len(profile))
    
    # Plot with label and color
    plt.plot(x_values, profile, colors[i], linewidth=2, 
             label=f'Profile at {percentage}%')

plt.title('Height Profiles at Different Positions')
plt.xlabel('X Position (mm)')
plt.ylabel('Height')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```
