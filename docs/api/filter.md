# Filter Module

The Filter module provides functions for processing and analyzing height maps, including Gaussian filtering, roughness/waviness extraction, and surface gradient calculations.

## Overview

Surface analysis often requires separating different scale components of a surface:

- **Waviness**: Low-frequency variations (general form)
- **Roughness**: High-frequency variations (surface texture)

This module provides tools to filter, separate, and analyze these components.

## Filtering Functions

::: tmd.utils.filter.apply_gaussian_filter

::: tmd.utils.filter.extract_waviness

::: tmd.utils.filter.extract_roughness

## Surface Metrics

::: tmd.utils.filter.calculate_rms_roughness

::: tmd.utils.filter.calculate_rms_waviness

## Surface Gradient Analysis

::: tmd.utils.filter.calculate_surface_gradient

::: tmd.utils.filter.calculate_slope

## Examples

### Basic Filtering

```python
from tmd.utils.filter import apply_gaussian_filter

# Apply Gaussian smoothing to remove high-frequency noise
smoothed_map = apply_gaussian_filter(height_map, sigma=1.0)
```

### Roughness/Waviness Separation

```python
from tmd.utils.filter import extract_waviness, extract_roughness
import matplotlib.pyplot as plt

# Extract waviness (low-frequency) component
waviness = extract_waviness(height_map, sigma=5.0)

# Extract roughness (high-frequency) component
roughness = extract_roughness(height_map, sigma=5.0)

# Visualize components
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(height_map, cmap='viridis')
axes[0].set_title("Original Height Map")
axes[1].imshow(waviness, cmap='viridis')
axes[1].set_title("Waviness Component")
axes[2].imshow(roughness, cmap='gray')
axes[2].set_title("Roughness Component")
plt.tight_layout()
plt.show()
```

### Surface Metrics Calculation

```python
from tmd.utils.filter import calculate_rms_roughness, calculate_rms_waviness

# Calculate roughness parameter
rms_roughness = calculate_rms_roughness(height_map, sigma=5.0)
print(f"RMS Roughness: {rms_roughness:.3f} µm")

# Calculate waviness parameter
rms_waviness = calculate_rms_waviness(height_map, sigma=5.0)
print(f"RMS Waviness: {rms_waviness:.3f} µm")
```

### Gradient and Slope Analysis

```python
from tmd.utils.filter import calculate_surface_gradient, calculate_slope
import matplotlib.pyplot as plt

# Calculate surface gradients
grad_x, grad_y = calculate_surface_gradient(height_map, scale=1.0)

# Calculate slope magnitude
slope = calculate_slope(height_map, scale=1.0)

# Visualize results
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].imshow(height_map, cmap='viridis')
axes[0, 0].set_title("Original Height Map")

axes[0, 1].imshow(grad_x, cmap='RdBu')
axes[0, 1].set_title("X Gradient")

axes[1, 0].imshow(grad_y, cmap='RdBu')
axes[1, 0].set_title("Y Gradient")

axes[1, 1].imshow(slope, cmap='magma')
axes[1, 1].set_title("Slope Magnitude")

plt.tight_layout()
plt.show()
```
