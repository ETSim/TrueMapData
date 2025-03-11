# Filter Module

The filter module provides functions for processing height maps, particularly for separating waviness (low-frequency) and roughness (high-frequency) components.

## Gaussian Filtering

::: tmd.utils.filter.apply_gaussian_filter

## Waviness and Roughness

### Extract Waviness

::: tmd.utils.filter.extract_waviness

### Extract Roughness

::: tmd.utils.filter.extract_roughness

## RMS Calculations

### RMS Roughness

::: tmd.utils.filter.calculate_rms_roughness

### RMS Waviness

::: tmd.utils.filter.calculate_rms_waviness

## Surface Gradients

### Calculate Surface Gradient

::: tmd.utils.filter.calculate_surface_gradient

### Calculate Slope

::: tmd.utils.filter.calculate_slope

## Examples

### Separating Waviness and Roughness

```python
import numpy as np
import matplotlib.pyplot as plt
from tmd.utils.filter import extract_waviness, extract_roughness

# Load or create a height map
height_map = np.load("your_height_map.npy")

# Extract waviness (low frequency) component with sigma=5.0
waviness = extract_waviness(height_map, sigma=5.0)

# Extract roughness (high frequency) component
roughness = extract_roughness(height_map, sigma=5.0)

# Plot the original and separated components
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(height_map, cmap='viridis')
axs[0].set_title('Original Height Map')
axs[1].imshow(waviness, cmap='viridis')
axs[1].set_title('Waviness Component')
axs[2].imshow(roughness, cmap='viridis')
axs[2].set_title('Roughness Component')
plt.tight_layout()
plt.show()
```

### Calculating Surface Properties

```python
from tmd.utils.filter import calculate_surface_gradient, calculate_slope
from tmd.utils.filter import calculate_rms_roughness, calculate_rms_waviness

# Calculate gradients
grad_x, grad_y = calculate_surface_gradient(height_map, scale_factor=1.0)

# Calculate slope (magnitude of gradient)
slope = calculate_slope(height_map)

# Calculate RMS values
rms_roughness = calculate_rms_roughness(height_map, sigma=1.0)
rms_waviness = calculate_rms_waviness(height_map, sigma=1.0)

print(f"RMS Roughness: {rms_roughness}")
print(f"RMS Waviness: {rms_waviness}")
```
