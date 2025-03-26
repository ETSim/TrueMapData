"""Test resources package for TMD testing."""

import os
import numpy as np

# Directory containing resource files
RESOURCE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_resource_path(filename):
    """Get full path to a resource file."""
    return os.path.join(RESOURCE_DIR, filename)

def create_sample_height_map(size=(5, 5), pattern="peak"):
    """Create a sample heightmap for testing.
    
    Args:
        size: Tuple of (height, width)
        pattern: Type of pattern - "peak", "slope", "random", "with_nan"
    
    Returns:
        NumPy array with the sample height map
    """
    height, width = size
    
    if pattern == "peak":
        # Simple peak in the center
        x = np.linspace(-3, 3, width)
        y = np.linspace(-3, 3, height)
        xx, yy = np.meshgrid(x, y)
        z = np.exp(-(xx**2 + yy**2)/4)
        return z
        
    elif pattern == "slope":
        # Simple slope
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        xx, yy = np.meshgrid(x, y)
        z = xx + yy
        return z
        
    elif pattern == "with_nan":
        # Height map with some NaN values
        z = create_sample_height_map(size, "peak")
        # Add some NaN values
        z[1, 1] = np.nan
        z[3, 3] = np.nan
        return z
        
    elif pattern == "random":
        # Random height map
        return np.random.rand(height, width)
    
    # Default: flat surface
    return np.ones((height, width)) * 0.5
