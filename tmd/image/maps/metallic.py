"""
Metallic map generator.

This module provides a generator for creating metallic maps from height maps,
which define which areas of a surface are metallic for PBR materials.
"""
import logging
import numpy as np

from .base_generator import MapGenerator

logger = logging.getLogger(__name__)

class MetallicMapGenerator(MapGenerator):
    """Generator for Metallic maps."""
    
    def __init__(
        self, 
        method: str = "constant", 
        value: float = 0.0,
        threshold: float = 0.7,
        **kwargs
    ):
        """
        Initialize the metallic map generator.
        
        Args:
            method: Method for generating map ('constant', 'height_threshold', 'pattern')
            value: Metallic value for constant method
            threshold: Height threshold for height_threshold method
            **kwargs: Additional default parameters
        """
        super().__init__(
            method=method, 
            value=value, 
            threshold=threshold, 
            **kwargs
        )
    
    def generate(self, height_map: np.ndarray, **kwargs) -> np.ndarray:
        """
        Generate a metallic map from a height map.
        
        Args:
            height_map: Input height map
            **kwargs: Generation parameters including:
                - method: Generation method
                - value: Constant value (for constant method)
                - threshold: Height threshold (for height_threshold method)
                - pattern_scale: Scale for pattern (for pattern method)
                - pattern_type: Type of pattern ('grid', 'checker', 'noise')
            
        Returns:
            Metallic map as numpy array (0-1 range)
        """
        # Get parameters
        params = self._get_params(**kwargs)
        method = params['method']
        value = params['value']
        threshold = params['threshold']
        
        # Prepare height map
        height_map_norm = self._prepare_height_map(height_map)
        
        # Generate metallic map based on method
        if method == "constant":
            # Constant value throughout (most common)
            return np.ones_like(height_map_norm) * value
            
        elif method == "height_threshold":
            # Areas above threshold are metallic
            return np.where(height_map_norm > threshold, 1.0, 0.0).astype(np.float32)
            
        elif method == "gradient":
            # Gradient based on height (higher = more metallic)
            return height_map_norm.copy()
            
        elif method == "pattern":
            # Get pattern parameters
            pattern_scale = params.get('pattern_scale', 1.0)
            pattern_type = params.get('pattern_type', 'grid')
            
            # Get dimensions
            h, w = height_map_norm.shape
            
            if pattern_type == "grid":
                # Grid pattern
                grid_size = int(min(h, w) / (10.0 / pattern_scale))
                grid_size = max(2, grid_size)  # Ensure at least 2 pixels
                
                x = np.arange(w) % grid_size
                y = np.arange(h) % grid_size
                
                x_grid, y_grid = np.meshgrid(x, y)
                border_width = max(1, grid_size // 4)
                
                # Create grid pattern with borders
                mask = (x_grid < border_width) | (x_grid >= grid_size - border_width) | \
                       (y_grid < border_width) | (y_grid >= grid_size - border_width)
                
                return mask.astype(np.float32)
                
            elif pattern_type == "checker":
                # Checkerboard pattern
                checker_size = int(min(h, w) / (10.0 / pattern_scale))
                checker_size = max(2, checker_size)  # Ensure at least 2 pixels
                
                x = (np.arange(w) // checker_size) % 2
                y = (np.arange(h) // checker_size) % 2
                
                x_grid, y_grid = np.meshgrid(x, y)
                checker = (x_grid + y_grid) % 2
                
                return checker.astype(np.float32)
                
            elif pattern_type == "noise":
                try:
                    from scipy.ndimage import gaussian_filter
                    
                    # Generate random noise
                    rng = np.random.RandomState(params.get('seed', 42))
                    noise = rng.rand(h, w)
                    
                    # Smooth the noise
                    smoothing = 5.0 / pattern_scale
                    noise = gaussian_filter(noise, sigma=smoothing)
                    
                    # Normalize to 0-1
                    noise = (noise - noise.min()) / (noise.max() - noise.min())
                    
                    # Apply threshold
                    noise_threshold = params.get('noise_threshold', 0.5)
                    return (noise > noise_threshold).astype(np.float32)
                    
                except ImportError:
                    logger.warning("SciPy not available for noise pattern. Using checkerboard instead.")
                    return self.generate(
                        height_map_norm, 
                        method="pattern", 
                        pattern_type="checker", 
                        pattern_scale=pattern_scale
                    )
        else:
            # Default: non-metallic
            return np.zeros_like(height_map_norm)
    
    def _validate_params(self, params):
        """Validate and adjust parameters."""
        # Ensure value is in [0, 1]
        if 'value' in params:
            params['value'] = np.clip(params['value'], 0.0, 1.0)
            
        # Ensure threshold is in [0, 1]
        if 'threshold' in params:
            params['threshold'] = np.clip(params['threshold'], 0.0, 1.0)
            
        return params
