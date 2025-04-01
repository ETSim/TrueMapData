"""Ambient occlusion map generator."""
import logging
import numpy as np
from scipy import ndimage
from .base_generator import MapGenerator

logger = logging.getLogger(__name__)

class AOMapGenerator(MapGenerator):
    """Generator for ambient occlusion maps."""
    
    def __init__(self, samples: int = 16, strength: float = 1.0, **kwargs):
        """
        Initialize with AO parameters.
        
        Args:
            samples: Number of sample directions for AO calculation
            strength: AO effect strength
            **kwargs: Additional parameters
        """
        super().__init__(samples=samples, strength=strength, **kwargs)
    
    def generate(self, height_map: np.ndarray, **kwargs) -> np.ndarray:
        """
        Generate ambient occlusion map from height data.
        
        Args:
            height_map: Input height map
            **kwargs: Additional parameters including:
                - samples: Number of sample directions
                - strength: AO effect strength
                
        Returns:
            Ambient occlusion map (0-1 range)
        """
        params = self._get_params(**kwargs)
        samples = int(params.get('samples', 16))
        strength = float(params.get('strength', 1.0))
        
        # Normalize height map
        height_map_norm = self._prepare_height_map(height_map, normalize=True)
        
        try:
            # For large maps, use optimized implementation
            if height_map_norm.size > 250000:  # ~500x500 or larger
                return self._generate_optimized_ao(height_map_norm, strength)
            
            # Get dimensions
            rows, cols = height_map_norm.shape
            
            # Initialize AO map with all 1's (no occlusion)
            ao_map = np.ones((rows, cols), dtype=np.float32)
            
            # Calculate pixel radius based on image size
            radius = max(1, int(min(rows, cols) * 0.02))  # 2% of smaller dimension
            
            # Sample positions on a hemisphere
            theta = np.linspace(0, 2 * np.pi, samples)
            x_samples = np.cos(theta)
            y_samples = np.sin(theta)
            
            # For each direction, calculate occlusion
            for i in range(samples):
                # Sample direction
                dx, dy = x_samples[i], y_samples[i]
                
                # Calculate sample offsets for meshgrid (ensure proper transposition)
                x_offset = int(dx * radius)
                y_offset = int(dy * radius)
                
                # Create coordinate matrices for proper indexing
                y_coords, x_coords = np.mgrid[:rows, :cols]
                
                # Shifted coordinates
                x_shifted = np.clip(x_coords + x_offset, 0, cols-1)
                y_shifted = np.clip(y_coords + y_offset, 0, rows-1)
                
                # Get height at shifted positions (use advanced indexing)
                sampled_heights = height_map_norm[y_shifted, x_shifted]
                
                # Calculate occlusion factor
                height_diff = sampled_heights - height_map_norm
                occlusion = np.maximum(0, height_diff) * strength
                
                # Apply to AO map
                ao_map -= occlusion / samples
            
            # Ensure values are in valid range [0,1]
            ao_map = np.clip(ao_map, 0, 1)
            
            return ao_map
            
        except Exception as e:
            logger.error(f"Error generating ambient occlusion map: {e}")
            # Return a flat white image as fallback
            return np.ones_like(height_map_norm, dtype=np.float32)

    def _generate_optimized_ao(self, height_map: np.ndarray, strength: float) -> np.ndarray:
        """
        Generate ambient occlusion using an optimized algorithm for large maps.
        
        Args:
            height_map: Normalized height map
            strength: AO strength parameter
            
        Returns:
            Ambient occlusion map (0-1 range)
        """
        try:
            # Calculate gradient-based AO (fast approximation)
            dx = ndimage.sobel(height_map, axis=1)
            dy = ndimage.sobel(height_map, axis=0)
            
            # Calculate local slope
            slope = np.sqrt(dx**2 + dy**2)
            
            # Normalize slope
            if np.max(slope) > 0:
                slope = slope / np.max(slope)
            
            # Invert and scale based on strength - areas with high slopes get more occlusion
            ao_map = 1.0 - (slope * strength * 0.5)
            
            # Apply blurring for smoother effect
            ao_map = ndimage.gaussian_filter(ao_map, sigma=1.0)
            
            # Ensure values are in valid range [0,1]
            return np.clip(ao_map, 0, 1)
            
        except Exception as e:
            logger.error(f"Error in optimized AO generation: {e}")
            return np.ones_like(height_map)
    
    def _validate_params(self, params):
        """Validate and adjust parameters."""
        # Ensure samples is an integer ≥ 1
        samples = params.get('samples', 16)
        if not isinstance(samples, int) or samples < 1:
            params['samples'] = 16
        
        # Ensure strength is positive
        if params.get('strength', 0) <= 0:
            params['strength'] = 1.0
            
        return params
