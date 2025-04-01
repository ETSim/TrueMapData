"""Normal map generator."""
import logging
import numpy as np
from .base_generator import MapGenerator

logger = logging.getLogger(__name__)

class NormalMapGenerator(MapGenerator):
    """Generator for normal maps."""
    
    def __init__(self, strength: float = 1.0, **kwargs):
        """
        Initialize with strength parameter.
        
        Args:
            strength: Factor to control the strength of normals
            **kwargs: Additional parameters
        """
        super().__init__(strength=strength, **kwargs)
    
    def generate(self, height_map: np.ndarray, **kwargs) -> np.ndarray:
        """
        Generate normal map from height data.
        
        Args:
            height_map: Input height map
            **kwargs: Additional parameters including:
                - strength: Controls the strength of normal map effect
                
        Returns:
            RGB normal map (0-1 range)
        """
        params = self._get_params(**kwargs)
        strength = params.get('strength', 1.0)
        
        # Get metadata for proper aspect ratio
        metadata = kwargs.get('metadata', {})
        
        # Get physical dimensions from metadata
        x_length = metadata.get('x_length', None)
        y_length = metadata.get('y_length', None)
        
        # Normalize height map - important for consistent results
        height_map_norm = self._prepare_height_map(height_map, normalize=True)
        
        try:
            # Get dimensions ensuring they're in the right order
            rows, cols = height_map_norm.shape
            
            # Create normal map
            normal_map = np.zeros((rows, cols, 3), dtype=np.float32)
            
            # Calculate proper scaling based on physical dimensions
            if x_length is not None and y_length is not None:
                dx = x_length / cols
                dy = y_length / rows
            else:
                aspect_ratio = cols / rows if rows > 0 else 1.0
                dx = 1.0
                dy = dx / aspect_ratio
            
            # Calculate gradients with proper handling of edges
            # Use central differences for interior pixels
            dx_array = np.zeros_like(height_map_norm)
            dy_array = np.zeros_like(height_map_norm)
            
            # X-gradient using central difference with physical dimensions
            dx_array[:, 1:-1] = (height_map_norm[:, 2:] - height_map_norm[:, :-2]) / (2.0 * dx)
            # Y-gradient using central difference with physical dimensions
            dy_array[1:-1, :] = (height_map_norm[2:, :] - height_map_norm[:-2, :]) / (2.0 * dy)
            
            # Handle edges
            # Left and right edges (use forward/backward difference)
            dx_array[:, 0] = (height_map_norm[:, 1] - height_map_norm[:, 0]) / dx
            dx_array[:, -1] = (height_map_norm[:, -1] - height_map_norm[:, -2]) / dx
            # Top and bottom edges (use forward/backward difference)
            dy_array[0, :] = (height_map_norm[1, :] - height_map_norm[0, :]) / dy
            dy_array[-1, :] = (height_map_norm[-1, :] - height_map_norm[-2, :]) / dy
            
            # Apply strength
            dx_array *= strength
            dy_array *= strength
            
            # Normal vectors: (-dx, -dy, 1)
            normal_map[..., 0] = -dx_array
            normal_map[..., 1] = -dy_array
            normal_map[..., 2] = 1.0
            
            # Normalize vectors
            norm = np.sqrt(np.sum(normal_map**2, axis=2, keepdims=True))
            normal_map = np.divide(normal_map, norm, out=normal_map, where=norm > 0)
            
            # Convert from [-1,1] to [0,1] range
            normal_map = (normal_map + 1.0) * 0.5
            
            return normal_map
            
        except Exception as e:
            logger.error(f"Error generating normal map: {e}")
            # Return a flat normal map (all pointing up)
            rows, cols = height_map_norm.shape
            default_normal = np.zeros((rows, cols, 3), dtype=np.float32)
            default_normal[..., 2] = 1.0  # Z component is 1.0
            default_normal = (default_normal + 1.0) * 0.5  # Convert to [0,1]
            return default_normal
            
    def _validate_params(self, params):
        """Validate and adjust parameters."""
        # Ensure strength is positive
        if params.get('strength', 0) <= 0:
            params['strength'] = 1.0
            
        return params
