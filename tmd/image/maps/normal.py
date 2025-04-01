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
        
        # Normalize height map
        height_map_norm = self._prepare_height_map(height_map, normalize=True)
        
        try:
            # Get dimensions
            rows, cols = height_map_norm.shape
            
            # Create normal map
            normal_map = np.zeros((rows, cols, 3), dtype=np.float32)
            
            # Calculate gradients with proper handling of edges
            # Use central differences for interior pixels
            dx = np.zeros_like(height_map_norm)
            dy = np.zeros_like(height_map_norm)
            
            # X-gradient using central difference
            dx[:, 1:-1] = (height_map_norm[:, 2:] - height_map_norm[:, :-2]) / 2.0
            # Y-gradient using central difference
            dy[1:-1, :] = (height_map_norm[2:, :] - height_map_norm[:-2, :]) / 2.0
            
            # Handle edges
            # Left and right edges (use forward/backward difference)
            dx[:, 0] = height_map_norm[:, 1] - height_map_norm[:, 0]
            dx[:, -1] = height_map_norm[:, -1] - height_map_norm[:, -2]
            # Top and bottom edges (use forward/backward difference)
            dy[0, :] = height_map_norm[1, :] - height_map_norm[0, :]
            dy[-1, :] = height_map_norm[-1, :] - height_map_norm[-2, :]
            
            # Apply strength
            dx *= strength
            dy *= strength
            
            # Normal vectors: (-dx, -dy, 1)
            normal_map[..., 0] = -dx
            normal_map[..., 1] = -dy
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
            default_normal = np.zeros((rows, cols, 3), dtype=np.float32)
            default_normal[..., 2] = 1.0  # Z component is 1.0
            return default_normal
            
    def _validate_params(self, params):
        """Validate and adjust parameters."""
        # Ensure strength is positive
        if params.get('strength', 0) <= 0:
            params['strength'] = 1.0
            
        return params
