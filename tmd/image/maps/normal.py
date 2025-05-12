"""Normal map generator."""
import logging
import numpy as np
from .base_generator import MapGenerator

logger = logging.getLogger(__name__)

class NormalMapGenerator(MapGenerator):
    """Generator for normal maps."""
    
    def __init__(self, strength: float = 1.0, normalize: bool = True, **kwargs):
        """
        Initialize with strength parameter.
        
        Args:
            strength: Factor to control the strength of normals
            normalize: Whether to normalize the height map before processing
            **kwargs: Additional parameters
        """
        super().__init__(strength=strength, normalize=normalize, **kwargs)
    
    def generate(self, height_map: np.ndarray, **kwargs) -> np.ndarray:
        """
        Generate normal map from height data.
        
        Args:
            height_map: Input height map
            **kwargs: Additional parameters including:
                - strength: Controls the strength of normal map effect
                - metadata: TMD metadata containing physical dimensions
                
        Returns:
            RGB normal map (0-1 range)
        """
        # 1) Safely extract metadata and ensure it's a dict
        metadata = kwargs.pop('metadata', {}) or {}
        
        # 2) Pull in algorithmic parameters
        params = self._get_params(**kwargs)
        strength = float(params.get('strength', 1.0))
        normalize = bool(params.get('normalize', True))
        
        # 3) Prepare the height map
        height_map_norm = self._prepare_height_map(height_map, normalize=normalize)
        logger.debug("Height map normalized." if normalize else "Using raw height values.")
        
        try:
            rows, cols = height_map_norm.shape
            scaling_applied = False
            normal_map = np.zeros((rows, cols, 3), dtype=np.float32)
            
            # 4a) Physical dimensions if available
            if 'x_length' in metadata and 'y_length' in metadata:
                dx = float(metadata['x_length']) / cols if cols > 0 else 1.0
                dy = float(metadata['y_length']) / rows if rows > 0 else 1.0
                scaling_applied = True
            
            # 4b) Millimeters per pixel override
            if 'mmpp' in metadata:
                mmpp = float(metadata['mmpp'])
                dx = dy = mmpp
                scaling_applied = True
                # adjust by magnification if present
                if 'magnification' in metadata:
                    mag = float(metadata['magnification'])
                    strength *= (0.5 + 0.5 / mag)
            
            # 4c) Fall back to aspect ratio
            if not scaling_applied:
                aspect = cols / rows if rows > 0 else 1.0
                dx = 1.0
                dy = dx / aspect
            
            # 5) Compute gradients
            dx_array = np.zeros_like(height_map_norm)
            dy_array = np.zeros_like(height_map_norm)
            
            # Central differences
            dx_array[:, 1:-1] = (height_map_norm[:, 2:] - height_map_norm[:, :-2]) / (2.0 * dx)
            dy_array[1:-1, :] = (height_map_norm[2:, :] - height_map_norm[:-2, :]) / (2.0 * dy)
            # Edges
            dx_array[:, 0]   = (height_map_norm[:, 1]   - height_map_norm[:, 0])   / dx
            dx_array[:, -1]  = (height_map_norm[:, -1]  - height_map_norm[:, -2])  / dx
            dy_array[0, :]   = (height_map_norm[1, :]   - height_map_norm[0, :])   / dy
            dy_array[-1, :]  = (height_map_norm[-1, :]  - height_map_norm[-2, :])  / dy
            
            # 6) Apply strength
            dx_array *= strength
            dy_array *= strength
            
            # 7) Build normal vectors and normalize
            normal_map[..., 0] = -dx_array
            normal_map[..., 1] = -dy_array
            normal_map[..., 2] = 1.0
            norm = np.linalg.norm(normal_map, axis=2, keepdims=True)
            normal_map = np.divide(normal_map, norm, out=normal_map, where=norm>0)
            
            # 8) Map [-1,1] -> [0,1] for RGB output
            return (normal_map + 1.0) * 0.5
        
        except Exception as e:
            logger.error(f"Error generating normal map: {e}")
            # Fallback: flat normal pointing up
            default = np.zeros((rows, cols, 3), dtype=np.float32)
            default[..., 2] = 1.0
            return (default + 1.0) * 0.5
    
    def _validate_params(self, params):
        """Validate and adjust parameters."""
        if params.get('strength', 0) <= 0:
            params['strength'] = 1.0
        params['normalize'] = bool(params.get('normalize', True))
        return params
