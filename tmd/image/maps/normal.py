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
                - metadata: TMD metadata containing physical dimensions
                
        Returns:
            RGB normal map (0-1 range)
        """
        params = self._get_params(**kwargs)
        strength = float(params.get('strength', 1.0))
        
        # Get metadata for proper physical scaling
        metadata = kwargs.get('metadata', {})
        
        # Normalize height map - important for consistent results
        height_map_norm = self._prepare_height_map(height_map, normalize=True)
        
        try:
            # Get dimensions
            rows, cols = height_map_norm.shape
            
            # Create normal map
            normal_map = np.zeros((rows, cols, 3), dtype=np.float32)
            
            # Calculate proper scaling based on physical dimensions
            # Try to use the most accurate physical dimension available
            scaling_applied = False
            
            # 1. Try millimeters per pixel (mmpp) if available - most accurate
            if 'mmpp' in metadata:
                mmpp = float(metadata['mmpp'])
                dx = dy = mmpp
                
                # Apply magnification adjustment if available
                if 'magnification' in metadata:
                    mag = float(metadata['magnification'])
                    # Higher magnification means more detail, so adjust strength
                    strength *= 0.5 + (0.5 / mag)
                    logger.debug(f"Adjusted normal strength by magnification: {mag}")
                
                logger.debug(f"Using mmpp from metadata: {mmpp} mm/pixel")
                scaling_applied = True
                
            # 2. Try physical dimensions next
            elif 'x_length' in metadata and 'y_length' in metadata:
                x_length = float(metadata['x_length'])
                y_length = float(metadata['y_length'])
                dx = x_length / cols if cols > 0 else 1.0
                dy = y_length / rows if rows > 0 else 1.0
                logger.debug(f"Using physical dimensions: x_length={x_length}, y_length={y_length}")
                scaling_applied = True
                
            # 3. Fall back to aspect ratio if no physical dimensions
            if not scaling_applied:
                aspect_ratio = cols / rows if rows > 0 else 1.0
                dx = 1.0
                dy = dx / aspect_ratio
                logger.debug(f"Using aspect ratio: {aspect_ratio}")
            
            # Apply camera settings adjustments if available
            if 'camera' in metadata:
                camera_data = metadata['camera']
                
                # Parse aperture to adjust normal strength
                if 'aperture_str' in camera_data:
                    aperture_str = camera_data['aperture_str']
                    try:
                        # Extract f-stop number (e.g., "F 8" -> 8.0)
                        f_stop = float(aperture_str.replace('F', '').strip())
                        # Smaller apertures (larger f-stop) need more strength
                        aperture_factor = min(max(f_stop / 5.6, 0.8), 1.5)
                        strength *= aperture_factor
                        logger.debug(f"Adjusted strength by aperture: {aperture_factor}")
                    except (ValueError, TypeError):
                        pass
            
            # Calculate gradients with physical scaling
            dx_array = np.zeros_like(height_map_norm)
            dy_array = np.zeros_like(height_map_norm)
            
            # X-gradient using central difference with proper physical scaling
            dx_array[:, 1:-1] = (height_map_norm[:, 2:] - height_map_norm[:, :-2]) / (2.0 * dx)
            # Y-gradient using central difference with proper physical scaling
            dy_array[1:-1, :] = (height_map_norm[2:, :] - height_map_norm[:-2, :]) / (2.0 * dy)
            
            # Handle edges with forward/backward differences
            dx_array[:, 0] = (height_map_norm[:, 1] - height_map_norm[:, 0]) / dx
            dx_array[:, -1] = (height_map_norm[:, -1] - height_map_norm[:, -2]) / dx
            dy_array[0, :] = (height_map_norm[1, :] - height_map_norm[0, :]) / dy
            dy_array[-1, :] = (height_map_norm[-1, :] - height_map_norm[-2, :]) / dy
            
            # Apply user-defined strength
            dx_array *= strength
            dy_array *= strength
            
            # Create normal vectors: (-dx, -dy, 1)
            normal_map[..., 0] = -dx_array
            normal_map[..., 1] = -dy_array
            normal_map[..., 2] = 1.0
            
            # Normalize vectors to unit length
            norm = np.sqrt(np.sum(normal_map**2, axis=2, keepdims=True))
            normal_map = np.divide(normal_map, norm, out=normal_map, where=norm > 0)
            
            # Convert from [-1,1] to [0,1] range for image export
            normal_map = (normal_map + 1.0) * 0.5
            
            return normal_map
            
        except Exception as e:
            logger.error(f"Error generating normal map: {e}")
            # Return a flat normal map (all pointing up) as fallback
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
