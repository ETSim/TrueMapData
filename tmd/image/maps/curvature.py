"""
Curvature map generator.

This module provides a generator for creating curvature maps from height maps,
which identify convex (positive curvature) and concave (negative curvature) regions.
"""
import logging
import numpy as np
from scipy import ndimage
from .base_generator import MapGenerator

logger = logging.getLogger(__name__)

class CurvatureMapGenerator(MapGenerator):
    """Generator for Curvature maps."""
    
    def __init__(
        self, 
        mode: str = "mean", 
        scale: float = 1.0,
        normalize: bool = False,
        **kwargs
    ):
        """
        Initialize the curvature map generator.
        
        Args:
            mode: Curvature type ('mean', 'gaussian', 'profile', 'planform', 'maximal', 'minimal')
            scale: Scaling factor for curvature values
            normalize: Whether to normalize the height map before processing
            **kwargs: Additional default parameters
        """
        super().__init__(
            mode=mode, 
            scale=scale,
            normalize=normalize,
            **kwargs
        )
    
    def generate(self, height_map: np.ndarray, **kwargs) -> np.ndarray:
        """
        Generate a curvature map from a height map.
        
        Args:
            height_map: Input height map
            **kwargs: Generation parameters including:
                - mode: Curvature type to calculate
                - scale: Scaling factor for curvature values
                - metadata: TMD metadata containing physical dimensions
            
        Returns:
            Curvature map as numpy array (-1 to 1 range, mapped to 0-1 for display)
        """
        # Get parameters
        params = self._get_params(**kwargs)
        mode = params.get('mode', 'mean')
        scale = float(params.get('scale', 1.0))
        normalize = bool(params.get('normalize', False))
        
        # Get metadata for scaling
        metadata = kwargs.get('metadata', {}) or {}
        
        # Prepare height map
        height_map_norm = self._prepare_height_map(height_map, normalize=normalize)
        
        try:
            # Calculate cell size based on metadata if available
            if 'x_length' in metadata and 'y_length' in metadata:
                height, width = height_map_norm.shape
                cell_size_x = metadata['x_length'] / width
                cell_size_y = metadata['y_length'] / height
            elif 'mmpp' in metadata:
                cell_size_x = cell_size_y = metadata['mmpp']
            else:
                cell_size_x = cell_size_y = 1.0
                
            # Apply Gaussian smoothing to reduce noise
            smooth_height = ndimage.gaussian_filter(height_map_norm, sigma=1.0)
                
            # Calculate first derivatives
            fx, fy = np.gradient(smooth_height, cell_size_x, cell_size_y)
            
            # Calculate second derivatives
            fxx, fxy = np.gradient(fx, cell_size_x, cell_size_y)
            fyx, fyy = np.gradient(fy, cell_size_x, cell_size_y)
            
            # Normalize gradient components
            p = fx**2 + fy**2
            q = p + 1.0
            
            # Calculate different curvature types
            if mode == 'gaussian':
                # Gaussian curvature = (fxx*fyy - fxy^2) / (1 + fx^2 + fy^2)^2
                curvature = (fxx * fyy - fxy * fyx) / (q**2)
            elif mode == 'mean':
                # Mean curvature = 0.5 * (fxx*(1+fy^2) - 2*fxy*fx*fy + fyy*(1+fx^2)) / (1 + fx^2 + fy^2)^(3/2)
                curvature = 0.5 * ((1 + fy**2) * fxx - 2 * fx * fy * fxy + (1 + fx**2) * fyy) / (q**(3/2))
            elif mode == 'maximal':
                # Maximum principal curvature
                H = 0.5 * ((1 + fy**2) * fxx - 2 * fx * fy * fxy + (1 + fx**2) * fyy) / (q**(3/2))
                K = (fxx * fyy - fxy * fyx) / (q**2)
                curvature = H + np.sqrt(np.maximum(H**2 - K, 0))
            elif mode == 'minimal':
                # Minimum principal curvature
                H = 0.5 * ((1 + fy**2) * fxx - 2 * fx * fy * fxy + (1 + fx**2) * fyy) / (q**(3/2))
                K = (fxx * fyy - fxy * fyx) / (q**2)
                curvature = H - np.sqrt(np.maximum(H**2 - K, 0))
            elif mode == 'profile':
                # Profile curvature (in direction of steepest slope)
                curvature = ((fx**2 * fxx + 2 * fx * fy * fxy + fy**2 * fyy) / 
                           (p * np.sqrt(q)))
            elif mode == 'planform':
                # Plan curvature (perpendicular to the direction of steepest slope)
                curvature = ((fy**2 * fxx - 2 * fx * fy * fxy + fx**2 * fyy) / 
                           (p * np.sqrt(q)))
            else:
                # Default to mean curvature
                curvature = 0.5 * ((1 + fy**2) * fxx - 2 * fx * fy * fxy + (1 + fx**2) * fyy) / (q**(3/2))
            
            # Apply scaling factor
            curvature = curvature * scale
            
            # Handle potential NaN or infinity values
            curvature = np.nan_to_num(curvature)
            
            # Clip extreme values (adjustable range)
            curvature_range = np.percentile(curvature, [2, 98])
            clip_min, clip_max = curvature_range[0] * 1.5, curvature_range[1] * 1.5
            curvature = np.clip(curvature, clip_min, clip_max)
            
            # Map from [-max, max] to [0, 1] for display
            abs_max = max(abs(clip_min), abs(clip_max))
            curvature = 0.5 + (curvature / (2.0 * abs_max))
            
            # Ensure output is in valid range
            return np.clip(curvature, 0.0, 1.0)
            
        except Exception as e:
            logger.error(f"Error generating curvature map: {e}")
            # Return a uniform gray image as fallback
            return np.ones_like(height_map_norm) * 0.5
    
    def _validate_params(self, params):
        """Validate and adjust parameters."""
        # Validate curvature mode
        valid_modes = ['mean', 'gaussian', 'profile', 'planform', 'maximal', 'minimal']
        if params.get('mode') not in valid_modes:
            params['mode'] = 'mean'
            
        # Validate scale (positive)
        if params.get('scale', 0) <= 0:
            params['scale'] = 1.0
            
        return params