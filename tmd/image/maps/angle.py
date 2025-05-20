"""
Angle map generator.

This module provides a generator for creating angle/slope maps from height maps,
which visualize the steepness of the terrain.
"""
import logging
import numpy as np
from scipy import ndimage
from .base_generator import MapGenerator

logger = logging.getLogger(__name__)

class AngleMapGenerator(MapGenerator):
    """Generator for Angle/Slope maps."""
    
    def __init__(
        self, 
        max_angle: float = 90.0, 
        normalize: bool = False,
        **kwargs
    ):
        """
        Initialize the angle map generator.
        
        Args:
            max_angle: Maximum angle (in degrees) to represent as white
            normalize: Whether to normalize the height map before processing
            **kwargs: Additional default parameters
        """
        super().__init__(
            max_angle=max_angle,
            normalize=normalize,
            **kwargs
        )
    
    def generate(self, height_map: np.ndarray, **kwargs) -> np.ndarray:
        """
        Generate an angle map from a height map.
        
        Args:
            height_map: Input height map
            **kwargs: Generation parameters including:
                - max_angle: Maximum angle in degrees to represent
                - metadata: TMD metadata containing physical dimensions
            
        Returns:
            Angle map as numpy array (0-1 range, where 1 represents max_angle)
        """
        # Get parameters
        params = self._get_params(**kwargs)
        max_angle = float(params.get('max_angle', 90.0))
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
                
            # Apply optional Gaussian smoothing to reduce noise
            smooth_height = ndimage.gaussian_filter(height_map_norm, sigma=0.5)
                
            # Calculate gradients (use Sobel for more accurate gradient)
            try:
                dx = ndimage.sobel(smooth_height, axis=1) / (8.0 * cell_size_x)
                dy = ndimage.sobel(smooth_height, axis=0) / (8.0 * cell_size_y)
            except:
                # Fallback to numpy gradient
                dx, dy = np.gradient(smooth_height, cell_size_x, cell_size_y)
            
            # Calculate slope magnitude in degrees
            slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
            slope_deg = np.degrees(slope_rad)
            
            # Normalize to [0,1] range based on max_angle
            angle_map = slope_deg / max_angle
            
            # Ensure output is in valid range
            return np.clip(angle_map, 0.0, 1.0)
            
        except Exception as e:
            logger.error(f"Error generating angle map: {e}")
            # Return a uniform image as fallback
            return np.ones_like(height_map_norm) * 0.25
    
    def _validate_params(self, params):
        """Validate and adjust parameters."""
        # Validate max_angle (positive, max 90)
        if params.get('max_angle', 0) <= 0:
            params['max_angle'] = 90.0
        elif params.get('max_angle', 0) > 90.0:
            params['max_angle'] = 90.0
            
        return params