"""
Depth map generator.

This module provides a generator for creating depth maps from height maps,
which represent the distance from the viewpoint to surfaces in the scene.
"""
import logging
import numpy as np
from scipy import ndimage
from .base_generator import MapGenerator

logger = logging.getLogger(__name__)

class DepthMapGenerator(MapGenerator):
    """Generator for Depth maps."""
    
    def __init__(
        self, 
        mode: str = "linear",
        reverse: bool = False,
        min_depth: float = 0.0,
        max_depth: float = 1.0,
        focal_plane: float = 0.5,
        focal_range: float = 0.2,
        smoothing: float = 0.0,
        visualization: str = "grayscale",
        colormap: str = "plasma",
        enhance_contrast: float = 1.0,
        normalize: bool = True,
        **kwargs
    ):
        """
        Initialize the depth map generator.
        
        Args:
            mode: Depth calculation mode ('linear', 'inverse', 'focal')
            reverse: Whether to reverse the depth values (far = dark, near = light)
            min_depth: Minimum depth value in output
            max_depth: Maximum depth value in output
            focal_plane: Relative position of focal plane (0-1) for 'focal' mode
            focal_range: Range around focal plane that appears in focus
            smoothing: Amount of smoothing to apply to depth values
            visualization: Output visualization type ('grayscale', 'color', 'heatmap')
            colormap: Matplotlib colormap name for color visualizations
            enhance_contrast: Contrast enhancement factor
            normalize: Whether to normalize the height map before processing
            **kwargs: Additional default parameters
        """
        super().__init__(
            mode=mode,
            reverse=reverse,
            min_depth=min_depth,
            max_depth=max_depth,
            focal_plane=focal_plane,
            focal_range=focal_range,
            smoothing=smoothing,
            visualization=visualization,
            colormap=colormap,
            enhance_contrast=enhance_contrast,
            normalize=normalize,
            **kwargs
        )
    
    def generate(self, height_map: np.ndarray, **kwargs) -> np.ndarray:
        """
        Generate a depth map from a height map.
        
        Args:
            height_map: Input height map
            **kwargs: Generation parameters (see __init__ for details)
            
        Returns:
            Depth map as numpy array (0-1 range, where darker typically means closer)
        """
        # Get parameters
        params = self._get_params(**kwargs)
        mode = params.get('mode', 'linear')
        reverse = bool(params.get('reverse', False))
        min_depth = float(params.get('min_depth', 0.0))
        max_depth = float(params.get('max_depth', 1.0))
        focal_plane = float(params.get('focal_plane', 0.5))
        focal_range = float(params.get('focal_range', 0.2))
        smoothing = float(params.get('smoothing', 0.0))
        visualization = params.get('visualization', 'grayscale')
        colormap = params.get('colormap', 'plasma')
        enhance_contrast = float(params.get('enhance_contrast', 1.0))
        normalize = bool(params.get('normalize', True))
        
        # Get metadata for scale information
        metadata = kwargs.get('metadata', {}) or {}
        
        # Prepare height map
        height_map_norm = self._prepare_height_map(height_map, normalize=normalize)
        
        try:
            # Generate raw depth values based on the mode
            if mode == 'linear':
                # Linear mapping of height to depth
                depth_map = height_map_norm.copy()
                
            elif mode == 'inverse':
                # Inverse mapping (lower heights = further from camera)
                # Useful for terrain viewed from above
                depth_map = 1.0 - height_map_norm
                
            elif mode == 'focal':
                # Distance from a focal plane
                depth_map = np.abs(height_map_norm - focal_plane)
                # Scale the depth map based on focal range
                depth_map = np.clip(depth_map / focal_range, 0.0, 1.0)
                
            else:
                # Default to linear mapping
                depth_map = height_map_norm.copy()
            
            # Apply smoothing if requested
            if smoothing > 0:
                depth_map = ndimage.gaussian_filter(depth_map, sigma=smoothing)
            
            # Apply reverse if requested (by default: higher values = further away)
            if reverse:
                depth_map = 1.0 - depth_map
            
            # Apply min/max depth range
            depth_range = max_depth - min_depth
            depth_map = min_depth + (depth_map * depth_range)
            
            # Apply visualization format
            if visualization == 'color' or visualization == 'heatmap':
                try:
                    # Create an RGB visualization
                    rgb_map = self._apply_colormap(depth_map, colormap, enhance_contrast)
                    return rgb_map
                except Exception as e:
                    logger.warning(f"Color visualization failed: {e}, falling back to grayscale")
                    visualization = 'grayscale'
            
            # Apply contrast enhancement for grayscale
            if visualization == 'grayscale' and enhance_contrast != 1.0:
                # Enhance contrast around the mean
                mean_val = np.mean(depth_map)
                depth_map = mean_val + (depth_map - mean_val) * enhance_contrast
                depth_map = np.clip(depth_map, 0.0, 1.0)
            
            return depth_map
            
        except Exception as e:
            logger.error(f"Error generating depth map: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            
            # Return a uniform gray image as fallback
            return np.ones_like(height_map_norm) * 0.5
    
    def _apply_colormap(self, depth_map, colormap_name, enhance_contrast=1.0):
        """Apply a colormap to the depth map."""
        # Create RGB visualization
        import matplotlib.pyplot as plt
        from matplotlib import cm
        
        # Enhance contrast if requested
        if enhance_contrast != 1.0:
            # Enhance contrast around the mean
            mean_val = np.mean(depth_map)
            enhanced = mean_val + (depth_map - mean_val) * enhance_contrast
            depth_map = np.clip(enhanced, 0.0, 1.0)
        
        # Get the colormap
        cmap = plt.get_cmap(colormap_name)
        
        # Apply the colormap
        rgb_map = cmap(depth_map)[..., :3]
        
        return rgb_map
    
    def _validate_params(self, params):
        """Validate and adjust parameters."""
        # Validate mode
        valid_modes = ['linear', 'inverse', 'focal']
        if params.get('mode') not in valid_modes:
            logger.warning(f"Invalid mode '{params.get('mode')}', defaulting to 'linear'")
            params['mode'] = 'linear'
            
        # Validate visualization
        valid_visualizations = ['grayscale', 'color', 'heatmap']
        if params.get('visualization') not in valid_visualizations:
            logger.warning(f"Invalid visualization '{params.get('visualization')}', defaulting to 'grayscale'")
            params['visualization'] = 'grayscale'
        
        # Validate depth range
        if params.get('min_depth', 0) >= params.get('max_depth', 1):
            logger.warning("Invalid depth range, min_depth must be less than max_depth")
            params['min_depth'] = 0.0
            params['max_depth'] = 1.0
            
        # Validate focal plane position (0-1)
        if params.get('focal_plane', 0) < 0 or params.get('focal_plane', 0) > 1:
            logger.warning("Invalid focal_plane value, should be between 0-1")
            params['focal_plane'] = 0.5
            
        # Validate focal range (positive)
        if params.get('focal_range', 0) <= 0:
            logger.warning("Invalid focal_range value, defaulting to 0.2")
            params['focal_range'] = 0.2
            
        # Validate smoothing (non-negative)
        if params.get('smoothing', 0) < 0:
            logger.warning("Invalid smoothing value, defaulting to 0.0")
            params['smoothing'] = 0.0
            
        # Validate enhance_contrast (positive)
        if params.get('enhance_contrast', 0) <= 0:
            logger.warning("Invalid enhance_contrast value, defaulting to 1.0")
            params['enhance_contrast'] = 1.0
            
        return params