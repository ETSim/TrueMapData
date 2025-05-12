"""Roughness map generator.

This module provides a generator for creating roughness maps from height maps,
which highlight areas of high frequency detail for PBR materials.
"""
import logging
import numpy as np

from .base_generator import MapGenerator

logger = logging.getLogger(__name__)

class RoughnessMapGenerator(MapGenerator):
    """Generator for Roughness maps."""
    
    def __init__(self, kernel_size: int = 3, scale: float = 1.0, **kwargs):
        """
        Initialize the roughness map generator.
        
        Args:
            kernel_size: Size of kernel for roughness detection
            scale: Strength multiplier for roughness effect
            **kwargs: Additional default parameters
        """
        super().__init__(kernel_size=kernel_size, scale=scale, **kwargs)
    
    def generate(self, height_map: np.ndarray, **kwargs) -> np.ndarray:
        """
        Generate a roughness map from a height map.
        
        Args:
            height_map: Input height map
            **kwargs: Generation parameters including:
                - kernel_size: Size of kernel for roughness detection
                - scale: Strength multiplier
            
        Returns:
            Roughness map as numpy array (0-1 range)
        """
        # Extract parameters
        params = self._get_params(**kwargs)
        kernel_size = params['kernel_size']
        scale = params['scale']
        
        # Normalize or prepare height map
        height_map_norm = self._prepare_height_map(height_map)
        
        try:
            import cv2
            # Use OpenCV Laplacian for high-frequency detection
            height_array = height_map_norm.astype(np.float32)
            laplacian = cv2.Laplacian(height_array, cv2.CV_32F, ksize=kernel_size)
            roughness = np.abs(laplacian) * scale
        except ImportError:
            # Fallback to numpy gradient if OpenCV not available
            dx, dy = np.gradient(height_map_norm)
            roughness = np.sqrt(dx * dx + dy * dy) * scale
        
        # Normalize to [0,1]
        rmin, rmax = roughness.min(), roughness.max()
        if rmax > rmin:
            return (roughness - rmin) / (rmax - rmin)
        return np.zeros_like(roughness)
    
    def _validate_params(self, params):
        """Validate and adjust parameters."""
        # Ensure kernel_size is odd
        if params.get('kernel_size', 0) % 2 == 0:
            params['kernel_size'] = max(3, params['kernel_size'] + 1)
        # Ensure scale is positive
        if params.get('scale', 0) <= 0:
            params['scale'] = 1.0
        return params
