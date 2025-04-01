"""
Ambient Occlusion map generator.

This module provides a generator for creating ambient occlusion maps from height maps,
which represent how exposed each point is to ambient lighting.
"""
import logging
import numpy as np

from .base_generator import MapGenerator

logger = logging.getLogger(__name__)

class AOMapGenerator(MapGenerator):
    """Generator for Ambient Occlusion maps."""
    
    def __init__(self, samples: int = 16, strength: float = 1.0, **kwargs):
        """
        Initialize the AO map generator.
        
        Args:
            samples: Number of samples for AO calculation
            strength: Strength of the ambient occlusion effect
            **kwargs: Additional default parameters
        """
        super().__init__(samples=samples, strength=strength, **kwargs)
    
    def generate(self, height_map: np.ndarray, **kwargs) -> np.ndarray:
        """
        Generate an ambient occlusion map from a height map.
        
        Args:
            height_map: Input height map
            **kwargs: Generation parameters including:
                - samples: Number of samples for AO calculation
                - strength: Strength of the AO effect
                - radius: Sampling radius (default: 10% of min dimension)
                - step_size: Step size for ray marching (default: 1.0)
            
        Returns:
            Ambient occlusion map as numpy array (0-1 range)
        """
        # Get parameters
        params = self._get_params(**kwargs)
        samples = params['samples']
        strength = params['strength']
        
        # Prepare height map
        height_map_norm = self._prepare_height_map(height_map, normalize=True)
        
        try:
            # Get image dimensions
            height, width = height_map_norm.shape
            
            # Initialize the AO map with ones (fully lit)
            ao_map = np.ones_like(height_map_norm)
            
            # Get additional parameters
            radius = params.get('radius', min(height, width) * 0.1)
            step_size = params.get('step_size', 1.0)
            
            # Sample rays at different angles
            angles = np.linspace(0, 2*np.pi, samples, endpoint=False)
            
            # Cast rays for each angle
            for angle in angles:
                # Calculate ray direction
                dx = np.cos(angle) * step_size
                dy = np.sin(angle) * step_size
                
                # Maximum distance to check
                max_distance = int(radius / step_size)
                
                # Set up initial positions and heights
                x_positions = np.arange(width)[:, np.newaxis] * np.ones(height)[np.newaxis, :]
                y_positions = np.ones(width)[:, np.newaxis] * np.arange(height)[np.newaxis, :]
                max_heights = height_map_norm.copy()
                
                # March rays
                for step in range(1, max_distance + 1):
                    # Update positions
                    x_positions += dx
                    y_positions += dy
                    
                    # Clip to image bounds
                    x_positions = np.clip(x_positions, 0, width - 1)
                    y_positions = np.clip(y_positions, 0, height - 1)
                    
                    # Sample heights
                    x_indices = np.round(x_positions).astype(int)
                    y_indices = np.round(y_positions).astype(int)
                    sampled_heights = height_map_norm[y_indices, x_indices]
                    
                    # Calculate occlusion
                    height_diff = sampled_heights - height_map_norm
                    occlusion = np.maximum(0, height_diff) * (1.0 - step / max_distance)
                    
                    # Update AO map
                    ao_map -= occlusion / samples * strength
                    
                    # Update max heights
                    max_heights = np.maximum(max_heights, sampled_heights)
            
            # Clamp to [0, 1] range
            return np.clip(ao_map, 0, 1)
            
        except Exception as e:
            logger.error(f"Error generating ambient occlusion map: {e}")
            import traceback
            traceback.print_exc()
            return np.ones_like(height_map)  # Return a default AO map (no occlusion)
    
    def _validate_params(self, params):
        """Validate and adjust parameters."""
        # Ensure samples is a reasonable number
        if params.get('samples', 0) < 4:
            params['samples'] = 16
            
        # Ensure strength is positive
        if params.get('strength', 0) <= 0:
            params['strength'] = 1.0
            
        return params
