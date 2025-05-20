"""
Parallax AO map generator.

This module provides a generator for creating parallax ambient occlusion maps,
which combine height-based ambient occlusion with slope-aware adjustments for enhanced depth perception.
"""
import logging
import numpy as np
from scipy import ndimage
from .base_generator import MapGenerator

logger = logging.getLogger(__name__)

class ParallaxAOMapGenerator(MapGenerator):
    """Generator for Parallax Ambient Occlusion maps with steep and slope configurations."""
    
    def __init__(
        self, 
        strength: float = 1.0,
        samples: int = 16,
        steep_threshold: float = 45.0,
        steep_multiplier: float = 2.0,
        slope_sensitivity: float = 0.5,
        shadow_softness: float = 1.0,
        max_distance: float = 0.05,
        directional_bias: float = 0.0,
        bias_direction: float = 315.0,  # Default: top-right light
        focus_range: tuple = None,  # Angle range to emphasize (min, max)
        multi_scale: bool = True,  # Use multi-resolution sampling
        cavity_emphasis: float = 1.0,  # Emphasize cavities/crevices
        **kwargs
    ):
        """
        Initialize the parallax AO map generator.
        
        Args:
            strength: Overall effect strength
            samples: Number of sample directions for AO calculation
            steep_threshold: Angle in degrees above which terrain is considered steep
            steep_multiplier: Effect multiplier for steep areas
            slope_sensitivity: How much slope affects the AO intensity (0-1)
            shadow_softness: Softness of shadow transitions
            max_distance: Maximum sampling distance as percentage of image size
            directional_bias: Bias strength toward a specific direction (0-1)
            bias_direction: Direction angle for bias in degrees (0=N, 90=E, etc.)
            focus_range: Tuple of (min_angle, max_angle) to emphasize in degrees
            multi_scale: Whether to use multi-resolution sampling for better quality
            cavity_emphasis: Factor to emphasize concave areas (crevices, valleys)
            **kwargs: Additional default parameters
        """
        super().__init__(
            strength=strength,
            samples=samples,
            steep_threshold=steep_threshold,
            steep_multiplier=steep_multiplier,
            slope_sensitivity=slope_sensitivity,
            shadow_softness=shadow_softness,
            max_distance=max_distance,
            directional_bias=directional_bias,
            bias_direction=bias_direction,
            focus_range=focus_range,
            multi_scale=multi_scale,
            cavity_emphasis=cavity_emphasis,
            **kwargs
        )
    
    def generate(self, height_map: np.ndarray, **kwargs) -> np.ndarray:
        """
        Generate a parallax AO map from a height map.
        
        Args:
            height_map: Input height map
            **kwargs: Generation parameters (see __init__ for details)
            
        Returns:
            Parallax AO map as numpy array (0-1 range)
        """
        # Get parameters
        params = self._get_params(**kwargs)
        strength = float(params.get('strength', 1.0))
        samples = int(params.get('samples', 16))
        steep_threshold = float(params.get('steep_threshold', 45.0))
        steep_multiplier = float(params.get('steep_multiplier', 2.0))
        slope_sensitivity = float(params.get('slope_sensitivity', 0.5))
        shadow_softness = float(params.get('shadow_softness', 1.0))
        max_distance = float(params.get('max_distance', 0.05))
        directional_bias = float(params.get('directional_bias', 0.0))
        bias_direction = float(params.get('bias_direction', 315.0))
        focus_range = params.get('focus_range')
        multi_scale = bool(params.get('multi_scale', True))
        cavity_emphasis = float(params.get('cavity_emphasis', 1.0))
        
        # Get metadata for scaling
        metadata = kwargs.get('metadata', {}) or {}
        
        # Prepare height map
        height_map_norm = self._prepare_height_map(height_map, normalize=True)
        
        try:
            logger.debug(f"Generating parallax AO map with {samples} samples")
            
            # Calculate physical dimensions
            cell_size_x, cell_size_y = self._get_cell_size(height_map_norm, metadata)
            
            # 1. Calculate terrain analysis maps (slope, aspect, curvature)
            terrain_data = self._analyze_terrain(height_map_norm, cell_size_x, cell_size_y)
            
            # Create curvature-based cavity map for emphasis if requested
            cavity_map = None
            if cavity_emphasis > 0:
                try:
                    # Use mean curvature as cavity indicator (negative = concave)
                    from .curvature import CurvatureMapGenerator
                    curvature_gen = CurvatureMapGenerator(mode='mean')
                    curvature = curvature_gen.generate(height_map_norm, metadata=metadata)
                    
                    # Isolate concave regions (values < 0.5 in the normalized map)
                    cavity_map = np.maximum(0, 0.5 - curvature) * 2  # Range 0-1
                    logger.debug("Curvature-based cavity map generated")
                except (ImportError, Exception) as e:
                    logger.warning(f"Could not generate cavity map: {e}")
                    # Fallback using simple local minima detection
                    height_blurred = ndimage.gaussian_filter(height_map_norm, sigma=2.0)
                    cavity_map = np.maximum(0, height_blurred - height_map_norm)
                    cavity_map = cavity_map / np.maximum(0.001, np.max(cavity_map))
            
            # 2. Calculate base AO map
            rows, cols = height_map_norm.shape
            ao_map = np.ones((rows, cols), dtype=np.float32)
            
            # Calculate sampling parameters
            max_dim = max(rows, cols)
            base_radius = int(max_distance * max_dim)
            base_radius = max(1, min(base_radius, max_dim // 8))
            
            # Add directional bias if requested
            if directional_bias > 0:
                bias_rad = np.radians(bias_direction)
                bias_x = np.cos(bias_rad)
                bias_y = np.sin(bias_rad)
            
            # For multi-scale sampling, use multiple radius values
            if multi_scale:
                # Sample at different scales for better quality
                radius_scales = [0.3, 0.6, 1.0] if base_radius > 3 else [1.0]
            else:
                radius_scales = [1.0]
                
            # Sample positions around a hemisphere with directional bias if needed
            sample_points = self._generate_sample_points(
                samples, 
                directional_bias,
                bias_direction
            )
            
            # Process each scale
            for scale in radius_scales:
                scale_weight = 1.0 / len(radius_scales)
                scale_radius = max(1, int(base_radius * scale))
                
                # Process each sample direction
                for i, (dx_sample, dy_sample) in enumerate(sample_points):
                    # Calculate direction-weighted radius map
                    radius_map = self._calculate_radius_map(
                        terrain_data,
                        dx_sample,
                        dy_sample,
                        scale_radius,
                        slope_sensitivity,
                        focus_range
                    )
                    
                    # Sample heights with dynamic radius
                    sampled_heights = self._sample_heights(
                        height_map_norm, 
                        dx_sample, 
                        dy_sample, 
                        radius_map
                    )
                    
                    # Calculate occlusion based on height differences
                    height_diff = sampled_heights - height_map_norm
                    occlusion = np.maximum(0, height_diff) / max(0.001, shadow_softness)
                    occlusion = np.minimum(1, occlusion)
                    
                    # Apply to AO map
                    ao_map -= (occlusion * scale_weight) / samples
            
            # 3. Apply steep adjustment
            # Create smooth transition around steep threshold
            slope_deg = terrain_data['slope_deg']
            steep_factor = np.clip((slope_deg - steep_threshold) / (90.0 - steep_threshold), 0, 1)
            
            # Apply non-linear curve for more natural transition
            steep_factor = steep_factor ** 0.7
            steep_influence = 1.0 + steep_factor * (steep_multiplier - 1.0)
            
            # 4. Finalize the map
            # Apply steep influence and overall strength
            parallax_ao = np.power(ao_map, steep_influence * strength)
            
            # Apply cavity emphasis if available
            if cavity_map is not None and cavity_emphasis > 0:
                # Emphasize cavities (concave areas)
                emphasis_factor = 1.0 + cavity_map * cavity_emphasis
                parallax_ao = np.power(parallax_ao, emphasis_factor)
            
            logger.debug(f"Parallax AO map generated successfully")
            
            # Ensure output is in valid range
            return np.clip(parallax_ao, 0.0, 1.0)
            
        except Exception as e:
            logger.error(f"Error generating parallax AO map: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            
            # Return a uniform image as fallback
            return np.ones_like(height_map_norm) * 0.5
    
    def _analyze_terrain(self, height_map, cell_size_x, cell_size_y):
        """Analyze terrain to extract slope, aspect, and curvature information."""
        try:
            # Apply light smoothing to reduce noise
            smooth_height = ndimage.gaussian_filter(height_map, sigma=0.7)
            
            # Calculate gradients using Sobel for better accuracy
            dx = ndimage.sobel(smooth_height, axis=1) / (8.0 * cell_size_x)
            dy = ndimage.sobel(smooth_height, axis=0) / (8.0 * cell_size_y)
            
            # Calculate slope
            slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
            slope_deg = np.degrees(slope_rad)
            slope_norm = np.clip(slope_deg / 90.0, 0, 1)
            
            # Calculate aspect (direction of steepest descent)
            # Aspect: 0 = North, π/2 = East, π = South, 3π/2 = West
            aspect = np.arctan2(dy, dx)
            
            # Calculate simple curvature indicator
            # Second derivatives
            dxx = ndimage.sobel(dx, axis=1) / (8.0 * cell_size_x)
            dyy = ndimage.sobel(dy, axis=0) / (8.0 * cell_size_y)
            
            # Laplacian approximation (positive = convex, negative = concave)
            laplacian = dxx + dyy
            
            # Return all terrain analysis data
            return {
                'dx': dx,
                'dy': dy,
                'slope_rad': slope_rad,
                'slope_deg': slope_deg,
                'slope_norm': slope_norm,
                'aspect': aspect,
                'laplacian': laplacian
            }
            
        except Exception as e:
            logger.warning(f"Error in terrain analysis, using fallback: {e}")
            # Return fallback values
            shape = height_map.shape
            return {
                'dx': np.zeros(shape),
                'dy': np.zeros(shape),
                'slope_rad': np.zeros(shape),
                'slope_deg': np.zeros(shape),
                'slope_norm': np.zeros(shape),
                'aspect': np.zeros(shape),
                'laplacian': np.zeros(shape)
            }
    
    def _get_cell_size(self, height_map, metadata):
        """Calculate cell size from metadata if available."""
        cell_size_x, cell_size_y = 1.0, 1.0
        
        if 'x_length' in metadata and 'y_length' in metadata:
            height, width = height_map.shape
            cell_size_x = metadata['x_length'] / width if width > 0 else 1.0
            cell_size_y = metadata['y_length'] / height if height > 0 else 1.0
        elif 'mmpp' in metadata:
            cell_size_x = cell_size_y = metadata['mmpp']
            
        return cell_size_x, cell_size_y
    
    def _generate_sample_points(self, samples, directional_bias, bias_direction):
        """Generate sample points with optional directional bias."""
        # Convert bias direction to radians
        bias_rad = np.radians(bias_direction)
        
        # Generate points distributed around a circle
        theta = np.linspace(0, 2 * np.pi, samples, endpoint=False)
        
        # Apply directional bias if needed
        if directional_bias > 0:
            # Adjust theta values to cluster around the bias direction
            # This creates more samples in the preferred direction
            theta = theta + directional_bias * np.sin(theta - bias_rad)
            
        # Convert to x,y coordinates
        x_samples = np.cos(theta)
        y_samples = np.sin(theta)
        
        return list(zip(x_samples, y_samples))
        
    def _calculate_radius_map(self, terrain_data, dx_sample, dy_sample, 
                             base_radius, slope_sensitivity, focus_range):
        """Calculate sampling radius map based on terrain characteristics."""
        slope_norm = terrain_data['slope_norm']
        aspect = terrain_data['aspect']
        
        # Base radius adjustment from slope
        radius_map = base_radius * (1.0 + slope_norm * slope_sensitivity)
        
        # Direction-aware adjustment: increased radius in the direction of the slope
        sample_dir = np.arctan2(dy_sample, dx_sample)
        dir_alignment = np.abs(np.cos(aspect - sample_dir))
        radius_map = radius_map * (1.0 + dir_alignment * 0.5)
        
        # Apply focus range if specified
        if focus_range is not None:
            min_angle, max_angle = focus_range
            slope_deg = terrain_data['slope_deg']
            
            # Create weight mask for focused angle range
            in_range = ((slope_deg >= min_angle) & (slope_deg <= max_angle))
            focus_weight = np.ones_like(slope_deg)
            focus_weight[in_range] = 2.0  # Double radius in focus range
            
            radius_map = radius_map * focus_weight
        
        # Ensure reasonable limits
        rows, cols = slope_norm.shape
        max_radius = min(rows//4, cols//4, 50)
        return np.clip(radius_map, 1, max_radius).astype(int)
        
    def _sample_heights(self, height_map, dx, dy, radius_map):
        """Sample heights using variable radius map."""
        rows, cols = height_map.shape
        y_coords, x_coords = np.mgrid[:rows, :cols]
        
        # Initialize output array
        sampled_heights = np.zeros_like(height_map)
        
        # Apply different offsets for each unique radius value
        # This is much faster than calculating for each pixel individually
        for radius in np.unique(radius_map):
            # Skip if no pixels have this radius
            mask = (radius_map == radius)
            if not np.any(mask):
                continue
            
            # Calculate offset for this radius
            x_offset = int(dx * radius)
            y_offset = int(dy * radius)
            
            # Calculate shifted coordinates
            x_shifted = np.clip(x_coords[mask] + x_offset, 0, cols-1)
            y_shifted = np.clip(y_coords[mask] + y_offset, 0, rows-1)
            
            # Get values at shifted positions
            sampled_heights[mask] = height_map[y_shifted, x_shifted]
        
        return sampled_heights
    
    def _validate_params(self, params):
        """Validate and adjust parameters."""
        # Validate strength (positive)
        if params.get('strength', 0) <= 0:
            logger.warning("Invalid strength value, defaulting to 1.0")
            params['strength'] = 1.0
            
        # Validate samples (positive integer)
        if not isinstance(params.get('samples', 0), int) or params.get('samples', 0) < 4:
            logger.warning(f"Invalid samples value ({params.get('samples')}), defaulting to 16")
            params['samples'] = 16
            
        # Validate steep_threshold (0-90)
        if params.get('steep_threshold', 0) < 0:
            logger.warning("Invalid steep_threshold value, defaulting to 0.0")
            params['steep_threshold'] = 0.0
        elif params.get('steep_threshold', 0) > 90:
            logger.warning("Invalid steep_threshold value, defaulting to 90.0")
            params['steep_threshold'] = 90.0
            
        # Validate steep_multiplier (positive)
        if params.get('steep_multiplier', 0) <= 0:
            logger.warning("Invalid steep_multiplier value, defaulting to 2.0")
            params['steep_multiplier'] = 2.0
            
        # Validate slope_sensitivity (0-1)
        if params.get('slope_sensitivity', 0) < 0:
            logger.warning("Invalid slope_sensitivity value, defaulting to 0.0")
            params['slope_sensitivity'] = 0.0
        elif params.get('slope_sensitivity', 0) > 1:
            logger.warning("Invalid slope_sensitivity value, defaulting to 1.0")
            params['slope_sensitivity'] = 1.0
            
        # Validate shadow_softness (positive)
        if params.get('shadow_softness', 0) <= 0:
            logger.warning("Invalid shadow_softness value, defaulting to 1.0")
            params['shadow_softness'] = 1.0
            
        # Validate max_distance (positive, reasonable percentage)
        if params.get('max_distance', 0) <= 0:
            logger.warning("Invalid max_distance value, defaulting to 0.05")
            params['max_distance'] = 0.05
        elif params.get('max_distance', 0) > 0.2:
            logger.warning("max_distance too large, capping at 0.2")
            params['max_distance'] = 0.2
            
        # Validate directional_bias (0-1)
        if params.get('directional_bias', 0) < 0:
            logger.warning("Invalid directional_bias value, defaulting to 0.0")
            params['directional_bias'] = 0.0
        elif params.get('directional_bias', 0) > 1:
            logger.warning("Invalid directional_bias value, defaulting to 1.0")
            params['directional_bias'] = 1.0
            
        # Validate bias_direction (0-360)
        if 'bias_direction' in params:
            params['bias_direction'] = params['bias_direction'] % 360.0
            
        # Validate focus_range if present
        if params.get('focus_range') is not None:
            try:
                min_angle, max_angle = params['focus_range']
                if min_angle < 0 or max_angle > 90 or min_angle >= max_angle:
                    logger.warning("Invalid focus_range value, ignoring")
                    params['focus_range'] = None
            except (ValueError, TypeError):
                logger.warning("Invalid focus_range format, ignoring")
                params['focus_range'] = None
                
        # Validate cavity_emphasis (non-negative)
        if params.get('cavity_emphasis', 0) < 0:
            logger.warning("Invalid cavity_emphasis value, defaulting to 1.0")
            params['cavity_emphasis'] = 1.0
            
        return params