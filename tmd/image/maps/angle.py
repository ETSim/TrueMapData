"""
Enhanced angle map generator.

This module provides a generator for creating advanced angle/slope maps from height maps,
with multiple visualization modes and enhanced customization options.
"""
import logging
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from .base_generator import MapGenerator

logger = logging.getLogger(__name__)

class AngleMapGenerator(MapGenerator):
    """Enhanced generator for Angle/Slope maps with multiple visualization modes."""
    
    def __init__(
        self, 
        mode: str = "gradient",
        max_angle: float = 90.0,
        min_angle: float = 0.0,
        normalize: bool = True,
        smoothing: float = 0.5,
        highlight_range: tuple = None,
        highlight_color: tuple = (1.0, 1.0, 1.0),
        aspect_direction: bool = False,
        custom_gradient: list = None,
        terrain_class: bool = False,
        edge_preservation: float = 0.5,
        **kwargs
    ):
        """
        Initialize the enhanced angle map generator.
        
        Args:
            mode: Visualization mode ('gradient', 'binary', 'hypsometric', 'aspect', 
                  'classified', 'contour', 'custom')
            max_angle: Maximum angle (in degrees) to represent as white (for gradient mode)
            min_angle: Minimum angle (in degrees) to represent as black
            normalize: Whether to normalize the height map before processing
            smoothing: Amount of smoothing to apply (0-5, where 0 means no smoothing)
            highlight_range: Tuple of (min_angle, max_angle) to highlight in degrees
            highlight_color: RGB tuple for highlight color
            aspect_direction: Whether to include directional information (aspect)
            custom_gradient: List of (value, color) tuples for custom gradient
            terrain_class: Whether to use terrain classification mode 
            edge_preservation: How much to preserve edges when smoothing (0-1)
            **kwargs: Additional default parameters
        """
        super().__init__(
            mode=mode,
            max_angle=max_angle,
            min_angle=min_angle,
            normalize=normalize,
            smoothing=smoothing,
            highlight_range=highlight_range,
            highlight_color=highlight_color,
            aspect_direction=aspect_direction,
            custom_gradient=custom_gradient,
            terrain_class=terrain_class,
            edge_preservation=edge_preservation,
            **kwargs
        )
    
    def generate(self, height_map: np.ndarray, **kwargs) -> np.ndarray:
        """
        Generate an angle map from a height map.
        
        Args:
            height_map: Input height map
            **kwargs: Generation parameters (see __init__ for details)
            
        Returns:
            Angle map as numpy array (0-1 range or RGB if colored mode)
        """
        # Get parameters
        params = self._get_params(**kwargs)
        mode = params.get('mode', 'gradient')
        max_angle = float(params.get('max_angle', 90.0))
        min_angle = float(params.get('min_angle', 0.0))
        normalize = bool(params.get('normalize', True))
        smoothing = float(params.get('smoothing', 0.5))
        highlight_range = params.get('highlight_range')
        highlight_color = params.get('highlight_color', (1.0, 1.0, 1.0))
        aspect_direction = bool(params.get('aspect_direction', False))
        custom_gradient = params.get('custom_gradient')
        terrain_class = bool(params.get('terrain_class', False))
        edge_preservation = float(params.get('edge_preservation', 0.5))
        
        # Get metadata for scaling
        metadata = kwargs.get('metadata', {}) or {}
        
        # Prepare height map
        height_map_norm = self._prepare_height_map(height_map, normalize=normalize)
        
        try:
            # Calculate cell size based on metadata if available
            cell_size_x, cell_size_y = self._get_cell_size(height_map_norm, metadata)
                
            # Apply adaptive edge-preserving smoothing if requested
            if smoothing > 0:
                # Edge-preserving smoothing using bilateral filter approximation
                if edge_preservation > 0:
                    try:
                        # Calculate gradient magnitude for edge detection
                        dx = ndimage.sobel(height_map_norm, axis=1)
                        dy = ndimage.sobel(height_map_norm, axis=0)
                        gradient_mag = np.sqrt(dx**2 + dy**2)
                        
                        # Create edge weights (higher for strong edges)
                        edge_weights = np.exp(-gradient_mag / edge_preservation)
                        
                        # Apply weighted smoothing (less smoothing at edges)
                        smooth_height = ndimage.gaussian_filter(
                            height_map_norm, 
                            sigma=smoothing, 
                            mode='nearest'
                        )
                        
                        # Blend original and smoothed based on edge weights
                        processed_height = edge_weights * height_map_norm + (1 - edge_weights) * smooth_height
                    except Exception as e:
                        logger.warning(f"Edge-preserving smoothing failed: {e}, using standard smoothing")
                        processed_height = ndimage.gaussian_filter(height_map_norm, sigma=smoothing)
                else:
                    # Standard Gaussian smoothing
                    processed_height = ndimage.gaussian_filter(height_map_norm, sigma=smoothing)
            else:
                processed_height = height_map_norm
                
            # Calculate gradients and terrain properties
            terrain_data = self._calculate_terrain_properties(
                processed_height, 
                cell_size_x, 
                cell_size_y
            )
            
            slope_deg = terrain_data['slope_deg']
            aspect = terrain_data['aspect']
            
            # Generate the appropriate map based on the selected mode
            if mode == 'gradient':
                # Basic slope gradient
                return self._generate_gradient_map(
                    slope_deg, 
                    min_angle, 
                    max_angle, 
                    highlight_range, 
                    highlight_color
                )
                
            elif mode == 'binary':
                # Binary threshold map
                threshold = params.get('threshold', 10.0)  # Default 10 degrees
                binary_map = (slope_deg >= threshold).astype(np.float32)
                return binary_map
                
            elif mode == 'hypsometric':
                # Colored hypsometric map
                return self._generate_hypsometric_map(
                    slope_deg, 
                    min_angle, 
                    max_angle, 
                    custom_gradient
                )
                
            elif mode == 'aspect':
                # Aspect direction map (colored by direction)
                return self._generate_aspect_map(aspect, slope_deg)
                
            elif mode == 'classified':
                # Classified terrain map
                return self._generate_classified_map(slope_deg, aspect)
                
            elif mode == 'contour':
                # Contour map with angle bands
                contour_interval = params.get('contour_interval', 10.0)
                return self._generate_contour_map(slope_deg, contour_interval)
                
            elif mode == 'custom' and custom_gradient:
                # Custom gradient mapping
                return self._apply_custom_gradient(slope_deg, custom_gradient)
                
            elif terrain_class:
                # Terrain classification
                return self._classify_terrain(slope_deg, aspect)
                
            else:
                # Default gradient mode
                return self._generate_gradient_map(
                    slope_deg, 
                    min_angle, 
                    max_angle, 
                    highlight_range, 
                    highlight_color
                )
            
        except Exception as e:
            logger.error(f"Error generating angle map: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            
            # Return a uniform image as fallback
            return np.ones_like(height_map_norm) * 0.25
    
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
    
    def _calculate_terrain_properties(self, height_map, cell_size_x, cell_size_y):
        """Calculate slope, aspect, and other terrain properties."""
        # Calculate gradients (use Sobel for more accurate gradient)
        try:
            dx = ndimage.sobel(height_map, axis=1) / (8.0 * cell_size_x)
            dy = ndimage.sobel(height_map, axis=0) / (8.0 * cell_size_y)
        except Exception:
            # Fallback to numpy gradient
            dx, dy = np.gradient(height_map, cell_size_x, cell_size_y)
        
        # Calculate slope in radians and degrees
        slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
        slope_deg = np.degrees(slope_rad)
        
        # Calculate aspect (direction of steepest descent)
        # Aspect: 0 = East, π/2 = North, π = West, 3π/2 = South
        aspect = np.arctan2(-dy, -dx)
        
        # Convert to 0-360 degrees for easier interpretation
        aspect_deg = np.degrees(aspect) % 360
        
        # Second derivatives for curvature
        try:
            dxx = ndimage.sobel(dx, axis=1) / (8.0 * cell_size_x)
            dyy = ndimage.sobel(dy, axis=0) / (8.0 * cell_size_y)
            dxy = ndimage.sobel(dx, axis=0) / (8.0 * cell_size_y)
            
            # Calculate plan curvature (perpendicular to direction of steepest descent)
            # Simplified version
            plan_curvature = (dxx * (dy**2) - 2 * dxy * dx * dy + dyy * (dx**2)) / (dx**2 + dy**2 + 1e-6)
        except Exception:
            plan_curvature = np.zeros_like(height_map)
        
        return {
            'dx': dx,
            'dy': dy,
            'slope_rad': slope_rad,
            'slope_deg': slope_deg,
            'aspect': aspect,
            'aspect_deg': aspect_deg,
            'plan_curvature': plan_curvature
        }
    
    def _generate_gradient_map(self, slope_deg, min_angle, max_angle, 
                             highlight_range=None, highlight_color=(1.0, 1.0, 1.0)):
        """Generate a gradient slope map with optional highlighted range."""
        # Normalize to 0-1 range based on min/max angles
        angle_range = max_angle - min_angle
        if angle_range <= 0:
            angle_range = 90.0
            
        angle_map = (slope_deg - min_angle) / angle_range
        angle_map = np.clip(angle_map, 0.0, 1.0)
        
        # Apply highlight if requested
        if highlight_range is not None:
            try:
                min_highlight, max_highlight = highlight_range
                highlight_mask = ((slope_deg >= min_highlight) & 
                                 (slope_deg <= max_highlight))
                
                # If RGB output requested with highlight
                if any(c != 1.0 for c in highlight_color):
                    # Create RGB map
                    rgb_map = np.zeros((*angle_map.shape, 3), dtype=np.float32)
                    
                    # Set grayscale for non-highlighted areas
                    rgb_map[..., 0] = angle_map
                    rgb_map[..., 1] = angle_map
                    rgb_map[..., 2] = angle_map
                    
                    # Apply highlight color
                    for i in range(3):
                        rgb_map[highlight_mask, i] = highlight_color[i] * angle_map[highlight_mask]
                    
                    return rgb_map
                else:
                    # Just boost the brightness in the highlighted range
                    boost_map = angle_map.copy()
                    boost_map[highlight_mask] = angle_map[highlight_mask] * 1.5
                    return np.clip(boost_map, 0.0, 1.0)
            except Exception as e:
                logger.warning(f"Error applying highlight: {e}")
        
        return angle_map
    
    def _generate_hypsometric_map(self, slope_deg, min_angle, max_angle, custom_gradient=None):
        """Generate a colored hypsometric slope map."""
        try:
            # Create RGB output map
            rows, cols = slope_deg.shape
            rgb_map = np.zeros((rows, cols, 3), dtype=np.float32)
            
            # Normalize slope to 0-1 for color mapping
            normalized_slope = np.clip((slope_deg - min_angle) / (max_angle - min_angle), 0, 1)
            
            # Define color stops for hypsometric tinting
            if custom_gradient:
                # Use custom gradient if provided
                color_stops = sorted(custom_gradient)
            else:
                # Default hypsometric colors for slopes
                color_stops = [
                    (0.0, (0.0, 0.5, 0.0)),    # Dark green for flat areas
                    (0.2, (0.0, 0.8, 0.0)),    # Green
                    (0.4, (1.0, 1.0, 0.0)),    # Yellow
                    (0.6, (1.0, 0.5, 0.0)),    # Orange
                    (0.8, (1.0, 0.0, 0.0)),    # Red
                    (1.0, (0.6, 0.0, 0.0))     # Dark red for steep areas
                ]
            
            # Apply the color gradient
            for i in range(len(color_stops) - 1):
                # Get current and next stop
                curr_val, curr_color = color_stops[i]
                next_val, next_color = color_stops[i + 1]
                
                # Create mask for this segment
                if i == len(color_stops) - 2:
                    # Include the endpoint in the last segment
                    mask = (normalized_slope >= curr_val) & (normalized_slope <= next_val)
                else:
                    mask = (normalized_slope >= curr_val) & (normalized_slope < next_val)
                
                # Skip if no pixels in this range
                if not np.any(mask):
                    continue
                
                # Calculate interpolation factor
                segment_range = next_val - curr_val
                interp = ((normalized_slope[mask] - curr_val) / segment_range) if segment_range > 0 else 0
                
                # Interpolate colors for each channel
                for c in range(3):
                    rgb_map[mask, c] = (
                        curr_color[c] + interp * (next_color[c] - curr_color[c])
                    )
            
            return rgb_map
            
        except Exception as e:
            logger.error(f"Error generating hypsometric map: {e}")
            # Return a grayscale map as fallback
            return np.clip((slope_deg - min_angle) / (max_angle - min_angle), 0, 1)
    
    def _generate_aspect_map(self, aspect, slope_deg, min_slope=2.0):
        """Generate an aspect direction map colored by direction."""
        try:
            # Create RGB output map
            rows, cols = aspect.shape
            rgb_map = np.zeros((rows, cols, 3), dtype=np.float32)
            
            # Convert aspect to 0-1 range for color mapping
            normalized_aspect = (np.degrees(aspect) % 360) / 360.0
            
            # HSV-style coloring where hue is determined by aspect
            # Flat areas (low slope) will be darker
            slope_mask = slope_deg >= min_slope
            
            # Create HSV-like mapping
            h = normalized_aspect
            s = np.ones_like(h)
            v = np.ones_like(h) * 0.3  # Base brightness for flat areas
            
            # Increase brightness for sloped areas
            v[slope_mask] = np.minimum(1.0, 0.3 + 0.7 * (slope_deg[slope_mask] / 90.0))
            
            # Convert HSV to RGB
            # This is a simplified conversion
            h_sector = (h * 6).astype(int) % 6
            f = (h * 6) - h_sector
            p = v * (1 - s)
            q = v * (1 - f * s)
            t = v * (1 - (1 - f) * s)
            
            # Apply colors based on sector
            for sector in range(6):
                mask = h_sector == sector
                if sector == 0:
                    rgb_map[mask, 0] = v[mask]
                    rgb_map[mask, 1] = t[mask]
                    rgb_map[mask, 2] = p[mask]
                elif sector == 1:
                    rgb_map[mask, 0] = q[mask]
                    rgb_map[mask, 1] = v[mask]
                    rgb_map[mask, 2] = p[mask]
                elif sector == 2:
                    rgb_map[mask, 0] = p[mask]
                    rgb_map[mask, 1] = v[mask]
                    rgb_map[mask, 2] = t[mask]
                elif sector == 3:
                    rgb_map[mask, 0] = p[mask]
                    rgb_map[mask, 1] = q[mask]
                    rgb_map[mask, 2] = v[mask]
                elif sector == 4:
                    rgb_map[mask, 0] = t[mask]
                    rgb_map[mask, 1] = p[mask]
                    rgb_map[mask, 2] = v[mask]
                else:  # sector == 5
                    rgb_map[mask, 0] = v[mask]
                    rgb_map[mask, 1] = p[mask]
                    rgb_map[mask, 2] = q[mask]
            
            return rgb_map
            
        except Exception as e:
            logger.error(f"Error generating aspect map: {e}")
            # Return a grayscale map as fallback
            return (np.degrees(aspect) % 360) / 360.0
    
    def _generate_classified_map(self, slope_deg, aspect):
        """Generate a classified terrain map based on slope and aspect."""
        try:
            # Create RGB output map
            rows, cols = slope_deg.shape
            rgb_map = np.zeros((rows, cols, 3), dtype=np.float32)
            
            # Define slope classes
            slope_classes = [
                (0, 5, "Flat", (0.0, 0.3, 0.0)),           # Dark green
                (5, 10, "Gentle", (0.0, 0.6, 0.0)),        # Medium green
                (10, 20, "Moderate", (0.0, 0.9, 0.0)),     # Bright green
                (20, 30, "Steep", (0.8, 0.8, 0.0)),        # Yellow
                (30, 45, "Very steep", (1.0, 0.4, 0.0)),   # Orange
                (45, 90, "Extreme", (0.8, 0.0, 0.0))       # Red
            ]
            
            # Apply classification
            for i, (min_slope, max_slope, name, color) in enumerate(slope_classes):
                mask = (slope_deg >= min_slope) & (slope_deg < max_slope)
                if i == len(slope_classes) - 1:  # Include max value in the last class
                    mask = (slope_deg >= min_slope)
                
                for c in range(3):
                    rgb_map[mask, c] = color[c]
            
            # Adjust brightness by aspect (North-facing slopes darker)
            aspect_deg = np.degrees(aspect) % 360
            north_facing = ((aspect_deg >= 315) | (aspect_deg < 45))
            south_facing = ((aspect_deg >= 135) & (aspect_deg < 225))
            
            # Darken north-facing slopes
            rgb_map[north_facing] *= 0.7
            
            # Brighten south-facing slopes
            brightness_boost = np.minimum(1.5, 1.0 / np.maximum(0.01, rgb_map[south_facing]))
            rgb_map[south_facing] *= brightness_boost.reshape(-1, 1)
            
            return np.clip(rgb_map, 0, 1)
            
        except Exception as e:
            logger.error(f"Error generating classified map: {e}")
            # Return a grayscale map as fallback
            return np.clip(slope_deg / 90.0, 0, 1)
    
    def _generate_contour_map(self, slope_deg, contour_interval=10.0):
        """Generate a contour map with angle bands."""
        try:
            # Create a base grayscale slope map
            slope_map = np.clip(slope_deg / 90.0, 0, 1)
            
            # Round slope values to the nearest contour interval
            contoured = np.round(slope_deg / contour_interval) * contour_interval
            
            # Find contour edges
            edges = np.abs(ndimage.sobel(contoured, axis=0)) + np.abs(ndimage.sobel(contoured, axis=1))
            edges = edges > 0
            
            # Create output map
            contour_map = slope_map.copy()
            contour_map[edges] = 0.0  # Dark contour lines
            
            return contour_map
            
        except Exception as e:
            logger.error(f"Error generating contour map: {e}")
            # Return a grayscale map as fallback
            return np.clip(slope_deg / 90.0, 0, 1)
    
    def _apply_custom_gradient(self, slope_deg, custom_gradient):
        """Apply a custom color gradient to the slope map."""
        try:
            # Ensure gradient is properly sorted
            gradient = sorted(custom_gradient, key=lambda x: x[0])
            
            # Create RGB output map
            rows, cols = slope_deg.shape
            rgb_map = np.zeros((rows, cols, 3), dtype=np.float32)
            
            # Apply the gradient for each color stop
            for i in range(len(gradient) - 1):
                curr_val, curr_color = gradient[i]
                next_val, next_color = gradient[i + 1]
                
                # Create mask for this segment
                mask = (slope_deg >= curr_val) & (slope_deg < next_val)
                if i == len(gradient) - 2:  # Include max value in last segment
                    mask = (slope_deg >= curr_val) & (slope_deg <= next_val)
                
                # Skip if no pixels in this range
                if not np.any(mask):
                    continue
                
                # Interpolate between colors
                factor = (slope_deg[mask] - curr_val) / (next_val - curr_val)
                
                for c in range(3):
                    rgb_map[mask, c] = curr_color[c] + factor * (next_color[c] - curr_color[c])
            
            return np.clip(rgb_map, 0, 1)
            
        except Exception as e:
            logger.error(f"Error applying custom gradient: {e}")
            # Return a grayscale map as fallback
            return np.clip(slope_deg / 90.0, 0, 1)
    
    def _classify_terrain(self, slope_deg, aspect):
        """Classify terrain into standard categories."""
        try:
            # Create terrain classification map
            terrain_map = np.zeros_like(slope_deg, dtype=np.int8)
            
            # Standard terrain classification based on slope
            terrain_map[(slope_deg < 2)] = 1      # Flat
            terrain_map[(slope_deg >= 2) & (slope_deg < 5)] = 2    # Nearly level
            terrain_map[(slope_deg >= 5) & (slope_deg < 10)] = 3   # Gently sloping
            terrain_map[(slope_deg >= 10) & (slope_deg < 15)] = 4  # Moderately sloping
            terrain_map[(slope_deg >= 15) & (slope_deg < 30)] = 5  # Steep
            terrain_map[(slope_deg >= 30) & (slope_deg < 45)] = 6  # Very steep
            terrain_map[(slope_deg >= 45)] = 7     # Extreme
            
            # Create RGB visualization
            # Define colors for each terrain class
            colors = [
                (0.0, 0.0, 0.0),      # 0: Unknown/error (black)
                (0.0, 0.3, 0.0),      # 1: Flat (dark green)
                (0.0, 0.5, 0.0),      # 2: Nearly level (green)
                (0.0, 0.7, 0.0),      # 3: Gently sloping (light green)
                (0.7, 0.7, 0.0),      # 4: Moderately sloping (yellow)
                (1.0, 0.5, 0.0),      # 5: Steep (orange)
                (0.8, 0.0, 0.0),      # 6: Very steep (red)
                (0.5, 0.0, 0.0)       # 7: Extreme (dark red)
            ]
            
            # Create RGB map
            rows, cols = slope_deg.shape
            rgb_map = np.zeros((rows, cols, 3), dtype=np.float32)
            
            # Apply colors
            for i, color in enumerate(colors):
                mask = terrain_map == i
                for c in range(3):
                    rgb_map[mask, c] = color[c]
            
            return rgb_map
            
        except Exception as e:
            logger.error(f"Error in terrain classification: {e}")
            # Return a grayscale map as fallback
            return np.clip(slope_deg / 90.0, 0, 1)
    
    def _validate_params(self, params):
        """Validate and adjust parameters."""
        # Validate mode
        valid_modes = ['gradient', 'binary', 'hypsometric', 'aspect', 
                       'classified', 'contour', 'custom']
        if params.get('mode') not in valid_modes:
            logger.warning(f"Invalid mode '{params.get('mode')}', defaulting to 'gradient'")
            params['mode'] = 'gradient'
        
        # Validate max_angle (positive, max 90)
        if params.get('max_angle', 0) <= 0:
            logger.warning("Invalid max_angle value, defaulting to 90.0")
            params['max_angle'] = 90.0
        elif params.get('max_angle', 0) > 90.0:
            logger.warning("max_angle too large, capping at 90.0")
            params['max_angle'] = 90.0
            
        # Validate min_angle (non-negative, less than max_angle)
        if params.get('min_angle', 0) < 0:
            logger.warning("Invalid min_angle value, defaulting to 0.0")
            params['min_angle'] = 0.0
        elif params.get('min_angle', 0) >= params.get('max_angle', 90.0):
            logger.warning("min_angle must be less than max_angle, setting to 0.0")
            params['min_angle'] = 0.0
            
        # Validate smoothing (non-negative)
        if params.get('smoothing', 0) < 0:
            logger.warning("Invalid smoothing value, defaulting to 0.5")
            params['smoothing'] = 0.5
        elif params.get('smoothing', 0) > 5:
            logger.warning("smoothing too large, capping at 5.0")
            params['smoothing'] = 5.0
            
        # Validate highlight_range if present
        if params.get('highlight_range') is not None:
            try:
                min_highlight, max_highlight = params['highlight_range']
                if min_highlight < 0 or max_highlight > 90 or min_highlight >= max_highlight:
                    logger.warning("Invalid highlight_range value, ignoring")
                    params['highlight_range'] = None
            except (ValueError, TypeError):
                logger.warning("Invalid highlight_range format, ignoring")
                params['highlight_range'] = None
                
        # Validate highlight_color if present
        if params.get('highlight_color') is not None:
            try:
                r, g, b = params['highlight_color']
                params['highlight_color'] = (
                    max(0, min(1, float(r))),
                    max(0, min(1, float(g))),
                    max(0, min(1, float(b)))
                )
            except (ValueError, TypeError):
                logger.warning("Invalid highlight_color format, using default")
                params['highlight_color'] = (1.0, 1.0, 1.0)
                
        # Validate custom_gradient if present
        if params.get('custom_gradient') is not None:
            try:
                # Ensure each entry is (value, (r,g,b)) format
                validated_gradient = []
                for entry in params['custom_gradient']:
                    val, color = entry
                    r, g, b = color
                    validated_gradient.append((
                        float(val),
                        (max(0, min(1, float(r))),
                         max(0, min(1, float(g))),
                         max(0, min(1, float(b))))
                    ))
                params['custom_gradient'] = validated_gradient
            except (ValueError, TypeError, IndexError):
                logger.warning("Invalid custom_gradient format, ignoring")
                params['custom_gradient'] = None
                
        # Validate edge_preservation (0-1)
        if params.get('edge_preservation', 0) < 0:
            logger.warning("Invalid edge_preservation value, defaulting to 0.5")
            params['edge_preservation'] = 0.5
        elif params.get('edge_preservation', 0) > 1:
            logger.warning("edge_preservation too large, capping at 1.0")
            params['edge_preservation'] = 1.0
            
        return params