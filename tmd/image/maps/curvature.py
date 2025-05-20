"""
Enhanced curvature map generator.

This module provides a generator for creating advanced curvature maps from height maps,
which identify convex (positive curvature) and concave (negative curvature) regions
with multiple visualization modes and feature detection capabilities.
"""
import logging
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from .base_generator import MapGenerator

logger = logging.getLogger(__name__)

class CurvatureMapGenerator(MapGenerator):
    """Enhanced generator for Curvature maps with advanced visualization options."""
    
    def __init__(
        self, 
        mode: str = "mean",
        visualization: str = "grayscale",
        scale: float = 1.0,
        normalize: bool = True,
        sigma: float = 1.0,
        multi_scale: bool = False,
        sigma_levels: list = None,
        highlight_features: bool = False,
        feature_threshold: float = 0.1,
        feature_colors: tuple = None,
        classify_features: bool = False,
        percentile_clip: tuple = (2, 98),
        enhance_contrast: float = 1.0,
        edge_width: int = 1,
        colormap: str = "coolwarm",
        **kwargs
    ):
        """
        Initialize the enhanced curvature map generator.
        
        Args:
            mode: Curvature type ('mean', 'gaussian', 'profile', 'planform', 'maximal', 'minimal')
            visualization: Output visualization type ('grayscale', 'color', 'classified', 
                          'edges', 'features', 'multi', 'divergent')
            scale: Scaling factor for curvature values
            normalize: Whether to normalize the height map before processing
            sigma: Gaussian smoothing radius for preprocessing
            multi_scale: Whether to use multi-scale analysis for better feature detection
            sigma_levels: List of smoothing radii for multi-scale analysis [small, medium, large]
            highlight_features: Whether to highlight detected features 
            feature_threshold: Threshold for feature detection (0-1)
            feature_colors: Colors for (ridges, valleys) features
            classify_features: Whether to classify surface features into categories
            percentile_clip: Percentiles for range clipping as (min, max)
            enhance_contrast: Contrast enhancement factor (1.0 = no enhancement)
            edge_width: Width of edges in 'edges' visualization mode
            colormap: Matplotlib colormap name for 'color' visualization
            **kwargs: Additional default parameters
        """
        # Initialize parameters
        if sigma_levels is None:
            sigma_levels = [0.7, 1.5, 3.0]
            
        if feature_colors is None:
            feature_colors = ((0.0, 0.7, 1.0), (1.0, 0.4, 0.0))  # Blue ridges, Orange valleys
            
        super().__init__(
            mode=mode,
            visualization=visualization,
            scale=scale,
            normalize=normalize,
            sigma=sigma,
            multi_scale=multi_scale,
            sigma_levels=sigma_levels,
            highlight_features=highlight_features,
            feature_threshold=feature_threshold,
            feature_colors=feature_colors,
            classify_features=classify_features,
            percentile_clip=percentile_clip,
            enhance_contrast=enhance_contrast,
            edge_width=edge_width,
            colormap=colormap,
            **kwargs
        )
    
    def generate(self, height_map: np.ndarray, **kwargs) -> np.ndarray:
        """
        Generate a curvature map from a height map.
        
        Args:
            height_map: Input height map
            **kwargs: Generation parameters (see __init__ for details)
            
        Returns:
            Curvature map as numpy array (grayscale or RGB depending on visualization mode)
        """
        # Get parameters
        params = self._get_params(**kwargs)
        mode = params.get('mode', 'mean')
        visualization = params.get('visualization', 'grayscale')
        scale = float(params.get('scale', 1.0))
        normalize = bool(params.get('normalize', True))
        sigma = float(params.get('sigma', 1.0))
        multi_scale = bool(params.get('multi_scale', False))
        sigma_levels = params.get('sigma_levels', [0.7, 1.5, 3.0])
        highlight_features = bool(params.get('highlight_features', False))
        feature_threshold = float(params.get('feature_threshold', 0.1))
        feature_colors = params.get('feature_colors', ((0.0, 0.7, 1.0), (1.0, 0.4, 0.0)))
        classify_features = bool(params.get('classify_features', False))
        percentile_clip = params.get('percentile_clip', (2, 98))
        enhance_contrast = float(params.get('enhance_contrast', 1.0))
        edge_width = int(params.get('edge_width', 1))
        colormap = params.get('colormap', 'coolwarm')
        
        # Get metadata for scaling
        metadata = kwargs.get('metadata', {}) or {}
        
        # Prepare height map
        height_map_norm = self._prepare_height_map(height_map, normalize=normalize)
        
        try:
            # Calculate cell size based on metadata if available
            cell_size_x, cell_size_y = self._get_cell_size(height_map_norm, metadata)
            
            # Process at multiple scales if requested
            if multi_scale:
                # Calculate curvature at multiple scales and combine
                curvature_maps = []
                weights = []
                
                for i, sigma_val in enumerate(sigma_levels):
                    # Use larger weight for finer scales
                    weight = 1.0 / (i + 1)
                    weights.append(weight)
                    
                    # Calculate curvature at this scale
                    smooth_height = ndimage.gaussian_filter(height_map_norm, sigma=sigma_val)
                    curvature = self._calculate_curvature(smooth_height, mode, cell_size_x, cell_size_y, scale)
                    curvature_maps.append(curvature)
                
                # Normalize weights
                total_weight = sum(weights)
                weights = [w / total_weight for w in weights]
                
                # Combine multiple scales with weighted average
                curvature = np.zeros_like(curvature_maps[0])
                for i, curv_map in enumerate(curvature_maps):
                    curvature += curv_map * weights[i]
            else:
                # Single-scale processing
                smooth_height = ndimage.gaussian_filter(height_map_norm, sigma=sigma)
                curvature = self._calculate_curvature(smooth_height, mode, cell_size_x, cell_size_y, scale)
            
            # Process the raw curvature map for visualization
            if visualization == 'grayscale':
                # Standard grayscale visualization
                return self._create_grayscale_visualization(
                    curvature, percentile_clip, enhance_contrast
                )
                
            elif visualization == 'color':
                # Colored visualization using divergent colormap
                return self._create_color_visualization(
                    curvature, percentile_clip, enhance_contrast, colormap
                )
                
            elif visualization == 'classified':
                # Classified surface features
                return self._create_classified_visualization(
                    curvature, height_map_norm, percentile_clip
                )
                
            elif visualization == 'edges':
                # Edge detection based on curvature
                return self._create_edge_visualization(
                    curvature, percentile_clip, edge_width
                )
                
            elif visualization == 'features':
                # Feature detection (ridges, valleys)
                return self._create_feature_visualization(
                    curvature, feature_threshold, feature_colors
                )
                
            elif visualization == 'multi':
                # Combined multi-visualization (features + grayscale)
                return self._create_multi_visualization(
                    curvature, height_map_norm, feature_threshold, feature_colors, 
                    percentile_clip, enhance_contrast
                )
                
            elif visualization == 'divergent':
                # Enhanced divergent visualization (separate processing for +/- curvature)
                return self._create_divergent_visualization(
                    curvature, percentile_clip, enhance_contrast
                )
                
            else:
                # Default to grayscale
                return self._create_grayscale_visualization(
                    curvature, percentile_clip, enhance_contrast
                )
                
        except Exception as e:
            logger.error(f"Error generating curvature map: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            
            # Return a uniform gray image as fallback
            return np.ones_like(height_map_norm) * 0.5
    
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
    
    def _calculate_curvature(self, height_map, mode, cell_size_x, cell_size_y, scale):
        """Calculate curvature of specified type from height map."""
        try:
            # Calculate first derivatives
            fx, fy = np.gradient(height_map, cell_size_x, cell_size_y)
            
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
                # Avoid division by zero
                safe_p = np.maximum(p, 1e-10)
                curvature = ((fx**2 * fxx + 2 * fx * fy * fxy + fy**2 * fyy) / 
                           (safe_p * np.sqrt(q)))
            elif mode == 'planform':
                # Plan curvature (perpendicular to the direction of steepest slope)
                # Avoid division by zero
                safe_p = np.maximum(p, 1e-10)
                curvature = ((fy**2 * fxx - 2 * fx * fy * fxy + fx**2 * fyy) / 
                           (safe_p * np.sqrt(q)))
            else:
                # Default to mean curvature
                curvature = 0.5 * ((1 + fy**2) * fxx - 2 * fx * fy * fxy + (1 + fx**2) * fyy) / (q**(3/2))
            
            # Apply scaling factor
            curvature = curvature * scale
            
            # Handle potential NaN or infinity values
            curvature = np.nan_to_num(curvature)
            
            return curvature
            
        except Exception as e:
            logger.error(f"Error in curvature calculation: {e}")
            # Return a zero array as fallback
            return np.zeros_like(height_map)
    
    def _create_grayscale_visualization(self, curvature, percentile_clip, enhance_contrast):
        """Create a standard grayscale visualization of curvature."""
        # Clip values to the specified percentile range
        min_pct, max_pct = percentile_clip
        curvature_range = np.percentile(curvature, [min_pct, max_pct])
        clip_min, clip_max = curvature_range[0], curvature_range[1]
        
        # Apply contrast enhancement if requested
        if enhance_contrast != 1.0:
            # Enhance the contrast by expanding the range
            mid = (clip_min + clip_max) / 2
            clip_min = mid - (mid - clip_min) * enhance_contrast
            clip_max = mid + (clip_max - mid) * enhance_contrast
        
        # Clip values
        curvature = np.clip(curvature, clip_min, clip_max)
        
        # Map from [-max, max] to [0, 1] for display
        # Ensure symmetric mapping for better visualization (0.5 is flat)
        abs_max = max(abs(clip_min), abs(clip_max))
        curvature_norm = 0.5 + (curvature / (2.0 * abs_max))
        
        # Ensure output is in valid range
        return np.clip(curvature_norm, 0.0, 1.0)
    
    def _create_color_visualization(self, curvature, percentile_clip, enhance_contrast, colormap):
        """Create a colored visualization using a divergent colormap."""
        try:
            # Get grayscale map first
            grayscale = self._create_grayscale_visualization(
                curvature, percentile_clip, enhance_contrast
            )
            
            # Create RGB visualization
            rows, cols = curvature.shape
            rgb_map = np.zeros((rows, cols, 3), dtype=np.float32)
            
            # Apply colormap (using matplotlib's colormaps if available)
            try:
                import matplotlib.pyplot as plt
                from matplotlib import cm
                
                # Get the colormap
                cmap = plt.get_cmap(colormap)
                
                # Apply the colormap
                rgb_map = cmap(grayscale)[..., :3]
                
            except (ImportError, ValueError):
                logger.warning(f"Could not apply colormap '{colormap}', using fallback")
                
                # Fallback: create a basic divergent colormap
                # Blue for valleys (negative curvature), red for ridges (positive)
                for y in range(rows):
                    for x in range(cols):
                        val = grayscale[y, x]
                        if val < 0.5:  # Valley - blue range
                            rgb_map[y, x, 0] = 0.0
                            rgb_map[y, x, 1] = val * 2  # Increase from 0-1
                            rgb_map[y, x, 2] = 1.0
                        else:  # Ridge - red range
                            rgb_map[y, x, 0] = 1.0
                            rgb_map[y, x, 1] = (1.0 - val) * 2  # Decrease from 1-0
                            rgb_map[y, x, 2] = 0.0
            
            return rgb_map
            
        except Exception as e:
            logger.error(f"Error creating color visualization: {e}")
            # Return grayscale as fallback
            return self._create_grayscale_visualization(
                curvature, percentile_clip, enhance_contrast
            )
    
    def _create_classified_visualization(self, curvature, height_map, percentile_clip):
        """Create a classified visualization of surface features."""
        try:
            # Clip to specified percentile range
            min_pct, max_pct = percentile_clip
            curvature_range = np.percentile(curvature, [min_pct, max_pct])
            clip_min, clip_max = curvature_range[0], curvature_range[1]
            curvature = np.clip(curvature, clip_min, clip_max)
            
            # Define classification thresholds
            # Normalize curvature to -1 to 1 range for consistent classification
            abs_max = max(abs(clip_min), abs(clip_max))
            curvature_norm = curvature / abs_max
            
            # Create RGB output map
            rows, cols = curvature.shape
            rgb_map = np.zeros((rows, cols, 3), dtype=np.float32)
            
            # Define feature classes and colors
            feature_classes = [
                # (min_curvature, max_curvature, name, color(r,g,b))
                (-1.0, -0.5, "Deep valley", (0.0, 0.0, 0.8)),      # Deep blue
                (-0.5, -0.2, "Valley", (0.0, 0.5, 1.0)),           # Light blue
                (-0.2, -0.05, "Shallow depression", (0.5, 0.8, 1.0)),  # Pale blue
                (-0.05, 0.05, "Flat/planar", (0.7, 0.7, 0.7)),     # Gray
                (0.05, 0.2, "Shallow ridge", (1.0, 0.8, 0.5)),     # Pale orange
                (0.2, 0.5, "Ridge", (1.0, 0.5, 0.0)),              # Orange
                (0.5, 1.0, "Sharp ridge", (0.8, 0.0, 0.0))         # Red
            ]
            
            # Apply classification
            for min_val, max_val, name, color in feature_classes:
                mask = (curvature_norm >= min_val) & (curvature_norm < max_val)
                for c in range(3):
                    rgb_map[mask, c] = color[c]
            
            # Add special detection for peaks and pits using local extrema
            try:
                from scipy import ndimage
                
                # Find local maxima (peaks)
                peak_mask = ndimage.maximum_filter(height_map, size=3) == height_map
                # Remove maxima at edges and plateaus
                structure = ndimage.generate_binary_structure(2, 2)
                peak_mask = peak_mask & ~ndimage.binary_erosion(peak_mask, structure=structure)
                
                # Find local minima (pits)
                pit_mask = ndimage.minimum_filter(height_map, size=3) == height_map
                # Remove minima at edges and plateaus
                pit_mask = pit_mask & ~ndimage.binary_erosion(pit_mask, structure=structure)
                
                # Highlight peaks and pits
                rgb_map[peak_mask] = [1.0, 1.0, 0.0]  # Yellow for peaks
                rgb_map[pit_mask] = [0.0, 1.0, 1.0]   # Cyan for pits
                
            except Exception as e:
                logger.warning(f"Could not detect peaks and pits: {e}")
            
            return rgb_map
            
        except Exception as e:
            logger.error(f"Error creating classified visualization: {e}")
            # Return grayscale as fallback
            return self._create_grayscale_visualization(
                curvature, percentile_clip, 1.0
            )
    
    def _create_edge_visualization(self, curvature, percentile_clip, edge_width):
        """Create an edge visualization based on curvature zero-crossings."""
        try:
            # Clip to percentile range
            min_pct, max_pct = percentile_clip
            curvature_range = np.percentile(curvature, [min_pct, max_pct])
            clip_min, clip_max = curvature_range[0], curvature_range[1]
            curvature = np.clip(curvature, clip_min, clip_max)
            
            # Find zero crossings in curvature (edges)
            rows, cols = curvature.shape
            edge_map = np.ones((rows, cols), dtype=np.float32)
            
            # Method 1: Sign changes in horizontal and vertical directions
            h_edges = ((curvature[:, :-1] * curvature[:, 1:]) <= 0)
            v_edges = ((curvature[:-1, :] * curvature[1:, :]) <= 0)
            
            # Extend to original size
            h_extended = np.zeros((rows, cols), dtype=bool)
            v_extended = np.zeros((rows, cols), dtype=bool)
            h_extended[:, :-1] = h_edges
            h_extended[:, -1] = h_extended[:, -2]  # Repeat last column
            v_extended[:-1, :] = v_edges
            v_extended[-1, :] = v_extended[-2, :]  # Repeat last row
            
            # Combine edges
            combined_edges = h_extended | v_extended
            
            # Thicken edges if requested
            if edge_width > 1:
                from scipy import ndimage
                combined_edges = ndimage.binary_dilation(
                    combined_edges, 
                    structure=np.ones((edge_width, edge_width))
                )
            
            # Apply to edge map (white background, black edges)
            edge_map[combined_edges] = 0.0
            
            return edge_map
            
        except Exception as e:
            logger.error(f"Error creating edge visualization: {e}")
            # Return grayscale as fallback
            return self._create_grayscale_visualization(
                curvature, percentile_clip, 1.0
            )
    
    def _create_feature_visualization(self, curvature, feature_threshold, feature_colors):
        """Create a feature visualization highlighting ridges and valleys."""
        try:
            # Get dimensions
            rows, cols = curvature.shape
            
            # Create a gradient-based normalization to deal with different scales
            abs_curvature = np.abs(curvature)
            threshold = np.mean(abs_curvature) + feature_threshold * np.std(abs_curvature)
            
            # Create RGB output map (white background)
            rgb_map = np.ones((rows, cols, 3), dtype=np.float32)
            
            # Color for ridges (positive curvature)
            ridge_color = feature_colors[0]
            # Color for valleys (negative curvature)
            valley_color = feature_colors[1]
            
            # Find significant features
            ridges = (curvature > threshold)
            valleys = (curvature < -threshold)
            
            # Apply colors
            for c in range(3):
                rgb_map[ridges, c] = ridge_color[c]
                rgb_map[valleys, c] = valley_color[c]
            
            return rgb_map
            
        except Exception as e:
            logger.error(f"Error creating feature visualization: {e}")
            # Return grayscale as fallback
            return self._create_grayscale_visualization(
                curvature, (2, 98), 1.0
            )
    
    def _create_multi_visualization(self, curvature, height_map, feature_threshold,
                                   feature_colors, percentile_clip, enhance_contrast):
        """Create a combined visualization with multiple techniques."""
        try:
            # Get grayscale base
            base_map = self._create_grayscale_visualization(
                curvature, percentile_clip, enhance_contrast
            )
            
            # Create RGB map initialized with the grayscale
            rows, cols = curvature.shape
            rgb_map = np.zeros((rows, cols, 3), dtype=np.float32)
            for c in range(3):
                rgb_map[..., c] = base_map
            
            # Get feature thresholds
            abs_curvature = np.abs(curvature)
            threshold = np.mean(abs_curvature) + feature_threshold * np.std(abs_curvature)
            
            # Colors
            ridge_color = feature_colors[0]
            valley_color = feature_colors[1]
            
            # Find features
            ridges = (curvature > threshold)
            valleys = (curvature < -threshold)
            
            # Add detected features with alpha blending
            for c in range(3):
                # Alpha blend features on top of the base map
                rgb_map[ridges, c] = rgb_map[ridges, c] * 0.3 + ridge_color[c] * 0.7
                rgb_map[valleys, c] = rgb_map[valleys, c] * 0.3 + valley_color[c] * 0.7
            
            # Find and highlight peaks and pits using local extrema if possible
            try:
                from scipy import ndimage
                
                # Find local maxima (peaks)
                peak_mask = ndimage.maximum_filter(height_map, size=3) == height_map
                peak_mask = peak_mask & ~ndimage.binary_erosion(peak_mask, structure=np.ones((3, 3)))
                
                # Find local minima (pits)
                pit_mask = ndimage.minimum_filter(height_map, size=3) == height_map
                pit_mask = pit_mask & ~ndimage.binary_erosion(pit_mask, structure=np.ones((3, 3)))
                
                # Add small highlight markers
                rgb_map[peak_mask] = [1.0, 1.0, 0.0]  # Yellow for peaks
                rgb_map[pit_mask] = [0.0, 1.0, 1.0]   # Cyan for pits
                
            except Exception as e:
                logger.warning(f"Could not detect peaks and pits: {e}")
            
            return np.clip(rgb_map, 0.0, 1.0)
            
        except Exception as e:
            logger.error(f"Error creating multi visualization: {e}")
            # Return grayscale as fallback
            return self._create_grayscale_visualization(
                curvature, percentile_clip, enhance_contrast
            )
    
    def _create_divergent_visualization(self, curvature, percentile_clip, enhance_contrast):
        """Create a divergent visualization with separate processing for +/- values."""
        try:
            # Split positive and negative curvature
            positive = np.maximum(0, curvature)
            negative = np.maximum(0, -curvature)
            
            # Process positive values (ridges)
            min_pct, max_pct = percentile_clip
            if np.any(positive > 0):
                pos_range = np.percentile(positive[positive > 0], [min_pct, max_pct])
                pos_min, pos_max = pos_range[0], pos_range[1]
                
                # Apply contrast enhancement if requested
                if enhance_contrast != 1.0:
                    pos_max = pos_max * enhance_contrast
                
                # Normalize positive values
                positive = np.clip(positive, pos_min, pos_max)
                positive = (positive - pos_min) / (pos_max - pos_min + 1e-10)
            else:
                positive = np.zeros_like(curvature)
            
            # Process negative values (valleys)
            if np.any(negative > 0):
                neg_range = np.percentile(negative[negative > 0], [min_pct, max_pct])
                neg_min, neg_max = neg_range[0], neg_range[1]
                
                # Apply contrast enhancement if requested
                if enhance_contrast != 1.0:
                    neg_max = neg_max * enhance_contrast
                
                # Normalize negative values
                negative = np.clip(negative, neg_min, neg_max)
                negative = (negative - neg_min) / (neg_max - neg_min + 1e-10)
            else:
                negative = np.zeros_like(curvature)
            
            # Create RGB output map
            rows, cols = curvature.shape
            rgb_map = np.zeros((rows, cols, 3), dtype=np.float32)
            
            # Fill with red (positive/ridge) and blue (negative/valley) components
            rgb_map[..., 0] = positive  # Red for ridges
            rgb_map[..., 2] = negative  # Blue for valleys
            
            # Add green component for better blending where both are present
            rgb_map[..., 1] = np.minimum(positive, negative) * 0.7
            
            return np.clip(rgb_map, 0.0, 1.0)
            
        except Exception as e:
            logger.error(f"Error creating divergent visualization: {e}")
            # Return grayscale as fallback
            return self._create_grayscale_visualization(
                curvature, percentile_clip, enhance_contrast
            )
    
    def _validate_params(self, params):
        """Validate and adjust parameters."""
        # Validate curvature mode
        valid_modes = ['mean', 'gaussian', 'profile', 'planform', 'maximal', 'minimal']
        if params.get('mode') not in valid_modes:
            logger.warning(f"Invalid mode '{params.get('mode')}', defaulting to 'mean'")
            params['mode'] = 'mean'
            
        # Validate visualization mode
        valid_visualizations = ['grayscale', 'color', 'classified', 'edges', 
                               'features', 'multi', 'divergent']
        if params.get('visualization') not in valid_visualizations:
            logger.warning(f"Invalid visualization '{params.get('visualization')}', defaulting to 'grayscale'")
            params['visualization'] = 'grayscale'
        
        # Validate scale (positive)
        if params.get('scale', 0) <= 0:
            logger.warning("Invalid scale value, defaulting to 1.0")
            params['scale'] = 1.0
            
        # Validate sigma (positive)
        if params.get('sigma', 0) <= 0:
            logger.warning("Invalid sigma value, defaulting to 1.0")
            params['sigma'] = 1.0
            
        # Validate sigma_levels if present
        if params.get('sigma_levels') is not None:
            try:
                sigma_levels = params['sigma_levels']
                if not all(s > 0 for s in sigma_levels) or len(sigma_levels) < 2:
                    logger.warning("Invalid sigma_levels, defaulting to [0.7, 1.5, 3.0]")
                    params['sigma_levels'] = [0.7, 1.5, 3.0]
            except (TypeError, ValueError):
                logger.warning("Invalid sigma_levels format, defaulting to [0.7, 1.5, 3.0]")
                params['sigma_levels'] = [0.7, 1.5, 3.0]
                
        # Validate feature_threshold (positive)
        if params.get('feature_threshold', 0) < 0:
            logger.warning("Invalid feature_threshold value, defaulting to 0.1")
            params['feature_threshold'] = 0.1
            
        # Validate feature_colors if present
        if params.get('feature_colors') is not None:
            try:
                ridge_color, valley_color = params['feature_colors']
                valid_color = lambda c: len(c) == 3 and all(0 <= v <= 1 for v in c)
                
                if not (valid_color(ridge_color) and valid_color(valley_color)):
                    logger.warning("Invalid feature_colors format, using defaults")
                    params['feature_colors'] = ((0.0, 0.7, 1.0), (1.0, 0.4, 0.0))
            except (ValueError, TypeError):
                logger.warning("Invalid feature_colors, using defaults")
                params['feature_colors'] = ((0.0, 0.7, 1.0), (1.0, 0.4, 0.0))
                
        # Validate percentile_clip (0-100 range)
        if params.get('percentile_clip') is not None:
            try:
                min_pct, max_pct = params['percentile_clip']
                if not (0 <= min_pct < max_pct <= 100):
                    logger.warning("Invalid percentile_clip range, defaulting to (2, 98)")
                    params['percentile_clip'] = (2, 98)
            except (ValueError, TypeError):
                logger.warning("Invalid percentile_clip format, defaulting to (2, 98)")
                params['percentile_clip'] = (2, 98)
                
        # Validate enhance_contrast (positive)
        if params.get('enhance_contrast', 0) <= 0:
            logger.warning("Invalid enhance_contrast value, defaulting to 1.0")
            params['enhance_contrast'] = 1.0
            
        # Validate edge_width (positive integer)
        if not isinstance(params.get('edge_width', 0), int) or params.get('edge_width', 0) <= 0:
            logger.warning("Invalid edge_width value, defaulting to 1")
            params['edge_width'] = 1
        elif params.get('edge_width', 0) > 5:
            logger.warning("edge_width too large, capping at 5")
            params['edge_width'] = 5
            
        return params