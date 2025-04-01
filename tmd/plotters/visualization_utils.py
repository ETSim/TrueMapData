#!/usr/bin/env python3
"""
TMD Enhanced Visualization Utilities

This module provides advanced visualization features that can be used
with multiple plotting backends (matplotlib, plotly, etc). 

Classes:
  - TMDVisualizationUtils: Utility functions for enhanced visualizations
  - ColorMapRegistry: Registry for custom colormaps and color utilities
  - HeightMapAnalyzer: Analysis tools for height maps
"""

import numpy as np
import logging
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import functools
import colorsys

# Set up logging
logger = logging.getLogger(__name__)

class ColorMapRegistry:
    """Registry for custom colormaps and color utilities for TMD visualization."""
    
    # Standard TMD colormaps for consistent visualization
    TMD_COLORMAPS = {
        "tmd_height": ["#000033", "#0000FF", "#00FFFF", "#FFFF00", "#FF0000", "#FFFFFF"],
        "tmd_terrain": ["#00441b", "#1b7837", "#5aae61", "#a6dba0", "#d9f0d3", 
                        "#e7d4e8", "#c2a5cf", "#9970ab", "#762a83", "#40004b"],
        "tmd_thermal": ["#000000", "#FF0000", "#FFFF00", "#FFFFFF"],
        "tmd_diverging": ["#2166ac", "#4393c3", "#92c5de", "#d1e5f0", "#f7f7f7", 
                          "#fddbc7", "#f4a582", "#d6604d", "#b2182b"]
    }
    
    @classmethod
    def get_cmap_list(cls, cmap_name: str) -> List[str]:
        """Get the color list for a registered colormap."""
        return cls.TMD_COLORMAPS.get(cmap_name, [])
    
    @classmethod
    def register_cmap(cls, name: str, colors: List[str]) -> None:
        """Register a new colormap."""
        cls.TMD_COLORMAPS[name] = colors
        logger.info(f"Registered new colormap: {name}")
    
    @classmethod
    def get_available_cmaps(cls) -> List[str]:
        """Get list of available custom colormaps."""
        return list(cls.TMD_COLORMAPS.keys())
    
    @staticmethod
    def create_matplotlib_cmap(cmap_name: str, n_colors: int = 256) -> Any:
        """Create a matplotlib colormap from a registered colormap name."""
        try:
            import matplotlib.colors as mcolors
            from matplotlib.colors import LinearSegmentedColormap
        except ImportError:
            logger.error("matplotlib is required for this functionality")
            return None
            
        if cmap_name in ColorMapRegistry.TMD_COLORMAPS:
            colors = ColorMapRegistry.TMD_COLORMAPS[cmap_name]
            return LinearSegmentedColormap.from_list(cmap_name, colors, N=n_colors)
        else:
            logger.warning(f"Unknown colormap: {cmap_name}. Using viridis.")
            return None
    
    @staticmethod
    def create_plotly_cmap(cmap_name: str) -> List[List[Union[float, str]]]:
        """Create a plotly colormap from a registered colormap name."""
        if cmap_name in ColorMapRegistry.TMD_COLORMAPS:
            colors = ColorMapRegistry.TMD_COLORMAPS[cmap_name]
            n_colors = len(colors)
            return [[i/(n_colors-1), color] for i, color in enumerate(colors)]
        else:
            logger.warning(f"Unknown colormap: {cmap_name}. Using viridis.")
            return None
    
    @staticmethod
    def height_to_color(height: float, min_height: float, max_height: float, 
                        cmap_name: str = "tmd_height") -> str:
        """Convert a height value to a color using a colormap."""
        if cmap_name in ColorMapRegistry.TMD_COLORMAPS:
            colors = ColorMapRegistry.TMD_COLORMAPS[cmap_name]
            n_colors = len(colors)
            
            # Normalize height to [0, 1]
            norm_height = (height - min_height) / (max_height - min_height)
            norm_height = max(0, min(1, norm_height))
            
            # Convert to color index
            idx = norm_height * (n_colors - 1)
            idx_low = int(idx)
            idx_high = min(idx_low + 1, n_colors - 1)
            frac = idx - idx_low
            
            # Interpolate between colors
            c1 = colors[idx_low]
            c2 = colors[idx_high]
            
            # Convert hex to RGB
            r1, g1, b1 = int(c1[1:3], 16)/255, int(c1[3:5], 16)/255, int(c1[5:7], 16)/255
            r2, g2, b2 = int(c2[1:3], 16)/255, int(c2[3:5], 16)/255, int(c2[5:7], 16)/255
            
            # Interpolate
            r = r1 * (1 - frac) + r2 * frac
            g = g1 * (1 - frac) + g2 * frac
            b = b1 * (1 - frac) + b2 * frac
            
            # Convert back to hex
            return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
        else:
            logger.warning(f"Unknown colormap: {cmap_name}")
            return "#FFFFFF"  # Default to white


class HeightMapAnalyzer:
    """Analysis tools for TMD height maps."""
    
    @staticmethod
    def compute_basic_stats(height_map: np.ndarray) -> Dict[str, float]:
        """Compute basic statistics for a height map."""
        # Handle empty or invalid arrays
        if height_map is None or height_map.size == 0:
            return {
                "min": 0, "max": 0, "mean": 0, "median": 0, 
                "std": 0, "range": 0, "rms": 0
            }
            
        # Compute statistics
        height_min = np.min(height_map)
        height_max = np.max(height_map)
        height_mean = np.mean(height_map)
        height_median = np.median(height_map)
        height_std = np.std(height_map)
        height_range = height_max - height_min
        height_rms = np.sqrt(np.mean(np.square(height_map)))
        
        return {
            "min": height_min,
            "max": height_max,
            "mean": height_mean,
            "median": height_median,
            "std": height_std,
            "range": height_range,
            "rms": height_rms
        }
    
    @staticmethod
    def compute_advanced_stats(height_map: np.ndarray) -> Dict[str, float]:
        """Compute advanced statistics for a height map."""
        basic_stats = HeightMapAnalyzer.compute_basic_stats(height_map)
        
        # Add advanced statistics
        if height_map is not None and height_map.size > 0:
            # Skewness
            mean = basic_stats["mean"]
            std = basic_stats["std"]
            if std > 0:
                skewness = np.mean(((height_map - mean) / std) ** 3)
            else:
                skewness = 0
                
            # Kurtosis
            if std > 0:
                kurtosis = np.mean(((height_map - mean) / std) ** 4) - 3
            else:
                kurtosis = 0
                
            # Surface roughness - Ra (average roughness)
            # For a height map, we calculate deviation from the mean plane
            roughness_ra = np.mean(np.abs(height_map - mean))
            
            # Root mean square roughness - Rq
            roughness_rq = np.sqrt(np.mean(np.square(height_map - mean)))
            
            # Ten-point mean roughness
            flattened = height_map.flatten()
            sorted_heights = np.sort(flattened)
            n = len(sorted_heights)
            if n >= 10:
                five_highest = sorted_heights[-5:]
                five_lowest = sorted_heights[:5]
                roughness_rz = np.mean(five_highest) - np.mean(five_lowest)
            else:
                roughness_rz = basic_stats["range"]
            
            advanced_stats = {
                "skewness": skewness,
                "kurtosis": kurtosis,
                "roughness_ra": roughness_ra,
                "roughness_rq": roughness_rq,
                "roughness_rz": roughness_rz
            }
        else:
            advanced_stats = {
                "skewness": 0, "kurtosis": 0, 
                "roughness_ra": 0, "roughness_rq": 0, "roughness_rz": 0
            }
            
        return {**basic_stats, **advanced_stats}
    
    @staticmethod
    def detect_features(height_map: np.ndarray, threshold: float = 0.5, 
                       min_size: int = 10) -> Tuple[np.ndarray, int]:
        """
        Detect features in a height map based on threshold.
        
        Args:
            height_map: 2D numpy array with height data
            threshold: Height threshold as a fraction of range [0.0-1.0]
            min_size: Minimum feature size in pixels
            
        Returns:
            Tuple of (binary mask of features, number of features)
        """
        try:
            from scipy import ndimage
        except ImportError:
            logger.error("scipy is required for feature detection")
            return None, 0
            
        # Normalize height map to [0, 1]
        height_min = np.min(height_map)
        height_max = np.max(height_map)
        height_range = height_max - height_min
        
        if height_range > 0:
            normalized = (height_map - height_min) / height_range
        else:
            return np.zeros_like(height_map, dtype=bool), 0
            
        # Create binary mask based on threshold
        binary = normalized > threshold
        
        # Label connected components
        labeled, num_features = ndimage.label(binary)
        
        # Filter small features
        if min_size > 1:
            for i in range(1, num_features + 1):
                if np.sum(labeled == i) < min_size:
                    binary[labeled == i] = False
                    
            # Re-label after filtering
            labeled, num_features = ndimage.label(binary)
            
        return binary, num_features
    
    @staticmethod
    def compute_profiles(height_map: np.ndarray,
                        x_pos: Optional[int] = None,
                        y_pos: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Compute horizontal and vertical profiles at specified positions.
        
        Args:
            height_map: 2D numpy array with height data
            x_pos: X position for vertical profile (default: middle)
            y_pos: Y position for horizontal profile (default: middle)
            
        Returns:
            Dictionary with horizontal and vertical profiles
        """
        if height_map is None or height_map.size == 0:
            return {"horizontal": np.array([]), "vertical": np.array([])}
            
        h, w = height_map.shape
        
        # Default to middle if not specified
        if x_pos is None:
            x_pos = w // 2
        if y_pos is None:
            y_pos = h // 2
            
        # Ensure positions are within bounds
        x_pos = max(0, min(w - 1, x_pos))
        y_pos = max(0, min(h - 1, y_pos))
        
        # Extract profiles
        horizontal = height_map[y_pos, :]
        vertical = height_map[:, x_pos]
        
        return {
            "horizontal": horizontal,
            "vertical": vertical,
            "x_pos": x_pos,
            "y_pos": y_pos
        }
    
    @staticmethod
    def compute_gradient(height_map: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute gradient (slope) of the height map.
        
        Args:
            height_map: 2D numpy array with height data
            
        Returns:
            Dictionary with gradient magnitude and direction
        """
        try:
            from scipy import ndimage
        except ImportError:
            logger.error("scipy is required for gradient computation")
            return {
                "magnitude": np.zeros_like(height_map),
                "direction": np.zeros_like(height_map)
            }
            
        # Compute gradients using Sobel operator
        grad_y = ndimage.sobel(height_map, axis=0)
        grad_x = ndimage.sobel(height_map, axis=1)
        
        # Compute magnitude and direction
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)
        
        return {
            "magnitude": magnitude,
            "direction": direction,
            "grad_x": grad_x,
            "grad_y": grad_y
        }


class TMDVisualizationUtils:
    """
    Utility functions for enhanced visualizations that can be used with
    multiple plotting backends.
    """
    
    @staticmethod
    def create_overlay_plot(plotter: Any, height_map: np.ndarray, 
                          overlay_data: np.ndarray, **kwargs) -> Any:
        """
        Create a plot with an overlay (e.g. features, gradient).
        
        Args:
            plotter: Plotter instance to use (matplotlib, plotly, etc.)
            height_map: Base height map
            overlay_data: Data to overlay (same shape as height_map)
            **kwargs: Additional options for the plotter
            
        Returns:
            Plot object from the plotter
        """
        # Check plotter type
        plotter_type = type(plotter).__name__
        
        if "Matplotlib" in plotter_type:
            return TMDVisualizationUtils._create_matplotlib_overlay(
                plotter, height_map, overlay_data, **kwargs
            )
        elif "Plotly" in plotter_type:
            return TMDVisualizationUtils._create_plotly_overlay(
                plotter, height_map, overlay_data, **kwargs
            )
        else:
            logger.warning(f"Unsupported plotter type: {plotter_type}")
            # Fall back to regular plotting
            return plotter.plot(height_map, **kwargs)
    
    @staticmethod
    def _create_matplotlib_overlay(plotter, height_map, overlay_data, **kwargs):
        """Create overlay plot with matplotlib."""
        try:
            import matplotlib.pyplot as plt
            
            # Extract parameters
            figsize = kwargs.get("figsize", (12, 10))
            base_cmap = kwargs.get("cmap", "viridis")
            overlay_cmap = kwargs.get("overlay_cmap", "plasma")
            title = kwargs.get("title", "Height Map with Overlay")
            alpha = kwargs.get("alpha", 0.7)
            
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            
            # Plot base height map
            im1 = ax.imshow(height_map, cmap=base_cmap, interpolation='nearest')
            
            # Plot overlay with transparency
            im2 = ax.imshow(overlay_data, cmap=overlay_cmap, alpha=alpha)
            
            # Add colorbars
            cbar1 = fig.colorbar(im1, ax=ax, location='left', shrink=0.6)
            cbar1.set_label("Height")
            
            cbar2 = fig.colorbar(im2, ax=ax, location='right', shrink=0.6)
            cbar2.set_label("Overlay")
            
            # Set labels
            ax.set_title(title)
            ax.set_xlabel("X Position")
            ax.set_ylabel("Y Position")
            
            return fig
            
        except ImportError:
            logger.error("matplotlib is required for this functionality")
            return None
    
    @staticmethod
    def _create_plotly_overlay(plotter, height_map, overlay_data, **kwargs):
        """Create overlay plot with plotly."""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # Extract parameters
            width = kwargs.get("width", 800)
            height = kwargs.get("height", 600)
            base_cmap = kwargs.get("cmap", "Viridis")
            overlay_cmap = kwargs.get("overlay_cmap", "Plasma")
            title = kwargs.get("title", "Height Map with Overlay")
            
            # Create figure with two subplots side by side
            fig = make_subplots(
                rows=1, cols=2, 
                subplot_titles=["Base Height Map", "With Overlay"],
                horizontal_spacing=0.1
            )
            
            # Plot base height map
            fig.add_trace(
                go.Heatmap(z=height_map, colorscale=base_cmap, showscale=True),
                row=1, col=1
            )
            
            # Plot overlay
            fig.add_trace(
                go.Heatmap(z=overlay_data, colorscale=overlay_cmap, showscale=True),
                row=1, col=2
            )
            
            # Update layout
            fig.update_layout(
                title_text=title,
                width=width,
                height=height
            )
            
            return fig
            
        except ImportError:
            logger.error("plotly is required for this functionality")
            return None
    
    @staticmethod
    def create_multi_view_plot(plotter: Any, height_map: np.ndarray, **kwargs) -> Any:
        """
        Create a multi-view visualization with different perspectives of the same data.
        
        Args:
            plotter: Plotter instance to use
            height_map: Height map data to visualize
            **kwargs: Additional options for the plotter
            
        Returns:
            Plot object from the plotter
        """
        # Check plotter type to dispatch to appropriate method
        plotter_type = type(plotter).__name__
        
        if "Matplotlib" in plotter_type:
            try:
                import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d import Axes3D
                
                # Extract parameters
                figsize = kwargs.get("figsize", (15, 10))
                cmap = kwargs.get("cmap", "viridis")
                title = kwargs.get("title", "Multi-view Visualization")
                
                # Create a figure with three subplots: 2D view, 3D view, and profile
                fig = plt.figure(figsize=figsize)
                fig.suptitle(title, fontsize=16)
                
                # 2D view
                ax1 = fig.add_subplot(131)
                im = ax1.imshow(height_map, cmap=cmap)
                ax1.set_title("2D View")
                fig.colorbar(im, ax=ax1, shrink=0.6, label="Height")
                
                # 3D view
                ax2 = fig.add_subplot(132, projection='3d')
                rows, cols = height_map.shape
                x, y = np.meshgrid(np.arange(cols), np.arange(rows))
                z_scale = kwargs.get("z_scale", 1.0)
                surf = ax2.plot_surface(
                    x, y, height_map * z_scale, 
                    cmap=cmap, linewidth=0, antialiased=True
                )
                ax2.set_title("3D View")
                
                # Profile view
                ax3 = fig.add_subplot(133)
                profile_row = kwargs.get("profile_row", height_map.shape[0] // 2)
                profile_data = height_map[profile_row, :]
                ax3.plot(profile_data)
                ax3.set_title(f"Profile (Row {profile_row})")
                ax3.grid(True, linestyle='--', alpha=0.7)
                
                plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
                return fig
                
            except ImportError:
                logger.error("matplotlib and mpl_toolkits are required for multi-view plots")
                return None
                
        elif "Plotly" in plotter_type:
            try:
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                
                # Extract parameters
                width = kwargs.get("width", 1200)
                height = kwargs.get("height", 600)
                colorscale = kwargs.get("colorscale", "Viridis")
                title = kwargs.get("title", "Multi-view Visualization")
                
                # Create figure with three subplots
                fig = make_subplots(
                    rows=1, cols=3,
                    subplot_titles=["2D View", "3D View", "Profile"],
                    specs=[[{"type": "heatmap"}, {"type": "surface"}, {"type": "scatter"}]],
                    horizontal_spacing=0.05
                )
                
                # 2D view
                fig.add_trace(
                    go.Heatmap(z=height_map, colorscale=colorscale, showscale=True),
                    row=1, col=1
                )
                
                # 3D view
                rows, cols = height_map.shape
                z_scale = kwargs.get("z_scale", 1.0)
                x = np.arange(cols)
                y = np.arange(rows)
                fig.add_trace(
                    go.Surface(
                        z=height_map * z_scale, 
                        x=x, y=y,
                        colorscale=colorscale, 
                        showscale=False
                    ),
                    row=1, col=2
                )
                
                # Profile view
                profile_row = kwargs.get("profile_row", height_map.shape[0] // 2)
                profile_data = height_map[profile_row, :]
                fig.add_trace(
                    go.Scatter(y=profile_data, mode="lines"),
                    row=1, col=3
                )
                
                # Update layout
                fig.update_layout(
                    title_text=title,
                    width=width,
                    height=height
                )
                
                return fig
                
            except ImportError:
                logger.error("plotly is required for this functionality")
                return None
        
        else:
            logger.warning(f"Unsupported plotter type: {plotter_type}")
            return None
