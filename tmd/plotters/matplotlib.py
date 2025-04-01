#!/usr/bin/env python3
"""
Matplotlib Plotters for TMD Files

This module defines two classes:
  - MatplotlibHeightMapPlotter: Provides methods for plotting single TMD height maps.
  - MatplotlibSequencePlotter: Provides methods for plotting TMD sequences (including animations, 2D/3D visualizations, and statistics).

Both classes implement the BasePlotter and BaseSequencePlotter interfaces from the
base module and use TMDFileUtilities for dependency management.
"""

import os
import warnings
import logging
from typing import Any, Dict, List, Optional, Tuple, Union, ClassVar

import numpy as np

# Import base classes and utilities
from tmd.plotters.base import BasePlotter, BaseSequencePlotter
from tmd.utils.files import TMDFileUtilities

# Set up logger
logger = logging.getLogger(__name__)

# Global constant for colorbar label
COLORBAR_LABEL = "Height (µm)"


class MatplotlibHeightMapPlotter(BasePlotter):
    """
    Matplotlib implementation of the TMD Plotter for single height maps.
    
    Provides methods for 3D surface plots, 2D heatmaps, contour plots, and profile visualizations.
    """
    
    NAME = "matplotlib"
    DEFAULT_COLORMAP = "viridis"
    SUPPORTED_MODES = ["2d", "3d", "contour", "profile"]
    REQUIRED_DEPENDENCIES = ["matplotlib.pyplot", "mpl_toolkits.mplot3d"]
    
    def __init__(self) -> None:
        """Initialize the Matplotlib plotter and check for dependencies."""
        super().__init__()
        # Lazy load matplotlib modules
        try:
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            self.plt = plt
            self.cm = cm
        except ImportError:
            raise ImportError("matplotlib.pyplot is required for MatplotlibHeightMapPlotter")
        
        # Check for 3D plotting capability
        try:
            from mpl_toolkits.mplot3d import Axes3D
            self.has_3d = True
        except ImportError:
            self.has_3d = False
            logger.warning("3D plotting not available - 3D plots will fall back to 2D contour plots")
    
    def plot(self, height_map: np.ndarray, **kwargs) -> Any:
        """
        Plot the TMD height map using Matplotlib.
        
        Args:
            height_map: 2D numpy array representing the height map.
            **kwargs: Additional options such as:
                - mode: Plot mode - "2d", "3d", "contour", "profile" (default: "2d")
                - colormap: Colormap name (default: "viridis")
                - figsize: Figure size (width, height) tuple in inches (default: (10, 8))
                - title: Plot title (default: "TMD Height Map")
                - colorbar_label: Label for the colorbar (default: "Height (µm)")
                - z_scale: Scaling factor for Z-axis in 3D plots (default: 1.0)
                - profile_row: Row index for profile plot (default: height_map.shape[0] // 2)
                - partial_range: Tuple (row_start, row_end, col_start, col_end) for plotting subset
                
        Returns:
            Matplotlib Figure object.
        """
        # Extract parameters with defaults
        mode = kwargs.get("mode", "2d").lower()
        figsize = kwargs.get("figsize", (10, 8))
        title = kwargs.get("title", "TMD Height Map")
        colorbar_label = kwargs.get("colorbar_label", COLORBAR_LABEL)
        cmap = kwargs.get("colormap", self.DEFAULT_COLORMAP)
        
        # Create figure
        fig = self.plt.figure(figsize=figsize)
        
        # Apply partial range if specified
        partial_range = kwargs.get("partial_range", None)
        if partial_range is not None:
            height_map = height_map[partial_range[0]:partial_range[1], partial_range[2]:partial_range[3]]
            logger.info(f"Partial render applied: rows {partial_range[0]}:{partial_range[1]}, "
                       f"cols {partial_range[2]}:{partial_range[3]}")
        
        # Create a copy of kwargs without the parameters we're explicitly passing
        # to avoid duplicate argument errors
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in 
                          ["title", "colorbar_label", "colormap", "cmap", "figsize"]}
        
        # Dispatch to appropriate plotting method based on mode
        if mode == "3d":
            # Make sure 'mode' is filtered out to avoid passing it to plot_surface
            if "mode" in filtered_kwargs:
                del filtered_kwargs["mode"]
                
            fig, ax = self._plot_3d_surface(height_map, fig=fig, cmap=cmap, 
                                           colorbar_label=colorbar_label, title=title, **filtered_kwargs)
        elif mode == "contour":
            fig, ax = self._plot_contour(height_map, fig=fig, cmap=cmap,
                                        colorbar_label=colorbar_label, title=title, **filtered_kwargs)
        elif mode == "profile":
            # Extract profile_row and remove it from kwargs to avoid duplicate parameter
            profile_row = kwargs.get("profile_row", height_map.shape[0] // 2)
            if "profile_row" in filtered_kwargs:
                del filtered_kwargs["profile_row"]
                
            fig, ax = self._plot_profile(height_map, profile_row, fig=fig, 
                                        colorbar_label=colorbar_label, title=title, **filtered_kwargs)
        else:  # Default to 2D
            fig, ax = self._plot_2d_heatmap(height_map, fig=fig, cmap=cmap,
                                           colorbar_label=colorbar_label, title=title, **filtered_kwargs)
        
        # Adjust layout and return figure
        self.plt.tight_layout()
        return fig
    
    def _plot_3d_surface(self, height_map: np.ndarray, fig: Any = None, ax: Any = None,
                         cmap: str = "viridis", z_scale: float = 1.0, 
                         colorbar_label: str = COLORBAR_LABEL,
                         title: str = "3D Surface Plot", **kwargs) -> Tuple[Any, Any]:
        """
        Create a 3D surface plot of a height map. Falls back to contour plot if 3D is unavailable.
        
        Args:
            height_map: 2D numpy array with height data.
            fig: Existing Figure (optional).
            ax: Existing Axes (optional).
            cmap: Colormap name.
            z_scale: Scaling factor for z-axis.
            colorbar_label: Label for colorbar.
            title: Plot title.
            **kwargs: Additional options.
            
        Returns:
            Tuple of (figure, axes).
        """
        if fig is None:
            fig = self.plt.figure(figsize=kwargs.get("figsize", (10, 8)))
            
        if not self.has_3d:
            warnings.warn("3D plotting not available - falling back to 2D contour plot")
            return self._plot_contour(height_map, fig=fig, cmap=cmap, 
                                    colorbar_label=colorbar_label, 
                                    title=f"{title} (2D Fallback)", **kwargs)
        
        if ax is None:
            # Import will succeed because we checked in __init__
            from mpl_toolkits.mplot3d import Axes3D
            ax = fig.add_subplot(111, projection="3d")
            
        # Create coordinate grid
        rows, cols = height_map.shape
        x = np.arange(cols)
        y = np.arange(rows)
        x, y = np.meshgrid(x, y)
        z = height_map * z_scale
        
        # Filter out parameters that should not be passed to plot_surface
        excluded_params = [
            "figsize", "colorbar_label", "title", "cmap", "z_scale", 
            "mode", "colormap", "profile_row", "partial_range", "interpolation",
            "show_markers", "marker_spacing", "marker_style", "show_grid", 
            "clean_display", "x_label", "y_label"
        ]
        
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in excluded_params}
        
        # Create surface plot with filtered kwargs
        surf = ax.plot_surface(
            x, y, z, cmap=cmap, linewidth=0, antialiased=True, **filtered_kwargs
        )
        
        # Add colorbar and labels
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        cbar.set_label(colorbar_label)
        ax.set_title(title)
        ax.set_xlabel("X Position (pixels)")
        ax.set_ylabel("Y Position (pixels)")
        ax.set_zlabel(colorbar_label)
        
        return fig, ax
    
    def _plot_2d_heatmap(self, height_map: np.ndarray, fig: Any = None, ax: Any = None,
                        cmap: str = "viridis", colorbar_label: str = COLORBAR_LABEL,
                        title: str = "2D Height Map", **kwargs) -> Tuple[Any, Any]:
        """
        Create a 2D heatmap of the height map.
        
        Args:
            height_map: 2D numpy array with height data.
            fig: Existing Figure (optional).
            ax: Existing Axes (optional).
            cmap: Colormap name.
            colorbar_label: Label for colorbar.
            title: Plot title.
            **kwargs: Additional options.
            
        Returns:
            Tuple of (figure, axes).
        """
        if fig is None:
            fig = self.plt.figure(figsize=kwargs.get("figsize", (10, 8)))
        if ax is None:
            ax = fig.add_subplot(111)
            
        # Create heatmap
        im = ax.imshow(height_map, cmap=cmap, origin="lower", 
                       interpolation=kwargs.get("interpolation", "nearest"))
        
        # Add colorbar and labels
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(colorbar_label)
        ax.set_title(title)
        ax.set_xlabel("X Position (pixels)")
        ax.set_ylabel("Y Position (pixels)")
        
        return fig, ax
    
    def _plot_contour(self, height_map: np.ndarray, fig: Any = None, ax: Any = None,
                     cmap: str = "viridis", colorbar_label: str = COLORBAR_LABEL,
                     title: str = "Contour Plot", **kwargs) -> Tuple[Any, Any]:
        """
        Create a contour plot of the height map.
        
        Args:
            height_map: 2D numpy array with height data.
            fig: Existing Figure (optional).
            ax: Existing Axes (optional).
            cmap: Colormap name.
            colorbar_label: Label for colorbar.
            title: Plot title.
            **kwargs: Additional options.
            
        Returns:
            Tuple of (figure, axes).
        """
        if fig is None:
            fig = self.plt.figure(figsize=kwargs.get("figsize", (10, 8)))
        if ax is None:
            ax = fig.add_subplot(111)
            
        # Create contour plot
        levels = kwargs.get("levels", 20)
        x = np.arange(height_map.shape[1])
        y = np.arange(height_map.shape[0])
        contour = ax.contourf(x, y, height_map, cmap=cmap, levels=levels)
        
        # Add contour lines if requested
        if kwargs.get("show_lines", True):
            line_levels = kwargs.get("line_levels", levels // 2)
            ax.contour(x, y, height_map, colors='k', linewidths=0.5, 
                      levels=line_levels, alpha=0.7)
        
        # Add colorbar and labels
        cbar = fig.colorbar(contour, ax=ax)
        cbar.set_label(colorbar_label)
        ax.set_title(title)
        ax.set_xlabel("X Position (pixels)")
        ax.set_ylabel("Y Position (pixels)")
        
        return fig, ax
    
    def _plot_profile(self, height_map: np.ndarray, profile_row: int, 
                     fig: Any = None, ax: Any = None, x_length: float = None,
                     colorbar_label: str = COLORBAR_LABEL,
                     title: str = None, **kwargs) -> Tuple[Any, Any]:
        """
        Create a profile plot along a specified row of the height map.
        
        Args:
            height_map: 2D numpy array with height data.
            profile_row: Row index for the profile.
            fig: Existing Figure (optional).
            ax: Existing Axes (optional).
            x_length: Physical length in x direction (optional).
            colorbar_label: Y-axis label.
            title: Plot title.
            **kwargs: Additional options.
            
        Returns:
            Tuple of (figure, axes).
        """
        # Filter out any profile_row from kwargs to avoid conflicts
        if "profile_row" in kwargs:
            del kwargs["profile_row"]
            
        if fig is None:
            fig = self.plt.figure(figsize=kwargs.get("figsize", (10, 6)))
        if ax is None:
            ax = fig.add_subplot(111)
            
        # Check profile row is valid
        if profile_row < 0 or profile_row >= height_map.shape[0]:
            profile_row = height_map.shape[0] // 2
            logger.warning(f"Invalid profile row. Using middle row: {profile_row}")
            
        # Create x coordinates (physical or pixel)
        width = height_map.shape[1]
        if x_length is not None:
            x_offset = kwargs.get("x_offset", 0)
            x_coords = np.linspace(x_offset, x_offset + x_length, num=width)
            x_label = "X Position (mm)"
        else:
            x_coords = np.arange(width)
            x_label = "X Position (pixels)"
            
        # Get the profile data
        y_profile = height_map[profile_row, :]
        
        # Generate plot title if not provided
        if title is None:
            title = f"Height Profile at Row {profile_row}"
            
        # Create the plot
        line_style = kwargs.get("line_style", {})
        ax.plot(x_coords, y_profile, linewidth=1, **line_style)
        
        # Add markers if requested
        if kwargs.get("show_markers", True):
            marker_spacing = kwargs.get("marker_spacing", max(1, width // 30))
            marker_style = kwargs.get("marker_style", {"color": "red", "s": 20})
            ax.scatter(x_coords[::marker_spacing], y_profile[::marker_spacing], **marker_style)
            
        # Add axis labels and title
        ax.set_xlabel(x_label)
        ax.set_ylabel(colorbar_label)
        ax.set_title(title)
        
        # Add grid
        if kwargs.get("show_grid", True):
            ax.grid(True, linestyle="--", alpha=0.7)
            
        return fig, ax
        
    def save(self, plot_obj: Any, filename: Union[str, Any], **kwargs) -> Optional[str]:
        """
        Save the plot to a file.
        
        Args:
            plot_obj: Matplotlib Figure object.
            filename: Output filename or path.
            **kwargs: Additional options such as:
                - dpi: Resolution in dots per inch (default: 300)
                - bbox_inches: Bounding box option (default: 'tight')
                - show_axes: Whether to show axes in the saved image (default: False)
                - transparent: Whether to save with transparent background (default: False)
                
        Returns:
            Filename if saved successfully, None otherwise.
        """
        try:
            filename = str(filename)
            directory = os.path.dirname(os.path.abspath(filename))
            os.makedirs(directory, exist_ok=True)
            
            dpi = kwargs.get("dpi", 300)
            bbox_inches = kwargs.get("bbox_inches", "tight")
            show_axes = kwargs.get("show_axes", False)
            transparent = kwargs.get("transparent", False)
            
            # Get the figure from the plot object
            if hasattr(plot_obj, "savefig"):  # It's a figure
                fig = plot_obj
            elif hasattr(plot_obj, "figure"):  # It's an axes
                fig = plot_obj.figure
            else:
                logger.warning(f"Unknown plot object type: {type(plot_obj)}")
                return None
            
            # Hide axes if requested
            if not show_axes:
                # Find all axes in the figure
                for ax in fig.get_axes():
                    ax.set_axis_off()
            
            # Save the figure
            fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches, transparent=transparent)
            logger.info(f"Plot saved to {filename}")
            return filename
        except Exception as e:
            logger.error(f"Error saving plot: {e}")
            return None


class MatplotlibSequencePlotter(BaseSequencePlotter):
    """
    Matplotlib implementation for TMD sequence plotting.
    
    Provides methods for creating animations, visualizing sequences in 2D/3D,
    and plotting statistics from TMD sequences.
    """
    
    NAME = "matplotlib"
    DEFAULT_COLORMAP = "viridis"
    SUPPORTED_MODES = ["2d", "3d", "animation", "statistics"]
    REQUIRED_DEPENDENCIES = ["matplotlib.pyplot", "matplotlib.animation"]
    
    def __init__(self) -> None:
        """Initialize the Matplotlib sequence plotter and verify dependencies."""
        super().__init__()
        
        # Get matplotlib modules
        try:
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            self.plt = plt
            self.cm = cm
        except ImportError:
            raise ImportError("matplotlib.pyplot is required for MatplotlibSequencePlotter")
            
        # Check for animation support
        try:
            import matplotlib.animation
            self.animation = matplotlib.animation
        except ImportError:
            self.animation = None
            logger.warning("matplotlib.animation not available - animation features will be disabled")
            
        # Check for 3D support
        try:
            from mpl_toolkits.mplot3d import Axes3D
            self.has_3d = True
        except ImportError:
            self.has_3d = False
            logger.warning("3D plotting not available - 3D sequence plots will fall back to 2D")

    def create_animation(self, frames_data: List[np.ndarray], **kwargs) -> Any:
        """
        Create a Matplotlib animation from sequence data.
        
        Args:
            frames_data: List of 2D numpy arrays representing the sequence.
            **kwargs: Additional options such as:
                - fps: Frames per second (default: 10)
                - colormap: Colormap name (default: 'viridis')
                - figsize: Figure size (width, height) in inches (default: (10, 8))
                - title: Animation title (default: 'TMD Sequence Animation')
                - colorbar_label: Label for the colorbar (default: 'Height')
                - interval: Delay between frames in milliseconds (default: calculated from fps)
                
        Returns:
            Matplotlib animation object if animation module is available, otherwise a figure.
        """
        if not frames_data:
            logger.error("No frame data provided for animation")
            fig, ax = self.plt.subplots()
            ax.text(0.5, 0.5, "No frames to animate",
                    horizontalalignment='center', verticalalignment='center')
            return fig
            
        if self.animation is None:
            logger.error("Animation module not available")
            # Fall back to showing first frame
            fig, ax = self.plt.subplots()
            ax.imshow(frames_data[0], cmap=kwargs.get('colormap', 'viridis'))
            ax.set_title(f"{kwargs.get('title', 'TMD Sequence')} (First Frame Only)")
            return fig

        # Extract parameters with defaults
        fps = kwargs.get('fps', 10)
        colormap = kwargs.get('colormap', self.DEFAULT_COLORMAP)
        figsize = kwargs.get('figsize', (10, 8))
        title = kwargs.get('title', 'TMD Sequence Animation')
        colorbar_label = kwargs.get('colorbar_label', 'Height')
        interval = kwargs.get('interval', 1000 / fps)
        
        # Create figure and plot first frame
        fig, ax = self.plt.subplots(figsize=figsize)
        ax.set_title(title)
        im = ax.imshow(frames_data[0], cmap=colormap, animated=True)
        cbar = self.plt.colorbar(im, ax=ax, label=colorbar_label)
        
        # Animation update function
        def update_frame(i):
            im.set_array(frames_data[i])
            return [im]

        try:
            # Create animation
            anim = self.animation.FuncAnimation(
                fig, update_frame, frames=len(frames_data),
                interval=interval, blit=True
            )
            return anim
        except Exception as e:
            logger.error(f"Error creating animation: {e}")
            return fig

    def visualize_sequence(self, frames_data: List[np.ndarray], **kwargs) -> Any:
        """
        Visualize a sequence of frames side by side.
        
        Args:
            frames_data: List of 2D numpy arrays representing the sequence.
            **kwargs: Options such as:
                - n_frames: Number of frames to display (default: min(len(frames_data), 5))
                - mode: Visualization mode, either '2d' or '3d' (default: '2d')
                - colormap: Colormap name (default: 'viridis')
                - figsize: Figure size in inches (default: (15, 8))
                - title: Visualization title (default: 'TMD Sequence Visualization')
                - layout: Layout of frames, 'grid', 'row', or 'column' (default: 'row')
                - colorbar_label: Label for the colorbar (default: 'Height')
                - frame_indices: Specific frame indices to include (default: None)
                
        Returns:
            Matplotlib Figure with the sequence visualization.
        """
        if not frames_data:
            logger.error("No frame data provided for visualization")
            fig = self.plt.figure()
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No frames to visualize",
                    horizontalalignment='center', verticalalignment='center')
            return fig

        # Extract parameters with defaults
        n_frames = kwargs.get('n_frames', min(len(frames_data), 5))
        mode = kwargs.get('mode', '2d').lower()
        colormap = kwargs.get('colormap', self.DEFAULT_COLORMAP)
        figsize = kwargs.get('figsize', (15, 8))
        title = kwargs.get('title', 'TMD Sequence Visualization')
        layout = kwargs.get('layout', 'row').lower()
        colorbar_label = kwargs.get('colorbar_label', 'Height')
        
        # Check if specific indices are provided
        frame_indices = kwargs.get('frame_indices', None)
        if frame_indices is not None:
            # Validate indices
            valid_indices = [i for i in frame_indices if 0 <= i < len(frames_data)]
            if not valid_indices:
                logger.warning("No valid frame indices provided, using default selection")
                frame_indices = None
                
        # Select frames to display
        if frame_indices is not None:
            indices = [i for i in frame_indices if 0 <= i < len(frames_data)]
            selected_frames = [frames_data[i] for i in indices]
        elif len(frames_data) > n_frames:
            # Sample frames evenly
            indices = np.linspace(0, len(frames_data) - 1, n_frames, dtype=int)
            selected_frames = [frames_data[i] for i in indices]
        else:
            indices = list(range(len(frames_data)))
            selected_frames = frames_data
            
        # Create figure with appropriate layout
        if layout == 'grid':
            # Calculate grid dimensions
            n_cols = int(np.ceil(np.sqrt(len(selected_frames))))
            n_rows = int(np.ceil(len(selected_frames) / n_cols))
        elif layout == 'column':
            n_rows = len(selected_frames)
            n_cols = 1
        else:  # default to row
            n_rows = 1
            n_cols = len(selected_frames)
            
        # Check if 3D mode is requested but not available
        if mode == '3d' and not self.has_3d:
            logger.warning("3D plotting not available - falling back to 2D")
            mode = '2d'
            
        # Create the visualization
        if mode == '3d':
            from mpl_toolkits.mplot3d import Axes3D
            fig = self.plt.figure(figsize=figsize)
            fig.suptitle(title)
            
            for i, (frame, idx) in enumerate(zip(selected_frames, indices)):
                # Create subplot position
                ax = fig.add_subplot(n_rows, n_cols, i + 1, projection='3d')
                
                # Create mesh grid
                height, width = frame.shape
                y, x = np.mgrid[0:height, 0:width]
                
                # Create surface plot
                surf = ax.plot_surface(x, y, frame, cmap=colormap, 
                                      linewidth=0, antialiased=True)
                                      
                # Set title and labels
                ax.set_title(f"Frame {idx}")
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel(colorbar_label)
                
                # Add colorbar
                if kwargs.get('show_colorbar', True):
                    fig.colorbar(surf, ax=ax, shrink=0.6, label=colorbar_label)
        else:
            # 2D visualization
            fig, axes = self.plt.subplots(n_rows, n_cols, figsize=figsize)
            fig.suptitle(title)
            
            # Handle single subplot case
            if n_rows * n_cols == 1:
                axes = np.array([axes])
            
            # Flatten axes for easy iteration
            axes = np.array(axes).flatten()
            
            for i, (frame, idx) in enumerate(zip(selected_frames, indices)):
                ax = axes[i]
                im = ax.imshow(frame, cmap=colormap)
                ax.set_title(f"Frame {idx}")
                
                # Add colorbar if requested
                if kwargs.get('show_colorbar', True):
                    fig.colorbar(im, ax=ax, label=colorbar_label)
                    
                # Optionally remove ticks for cleaner display
                if kwargs.get('clean_display', True):
                    ax.set_xticks([])
                    ax.set_yticks([])
                else:
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
            
            # Hide unused subplots
            for j in range(len(selected_frames), len(axes)):
                axes[j].axis('off')
                
        # Adjust layout
        self.plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
        return fig

    def visualize_statistics(self, stats_data: Dict[str, List[float]], **kwargs) -> Any:
        """
        Visualize statistical data from the sequence.
        
        Args:
            stats_data: Dictionary with metric names as keys and lists of values.
            **kwargs: Additional options such as:
                - figsize: Figure size (width, height) in inches (default: (12, 8))
                - title: Plot title (default: 'TMD Sequence Statistics')
                - x_label: X-axis label (default: 'Frame')
                - y_label: Y-axis label (default: 'Value')
                - metrics: List of metrics to plot (default: all metrics except 'timestamps')
                - style: Style of the plot ('line', 'bar', etc.) (default: 'line')
                - legend_loc: Location for the legend (default: 'best')
                - marker: Marker style for line plots (default: 'o')
                
        Returns:
            Matplotlib Figure with statistical visualization.
        """
        if not stats_data:
            logger.error("No statistical data provided for visualization")
            fig = self.plt.figure()
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No statistics data to visualize",
                    horizontalalignment='center', verticalalignment='center')
            return fig

        # Extract parameters with defaults
        figsize = kwargs.get('figsize', (12, 8))
        title = kwargs.get('title', 'TMD Sequence Statistics')
        x_label = kwargs.get('x_label', 'Frame')
        y_label = kwargs.get('y_label', 'Value')
        style = kwargs.get('style', 'line').lower()
        legend_loc = kwargs.get('legend_loc', 'best')
        
        # Identify metrics to plot
        available_metrics = [m for m in stats_data.keys() if m != 'timestamps']
        metrics = kwargs.get('metrics', available_metrics)
        
        # Validate metrics
        valid_metrics = [m for m in metrics if m in stats_data]
        if not valid_metrics:
            logger.error("No valid metrics found in the data")
            fig = self.plt.figure()
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No valid metrics to visualize",
                    horizontalalignment='center', verticalalignment='center')
            return fig
            
        # Get x-axis values
        x_values = stats_data.get('timestamps', list(range(max(len(stats_data[m]) for m in valid_metrics))))
        
        # Create figure
        fig, ax = self.plt.subplots(figsize=figsize)
        
        # Plot each metric
        for metric in valid_metrics:
            values = stats_data[metric]
            
            # Truncate x_values to match length of values
            plot_x = x_values[:len(values)]
            
            if style == 'bar':
                # For bar charts, create grouped bars
                ax.bar(plot_x, values, label=metric, alpha=0.7)
            else:
                # Default to line plot
                marker = kwargs.get('marker', 'o')
                linewidth = kwargs.get('linewidth', 1.5)
                markersize = kwargs.get('markersize', 5)
                ax.plot(plot_x, values, label=metric, marker=marker, 
                       linewidth=linewidth, markersize=markersize)
        
        # Add labels and title
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        
        # Add grid
        if kwargs.get('show_grid', True):
            ax.grid(True, linestyle='--', alpha=0.7)
            
        # Add legend
        if len(valid_metrics) > 1:
            ax.legend(loc=legend_loc)
            
        # Adjust layout
        self.plt.tight_layout()
        return fig
    
    def save_figure(self, fig: Any, filename: Union[str, Any], **kwargs) -> Optional[str]:
        """
        Save a matplotlib figure to a file.
        
        Args:
            fig: Matplotlib figure or animation object.
            filename: Output filename.
            **kwargs: Additional options such as:
                - dpi: Dots per inch (default: 300)
                - writer: Animation writer (default: 'pillow')
                - fps: Frames per second for animation (default: 10)
                
        Returns:
            Filename if saved successfully, None otherwise.
        """
        try:
            filename = str(filename)
            directory = os.path.dirname(os.path.abspath(filename))
            os.makedirs(directory, exist_ok=True)
            dpi = kwargs.get('dpi', 300)
            writer = kwargs.get('writer', 'pillow')
            fps = kwargs.get('fps', 10)
            bbox_inches = kwargs.get('bbox_inches', 'tight')
            # Check if it's an animation

            if isinstance(fig, self.animation.FuncAnimation):
                fig.save(filename, writer=writer, fps=fps, dpi=dpi, bbox_inches=bbox_inches)
            else:
                fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
            logger.info(f"Figure saved to {filename}")
            return filename
        except Exception as e:
            logger.error(f"Error saving figure: {e}")
            return None
        finally:
            # Close the figure to free up memory
            self.plt.close(fig)
