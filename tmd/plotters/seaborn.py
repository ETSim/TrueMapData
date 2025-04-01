#!/usr/bin/env python3
"""
Seaborn-based visualization classes for TMD data.

This module provides three classes:
  - SeabornHeightMapPlotter: For creating heatmaps of height maps (implements BasePlotter).
  - SeabornProfilePlotter: For creating distribution plots, profile comparisons,
    and joint distribution plots based on height maps.
  - SeabornSequencePlotter: For visualizing sequences of height maps (implements BaseSequencePlotter).

All classes use Seaborn (and Matplotlib) for visualizations and rely on external
utility functions (e.g. for lazy imports and dependency checking).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import logging
from typing import Any, Dict, List, Optional, Tuple, Union, ClassVar
import functools

from tmd.utils.utils import TMDUtils
from tmd.utils.files import TMDFileUtilities
from tmd.plotters.base import BasePlotter, BaseSequencePlotter

# Set up logging
logger = logging.getLogger(__name__)

# Check Seaborn dependency
dependencies = ['seaborn']
HAS_SEABORN = TMDFileUtilities.import_optional_dependency('seaborn') is not None

# Lazy-import Seaborn
sns = TMDFileUtilities.import_optional_dependency('seaborn')

# Default settings
COLORBAR_LABEL = "Height (µm)"

# Helper decorator to check for seaborn
def requires_seaborn(func):
    """Decorator to check if seaborn is available."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not HAS_SEABORN:
            raise ImportError("The seaborn module is required for this functionality.")
        return func(*args, **kwargs)
    return wrapper

class SeabornHeightMapPlotter(BasePlotter):
    """Provides Seaborn-based visualizations for TMD height maps.
    Implements the BasePlotter interface.

    Methods:
      - plot: Plot a height map in the default style (implements BasePlotter interface).
      - plot_heatmap: Create a basic heatmap.
      - plot_enhanced_heatmap: Create a heatmap with contour lines and annotations.
      - plot_2d: Plot a 2D heatmap.
      - plot_3d: Falls back to Matplotlib for 3D visualization.
      - save: Save the plot to a file (implements BasePlotter interface).
    """
    def plot(self, height_map: np.ndarray, **kwargs) -> plt.Figure:
        """Plot the TMD height map using Seaborn.
        Implements the BasePlotter interface.

        Args:
            height_map: 2D numpy array representing the height map.
            **kwargs: Additional options such as:
                - title: Plot title (default: "Height Map (Seaborn)")
                - colorbar_label: Label for the color bar (default: "Height (µm)")
                - mode: "2d" for heatmap (default), "3d" falls back to Matplotlib
                - filename: If provided, save the plot to this file
                - cmap: Colormap to use (default: "viridis")
                - partial_range: Optional (row_start, row_end, col_start, col_end)
                - show: Whether to show the plot (default: False)

        Returns:
            Matplotlib figure object.
        """
        # Extract parameters with defaults
        title = kwargs.get("title", "Height Map (Seaborn)")
        colorbar_label = kwargs.get("colorbar_label", COLORBAR_LABEL)
        mode = kwargs.get("mode", "2d")
        filename = kwargs.get("filename", None)
        cmap = kwargs.get("cmap", "viridis")
        partial_range = kwargs.get("partial_range", None)
        show = kwargs.get("show", False)

        if mode == "3d":
            fig = self.plot_3d(height_map, **kwargs)
        elif kwargs.get("enhanced", False):
            fig = self.plot_enhanced_heatmap(
                height_map=height_map,
                colorbar_label=colorbar_label,
                filename=filename if filename else "seaborn_enhanced_heatmap.png",
                title=title,
                cmap=cmap
            )
        else:
            fig = self.plot_heatmap(
                height_map=height_map,
                colorbar_label=colorbar_label,
                filename=filename if filename else "seaborn_height_map.png",
                partial_range=partial_range,
                title=title,
                cmap=cmap
            )

        if show:
            plt.show()

        return fig

    @requires_seaborn
    def plot_heatmap(
        self,
        height_map: np.ndarray,
        colorbar_label: Optional[str] = None,
        filename: str = "seaborn_height_map.png",
        partial_range: Optional[Tuple[int, int, int, int]] = None,
        title: str = "Height Map (Seaborn)",
        cmap: str = "viridis"
    ) -> plt.Figure:
        """Create a heatmap visualization of the height map using Seaborn.

        Args:
            height_map: 2D numpy array of height values.
            colorbar_label: Label for the color bar (default: "Height (µm)").
            filename: Name of the image file to save.
            partial_range: Optional (row_start, row_end, col_start, col_end)
                           to subset the array.
            title: Plot title.
            cmap: Colormap to use.

        Returns:
            Matplotlib figure object.
        """
        if colorbar_label is None:
            colorbar_label = COLORBAR_LABEL

        if partial_range is not None:
            height_map = height_map[
                partial_range[0]:partial_range[1], partial_range[2]:partial_range[3]
            ]
            print(f"Partial render applied: rows {partial_range[0]}:{partial_range[1]}, "
                  f"cols {partial_range[2]}:{partial_range[3]}")

        sns.set(style="whitegrid")
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(height_map, cmap=cmap, cbar_kws={"label": colorbar_label}, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        if filename:
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"Seaborn heatmap saved to {filename}")

        return fig

    @requires_seaborn
    def plot_enhanced_heatmap(
        self,
        height_map: np.ndarray,
        colorbar_label: Optional[str] = None,
        filename: str = "seaborn_enhanced_heatmap.png",
        title: str = "Enhanced Height Map (Seaborn)",
        cmap: str = "viridis"
    ) -> plt.Figure:
        """Create a detailed 2D heatmap of the height map using Seaborn with contour annotations.

        Args:
            height_map: 2D numpy array of height values.
            colorbar_label: Label for the color bar (default: "Height (µm)").
            filename: Name of the image file to save.
            title: Plot title.
            cmap: Colormap to use.

        Returns:
            Matplotlib figure object.
        """
        if colorbar_label is None:
            colorbar_label = COLORBAR_LABEL

        sns.set(style="ticks")
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(height_map, cmap=cmap, cbar_kws={"label": colorbar_label}, ax=ax)

        rows, cols = height_map.shape
        if rows <= 1000 and cols <= 1000:
            # Add contour lines for better visualization of elevation changes
            x = np.arange(0, cols, 1)
            y = np.arange(0, rows, 1)
            X, Y = np.meshgrid(x, y)
            
            # Calculate appropriate number of contour levels
            levels = min(20, int(np.sqrt(rows * cols) / 10))
            
            # Draw contours
            contours = ax.contour(X, Y, height_map, levels=levels, colors='white', alpha=0.5, linewidths=0.5)
            
            # Add contour labels if the size is reasonable
            if rows <= 300 and cols <= 300:
                plt.clabel(contours, inline=True, fontsize=8, fmt='%.1f')
        
        # Add title and improve appearance
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        sns.despine()
        
        # Save the figure if a filename is provided
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"Enhanced Seaborn heatmap saved to {filename}")
            
        return fig

    def plot_2d(self, height_map: np.ndarray, **kwargs) -> plt.Figure:
        """Plot a 2D representation of the height map using Seaborn.

        Args:
            height_map: 2D numpy array representing the height map.
            **kwargs: Additional options for 2D plotting.

        Returns:
            Matplotlib figure object.
        """
        return self.plot(height_map, mode="2d", **kwargs)

    def plot_3d(self, height_map: np.ndarray, **kwargs) -> plt.Figure:
        """
        Plot a 3D representation of the height map.
        
        Since Seaborn doesn't directly support 3D plotting, this method
        falls back to matplotlib for 3D visualization.
        
        Args:
            height_map: 2D numpy array representing the height map.
            **kwargs: Additional options including:
                - z_scale: Z-axis scaling factor
                - title: Plot title
                - colormap: Colormap name
                - show_axes: Whether to show axes
                - figsize: Figure size in inches
                
        Returns:
            Matplotlib figure object.
        """
        logger.info("Seaborn doesn't directly support 3D plotting. Falling back to matplotlib.")
        
        # Extract parameters with defaults
        z_scale = kwargs.get("z_scale", 1.0)
        title = kwargs.get("title", "3D Height Map")
        colormap = kwargs.get("colormap", "viridis")
        show_axes = kwargs.get("show_axes", True)
        figsize = kwargs.get("figsize", (10, 8))
        
        # Create figure with matplotlib
        try:
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
            
            # Create coordinate grid
            rows, cols = height_map.shape
            x = np.arange(cols)
            y = np.arange(rows)
            x, y = np.meshgrid(x, y)
            
            # Plot surface
            surf = ax.plot_surface(
                x, y, height_map * z_scale,
                cmap=colormap,
                linewidth=0,
                antialiased=True
            )
            
            # Add colorbar and labels if showing axes
            if show_axes:
                fig.colorbar(surf, shrink=0.6, aspect=10, label='Height')
                ax.set_title(title)
                ax.set_xlabel("X Position")
                ax.set_ylabel("Y Position")
                ax.set_zlabel("Height")
            else:
                ax.set_axis_off()
                
            plt.tight_layout()
            return fig
        except ImportError:
            logger.error("3D plotting requires mpl_toolkits.mplot3d")
            # Create an informative 2D plot instead
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "3D plotting requires matplotlib's mpl_toolkits.mplot3d",
                   horizontalalignment='center', verticalalignment='center')
            ax.set_title(title)
            ax.set_xlabel("X Position")
            ax.set_ylabel("Y Position")
            return fig

    def save(self, plot_obj: Any, filename: str, **kwargs) -> Optional[str]:
        """
        Save the plot to a file.

        Implements the required save method from the BasePlotter interface.

        Args:
            plot_obj: Matplotlib figure object
            filename: Output filename
            **kwargs: Additional options such as:
                - dpi: Resolution in dots per inch (default: 300)
                - bbox_inches: Bounding box option (default: 'tight')
                - transparent: Whether to save with transparent background (default: False)

        Returns:
            Filename if saved successfully, None otherwise
        """
        try:
            # Create output directory if it doesn't exist
            directory = os.path.dirname(os.path.abspath(filename))
            os.makedirs(directory, exist_ok=True)

            # Extract save options
            dpi = kwargs.get("dpi", 300)
            bbox_inches = kwargs.get("bbox_inches", "tight")
            transparent = kwargs.get("transparent", False)

            # Save figure
            plot_obj.savefig(filename, dpi=dpi, bbox_inches=bbox_inches, transparent=transparent)
            logger.info(f"Plot saved to {filename}")

            # Close figure to free memory (optional)
            if kwargs.get("close", True):
                plt.close(plot_obj)

            return filename
        except Exception as e:
            logger.error(f"Error saving plot: {e}")
            return None

class SeabornSequencePlotter(BaseSequencePlotter):
    """
    Provides Seaborn-based visualizations for TMD sequences.
    Implements the BaseSequencePlotter interface.
    """
    NAME = "seaborn"
    DEFAULT_COLORMAP = "viridis"
    SUPPORTED_MODES = ["2d", "animation", "statistics"]
    REQUIRED_DEPENDENCIES = ["seaborn", "matplotlib.pyplot", "matplotlib.animation"]

    def __init__(self):
        """Initialize the Seaborn sequence plotter."""
        super().__init__()
        
        # Get seaborn and matplotlib modules
        self.sns = TMDFileUtilities.import_optional_dependency('seaborn')
        self.plt = TMDFileUtilities.import_optional_dependency('matplotlib.pyplot')
        
        # Check for animation support
        try:
            import matplotlib.animation
            self.animation = matplotlib.animation
        except ImportError as e:
            self.animation = None
            logger.warning("matplotlib.animation not available - animation features will be disabled")
            logger.debug(f"Error importing dependencies: {e}")
        
        # Create profile plotter for use in statistics
        self.profile_plotter = SeabornProfilePlotter()

    @requires_seaborn
    def visualize_sequence(self, frames: List[np.ndarray], **kwargs) -> plt.Figure:
        """
        Visualize a sequence of TMD height maps using Seaborn.
        
        Args:
            frames: List of 2D numpy arrays representing the height maps.
            **kwargs: Additional options such as:
                - n_frames: Number of frames to visualize (default: 5)
                - mode: Visualization mode (only '2d' is supported)
                - colormap: Colormap to use (default: 'viridis')
                - title: Plot title
                - show: Whether to display the plot immediately
                
        Returns:
            Matplotlib figure with the sequence visualization.
        """
        if not frames:
            logger.error("No frame data provided for visualization")
            fig, ax = self.plt.subplots()
            ax.text(0.5, 0.5, "No frames to visualize",
                    horizontalalignment='center', verticalalignment='center')
            return fig

        # Extract parameters with defaults
        n_frames = kwargs.get('n_frames', min(len(frames), 5))
        cmap = kwargs.get('colormap', self.DEFAULT_COLORMAP)
        figsize = kwargs.get('figsize', (15, 8))
        title = kwargs.get('title', 'TMD Sequence Visualization (Seaborn)')
        show = kwargs.get('show', False)
        
        # Select frames to display
        if len(frames) > n_frames:
            indices = np.linspace(0, len(frames) - 1, n_frames, dtype=int)
            selected_frames = [frames[i] for i in indices]
        else:
            indices = list(range(len(frames)))
            selected_frames = frames

        # Set Seaborn style
        self.sns.set(style="white")
        
        # Create facet grid for frames
        n_cols = min(5, n_frames)
        n_rows = (n_frames + n_cols - 1) // n_cols
        
        fig, axes = self.plt.subplots(n_rows, n_cols, figsize=figsize)
        fig.suptitle(title, fontsize=16)
        
        # Flatten axes for easy iteration
        if n_rows * n_cols == 1:
            axes = np.array([axes])
        else:
            axes = np.array(axes).flatten()
        
        # Plot each frame
        for i, (frame, idx) in enumerate(zip(selected_frames, indices)):
            if i < len(axes):
                ax = axes[i]
                # Use Seaborn's heatmap for each frame
                self.sns.heatmap(frame, cmap=cmap, cbar=False, ax=ax)
                ax.set_title(f"Frame {idx}")
                ax.set_xticks([])
                ax.set_yticks([])
        
        # Hide any unused subplots
        for i in range(len(selected_frames), len(axes)):
            axes[i].axis('off')
        
        # Add a single colorbar for all subplots
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        sm = self.plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax)
        
        # Adjust layout
        fig.tight_layout(rect=[0, 0, 0.9, 0.95])
        
        if show:
            self.plt.show()
            
        return fig

    @requires_seaborn
    def create_animation(self, frames: List[np.ndarray], **kwargs) -> Any:
        """
        Create a simple animation from sequence frames.
        
        Note: Seaborn doesn't natively support animations. This method creates
        a static figure with multiple frames and displays a message about the limitation.
        For true animations, use matplotlib.animation or plotly.
        
        Args:
            frames: List of 2D numpy arrays representing the sequence.
            **kwargs: Additional options such as:
                - fps: Frames per second (ignored, for API compatibility)
                - title: Animation title
                - colormap: Colormap name
                
        Returns:
            Matplotlib figure with multiple frames.
        """
        logger.info("Seaborn doesn't natively support animations. Creating a multi-frame visualization.")
        
        # Extract parameters
        title = kwargs.get('title', 'TMD Sequence (Animation Not Supported in Seaborn)')
        
        # Use visualize_sequence to create a multi-frame plot
        fig = self.visualize_sequence(
            frames,
            n_frames=min(len(frames), 9),  # Show at most 9 frames
            title=title,
            **{k: v for k, v in kwargs.items() if k != 'title'}
        )
        
        # Add a text note about animation limitation
        fig.text(0.5, 0.01, 
                "Note: True animations are not supported by Seaborn.\n"
                "Consider using Matplotlib or Plotly for animations.",
                ha='center', fontsize=10, style='italic')
        
        return fig

    @requires_seaborn
    def visualize_statistics(self, stats_data: Dict[str, List[float]], **kwargs) -> plt.Figure:
        """
        Visualize statistical data from the sequence using Seaborn.
        
        Args:
            stats_data: Dictionary with metric names as keys and lists of values.
            **kwargs: Additional options such as:
                - title: Plot title (default: "TMD Sequence Statistics")
                - figsize: Figure size (width, height) in inches
                - style: Plot style ('line', 'bar', 'box', 'violin') (default: 'line')
                - palette: Color palette for the plot
                - show_grid: Whether to show grid lines
                - x_label: Label for x-axis
                - y_label: Label for y-axis
                - metrics: List of specific metrics to include
                - show: Whether to display the plot immediately
                
        Returns:
            Matplotlib figure with the statistical visualization.
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
        style = kwargs.get('style', 'line').lower()
        palette = kwargs.get('palette', 'muted')
        show_grid = kwargs.get('show_grid', True)
        x_label = kwargs.get('x_label', 'Frame')
        y_label = kwargs.get('y_label', 'Value')
        show = kwargs.get('show', False)
        
        # Set Seaborn style
        self.sns.set(style="whitegrid" if show_grid else "white")
        
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
        
        # Prepare data for visualization
        data = []
        
        # Reshape data for seaborn
        for metric in valid_metrics:
            values = stats_data[metric]
            x_coords = x_values[:len(values)]
            for i, (x, y) in enumerate(zip(x_coords, values)):
                data.append({
                    'Frame': x,
                    'Value': y,
                    'Metric': metric
                })
        
        # Convert to DataFrame for Seaborn
        df = pd.DataFrame(data)
        
        # Create figure and appropriate visualization based on style
        fig, ax = self.plt.subplots(figsize=figsize)
        
        if style == 'line':
            # Line plot with error bands
            self.sns.lineplot(
                data=df, x='Frame', y='Value', hue='Metric',
                palette=palette, ax=ax
            )
        elif style == 'bar':
            self.sns.barplot(
                data=df, x='Frame', y='Value', hue='Metric',
                palette=palette, ax=ax
            )
        elif style == 'box':
            # Box plot of metrics
            self.sns.boxplot(
                data=df, x='Metric', y='Value',
                palette=palette, ax=ax
            )
        elif style == 'violin':
            # Violin plot of metrics
            self.sns.violinplot(
                data=df, x='Metric', y='Value',
                palette=palette, ax=ax
            )
        else:
            # Default to line plot
            logger.warning(f"Unknown style '{style}', defaulting to line plot")
            self.sns.lineplot(
                data=df, x='Frame', y='Value', hue='Metric',
                palette=palette, ax=ax
            )
        
        # Set labels and title
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        
        # Adjust legend
        if len(valid_metrics) > 1:
            ax.legend(title="Metrics")
        
        # Tight layout
        fig.tight_layout()
        
        if show:
            self.plt.show()
            
        return fig
        
    def save_figure(self, fig: Any, filename: str, **kwargs) -> Optional[str]:
        """
        Save a figure to a file.
        
        Args:
            fig: Matplotlib figure object.
            filename: Output filename.
            **kwargs: Additional options such as:
                - dpi: Resolution in dots per inch (default: 300)
                - bbox_inches: Bounding box option (default: 'tight')
                
        Returns:
            Filename if saved successfully, None otherwise.
        """
        try:
            # Create directory if it doesn't exist
            directory = os.path.dirname(os.path.abspath(filename))
            os.makedirs(directory, exist_ok=True)
            
            # Get save options
            dpi = kwargs.get('dpi', 300)
            bbox_inches = kwargs.get('bbox_inches', 'tight')
            
            # Save the figure
            fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
            logger.info(f"Figure saved to {filename}")
            
            return filename
        except Exception as e:
            logger.error(f"Error saving figure: {e}")
            return None


class SeabornProfilePlotter:
    """
    Seaborn plotter for profile analysis and distribution visualizations.
    
    This plotter creates statistical visualizations of height maps such as:
    - Height distributions
    - Profile analysis
    - Joint distributions
    - Correlation heatmaps
    """
    
    def __init__(self):
        """Initialize the Seaborn profile plotter."""
        # Import seaborn lazily
        self.sns = TMDFileUtilities.import_optional_dependency('seaborn')
        self.plt = TMDFileUtilities.import_optional_dependency('matplotlib.pyplot')
        
        if self.sns is None:
            logger.warning("Seaborn is not available - visualizations will be limited")
    
    @requires_seaborn
    def plot_height_distribution(self, height_map: np.ndarray, **kwargs) -> plt.Figure:
        """
        Create a distribution plot of height values.
        
        Args:
            height_map: 2D numpy array with height data
            **kwargs: Additional options including:
                - kde: Whether to include kernel density estimate (default: True)
                - bins: Number of histogram bins (default: 50)
                - figsize: Figure size as tuple (width, height) in inches
                - title: Plot title
                - show_stats: Whether to display statistics (default: True)
                
        Returns:
            Matplotlib figure with distribution plot
        """
        # Get parameters with defaults
        kde = kwargs.get('kde', True)
        bins = kwargs.get('bins', 50)
        figsize = kwargs.get('figsize', (10, 6))
        title = kwargs.get('title', 'Height Distribution')
        show_stats = kwargs.get('show_stats', True)
        
        # Set Seaborn style
        self.sns.set(style="whitegrid")
        
        # Create figure and axes
        fig, ax = self.plt.subplots(figsize=figsize)
        
        # Get height values as 1D array
        heights = height_map.flatten()
        
        # Create distribution plot
        self.sns.histplot(heights, kde=kde, bins=bins, ax=ax)
        
        # Add title and labels
        ax.set_title(title)
        ax.set_xlabel('Height')
        ax.set_ylabel('Frequency')
        
        # Add statistical annotations if requested
        if show_stats:
            stats_text = (
                f"Mean: {np.mean(heights):.4f}\n"
                f"Median: {np.median(heights):.4f}\n"
                f"Std Dev: {np.std(heights):.4f}\n"
                f"Min: {np.min(heights):.4f}\n"
                f"Max: {np.max(heights):.4f}"
            )
            
            # Position the text box in the upper right
            ax.text(0.95, 0.95, stats_text,
                    transform=ax.transAxes,
                    fontsize=10,
                    va='top', ha='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        fig.tight_layout()
        return fig

    @requires_seaborn
    def plot_profile_comparison(self, profiles: List[np.ndarray], 
                              labels: List[str] = None, **kwargs) -> plt.Figure:
        """
        Compare multiple height profiles.
        
        Args:
            profiles: List of 1D arrays with profile data
            labels: Labels for each profile
            **kwargs: Additional options
            
        Returns:
            Matplotlib figure with profile comparison
        """
        # Parameters with defaults
        figsize = kwargs.get('figsize', (12, 6))
        title = kwargs.get('title', 'Profile Comparison')
        
        # Set Seaborn style
        self.sns.set(style="darkgrid")
        
        # Create figure
        fig, ax = self.plt.subplots(figsize=figsize)
        
        # Generate default labels if needed
        if labels is None:
            labels = [f"Profile {i+1}" for i in range(len(profiles))]
        
        # Plot each profile
        for i, (profile, label) in enumerate(zip(profiles, labels)):
            x = np.arange(len(profile))
            self.sns.lineplot(x=x, y=profile, label=label, ax=ax)
        
        # Add title and labels
        ax.set_title(title)
        ax.set_xlabel('Position')
        ax.set_ylabel('Height')
        
        fig.tight_layout()
        return fig

    @requires_seaborn
    def plot_joint_distribution(self, height_map: np.ndarray, **kwargs) -> plt.Figure:
        """
        Create a joint distribution plot of heights and their spatial distribution.
        
        Args:
            height_map: 2D numpy array with height data
            **kwargs: Additional options
            
        Returns:
            Matplotlib figure with joint distribution
        """
        # Parameters with defaults
        figsize = kwargs.get('figsize', (10, 10))
        title = kwargs.get('title', 'Joint Height Distribution')
        
        # Set Seaborn style
        self.sns.set(style="white")
        
        # Create coordinates and height data
        h, w = height_map.shape
        Y, X = np.mgrid[:h, :w]
        coords_x = X.flatten()
        coords_y = Y.flatten()
        heights = height_map.flatten()
        
        # Create dataframe for seaborn
        data = pd.DataFrame({
            'X': coords_x,
            'Y': coords_y,
            'Height': heights
        })
        
        # Sample data if too large
        if len(data) > 10000:
            data = data.sample(10000, random_state=42)
            logger.info(f"Sampled data to 10000 points for joint distribution plot")
        
        # Create joint plot
        g = self.sns.jointplot(
            data=data, x='X', y='Y', hue='Height',
            kind='scatter', palette='viridis',
            height=figsize[0] // 2
        )
        
        # Add title
        g.fig.suptitle(title, y=1.02)
        
        g.fig.tight_layout()
        return g.fig
