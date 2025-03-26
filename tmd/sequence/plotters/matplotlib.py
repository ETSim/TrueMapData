"""Matplotlib plotter for TMD sequences."""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

from .base import BasePlotter

logger = logging.getLogger(__name__)

class MatplotlibPlotter(BasePlotter):
    """Matplotlib-based plotter for TMD sequences."""
    
    def __init__(self):
        """Initialize the Matplotlib plotter."""
        super().__init__()
        
    def _check_dependencies(self) -> bool:
        """
        Check if required dependencies are available.
        
        Returns:
            True if all dependencies are available, False otherwise
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.animation as animation
            from mpl_toolkits.mplot3d import Axes3D
            return True
        except ImportError:
            logger.error("Matplotlib dependencies not found. Install with: pip install matplotlib")
            return False
    
    def create_animation(self, frames_data: List[np.ndarray], **kwargs) -> plt.Figure:
        """
        Create a matplotlib animation from sequence data.
        
        Args:
            frames_data: List of 2D arrays containing frame data
            **kwargs: Additional visualization options
            
        Returns:
            Matplotlib animation object
        """
        if not frames_data or len(frames_data) == 0:
            logger.error("No frame data provided for animation")
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No frames to animate", 
                    horizontalalignment='center', verticalalignment='center')
            return fig
        
        # Get options from kwargs
        fps = kwargs.get('fps', 10)
        colormap = kwargs.get('colormap', 'viridis')
        figsize = kwargs.get('figsize', (10, 8))
        title = kwargs.get('title', 'Sequence Animation')
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize)
        
        # First frame initialization
        im = ax.imshow(frames_data[0], cmap=colormap, animated=True)
        ax.set_title(title)
        
        plt.colorbar(im, ax=ax, label='Height')
        
        # Function to update the frame
        def update_frame(i):
            if i < len(frames_data):  # Check for valid index
                im.set_array(frames_data[i])
            return [im]
        
        # Create animation
        try:
            anim = animation.FuncAnimation(
                fig, 
                update_frame, 
                frames=len(frames_data),
                interval=1000/fps,
                blit=True
            )
            
            return anim
        except Exception as e:
            logger.error(f"Error creating animation: {str(e)}")
            # Return the figure for error handling
            return fig
    
    def visualize_sequence(self, frames_data: List[np.ndarray], **kwargs) -> plt.Figure:
        """
        Visualize sequence data using matplotlib.
        
        Args:
            frames_data: List of 2D arrays containing frame data
            **kwargs: Additional visualization options
            
        Returns:
            Matplotlib figure object
        """
        if not frames_data or len(frames_data) == 0:
            logger.error("No frame data provided for visualization")
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No frames to visualize", 
                    horizontalalignment='center', verticalalignment='center')
            return fig
        
        # Get options from kwargs
        n_frames = kwargs.get('n_frames', min(len(frames_data), 5))
        mode = kwargs.get('mode', '2d')
        colormap = kwargs.get('colormap', 'viridis')
        figsize = kwargs.get('figsize', (15, 8))
        
        # Choose frames to display
        if len(frames_data) >= n_frames:
            indices = np.linspace(0, len(frames_data) - 1, n_frames, dtype=int)
            selected_frames = [frames_data[i] for i in indices]
        else:
            # If fewer frames than requested, use all available frames
            indices = range(len(frames_data))
            selected_frames = frames_data
        
        if mode == '3d':
            # Create 3D visualizations
            fig = plt.figure(figsize=figsize)
            
            for i, frame in enumerate(selected_frames):
                # Create subplot
                ax = fig.add_subplot(1, len(selected_frames), i+1, projection='3d')
                
                # Get dimensions
                height, width = frame.shape
                y, x = np.mgrid[0:height, 0:width]
                
                # Plot the surface
                surf = ax.plot_surface(
                    x, y, frame,
                    cmap=colormap,
                    linewidth=0,
                    antialiased=True
                )
                
                # Customize appearance
                ax.set_title(f"Frame {indices[i]}")
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Height')
                
            plt.tight_layout()
            return fig
            
        else:  # Default to 2D mode
            # Create a grid of 2D plots
            fig, axes = plt.subplots(1, len(selected_frames), figsize=figsize)
            
            # Handle case where n_frames is 1
            if len(selected_frames) == 1:
                axes = np.array([axes])
                
            # Plot each frame
            for i, (frame, ax) in enumerate(zip(selected_frames, axes)):
                im = ax.imshow(frame, cmap=colormap)
                ax.set_title(f"Frame {indices[i]}")
                ax.set_xticks([])
                ax.set_yticks([])
                plt.colorbar(im, ax=ax, label='Height')
                
            plt.tight_layout()
            return fig
    
    def visualize_statistics(self, stats_data: Dict[str, List[float]], **kwargs) -> plt.Figure:
        """
        Visualize statistical data from a sequence using matplotlib.
        
        Args:
            stats_data: Dictionary of statistical data (metric_name -> list of values)
            **kwargs: Additional visualization options
            
        Returns:
            Matplotlib figure object
        """
        if not stats_data:
            logger.error("No statistical data provided for visualization")
            return None
        
        # Get options from kwargs
        figsize = kwargs.get('figsize', (12, 8))
        metrics = kwargs.get('metrics', list(stats_data.keys()))
        x_label = kwargs.get('x_label', 'Frame')
        title = kwargs.get('title', 'Sequence Statistics')
        
        # Filter metrics to those available in the data
        metrics = [m for m in metrics if m in stats_data and m != 'timestamps']
        
        if not metrics:
            logger.error("No valid metrics found in the data")
            return None
        
        # Get x-axis values (timestamps or indices)
        x_values = stats_data.get('timestamps', list(range(len(stats_data[metrics[0]]))))
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot each metric
        for metric in metrics:
            values = stats_data[metric]
            ax.plot(x_values[:len(values)], values, label=metric, marker='o')
        
        # Customize appearance
        ax.set_xlabel(x_label)
        ax.set_ylabel('Value')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig
    
    def save_figure(self, fig: plt.Figure, filename: str, **kwargs) -> Optional[str]:
        """
        Save a matplotlib figure to disk.
        
        Args:
            fig: Matplotlib figure object to save
            filename: Output filename
            **kwargs: Additional saving options
            
        Returns:
            Path to saved file or None if saving failed
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
            
            # Get save options
            dpi = kwargs.get('dpi', 300)
            bbox_inches = kwargs.get('bbox_inches', 'tight')
            
            # Save the figure
            plt.figure(fig.number)
            plt.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
            logger.info(f"Saved figure to {filename}")
            
            return filename
            
        except Exception as e:
            logger.error(f"Error saving figure: {e}")
            return None
