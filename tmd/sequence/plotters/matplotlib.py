"""
Matplotlib-based visualization for TMD sequence data.

This module provides Matplotlib-based visualization functions for sequences of height maps.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Union, Tuple

import numpy as np

from .base import BasePlotter

# Set up logging
logger = logging.getLogger(__name__)

class MatplotlibPlotter(BasePlotter):
    """
    Matplotlib-based plotter for TMD sequence data.
    
    This class provides visualization methods for TMD sequence data using Matplotlib.
    """
    
    def __init__(self):
        """Initialize matplotlib plotter."""
        self._has_matplotlib = self._check_matplotlib()
        super().__init__()
    
    def _check_matplotlib(self) -> bool:
        """
        Check if matplotlib is available.
        
        Returns:
            True if matplotlib is available, False otherwise
        """
        try:
            import matplotlib.pyplot as plt
            return True
        except ImportError:
            logger.warning("Matplotlib not installed. Install with: pip install matplotlib")
            return False
    
    def check_dependencies(self) -> bool:
        """
        Check if required dependencies are available.
        
        Returns:
            True if dependencies are available, False otherwise
        """
        if not self._has_matplotlib:
            return False
            
        try:
            import matplotlib.pyplot as plt
            import matplotlib.animation as animation
            return True
        except ImportError:
            logger.warning("Required matplotlib modules not available")
            return False
    
    def visualize_sequence(
        self,
        frames_data: List[np.ndarray],
        timestamps: Optional[List[Any]] = None,
        view_type: str = '2d',
        colorscale: str = 'viridis',
        output_dir: Optional[str] = None,
        show: bool = True,
        **kwargs
    ) -> List[Any]:
        """
        Visualize a sequence of height maps using Matplotlib.
        
        Args:
            frames_data: List of height map arrays
            timestamps: Optional list of timestamps or labels
            view_type: Type of visualization ('2d' or '3d')
            colorscale: Matplotlib colormap name
            output_dir: Optional directory to save visualizations
            show: Whether to display the plots
            **kwargs: Additional visualization options
            
        Returns:
            List of created figure objects
        """
        if not self._has_matplotlib:
            logger.error("Matplotlib is not installed. Cannot visualize sequence.")
            return []
            
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("Failed to import matplotlib")
            return []
            
        if not frames_data:
            logger.warning("No frames to visualize")
            return []
            
        # Use frame indices if no timestamps provided
        if timestamps is None:
            timestamps = [f"Frame {i+1}" for i in range(len(frames_data))]
            
        # Ensure we have enough timestamps
        if len(timestamps) < len(frames_data):
            timestamps = timestamps + [f"Frame {i+1}" for i in range(len(timestamps), len(frames_data))]
        
        figures = []
        
        # Create visualizations for each frame
        for i, (frame, timestamp) in enumerate(zip(frames_data, timestamps)):
            # Create figure
            fig = plt.figure(figsize=kwargs.get('figsize', (10, 8)))
            
            # Create visualization based on view type
            if view_type == '3d':
                try:
                    from mpl_toolkits.mplot3d import Axes3D
                    
                    # Create 3D plot
                    ax = fig.add_subplot(111, projection='3d')
                    
                    # Create mesh grid for 3D plot
                    rows, cols = frame.shape
                    x = np.arange(0, cols)
                    y = np.arange(0, rows)
                    X, Y = np.meshgrid(x, y)
                    
                    # Plot the surface
                    surf = ax.plot_surface(
                        X, Y, frame,
                        cmap=colorscale,
                        linewidth=0,
                        antialiased=True
                    )
                    
                    # Add color bar
                    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Height')
                    
                    # Set labels
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Height')
                    
                except (ImportError, RuntimeError) as e:
                    logger.error(f"Error creating 3D visualization: {e}")
                    logger.info("Falling back to 2D visualization")
                    
                    # Fall back to 2D
                    ax = fig.add_subplot(111)
                    im = ax.imshow(frame, cmap=colorscale, origin='lower')
                    fig.colorbar(im, ax=ax, label='Height')
            else:
                # Create 2D plot
                ax = fig.add_subplot(111)
                im = ax.imshow(frame, cmap=colorscale, origin='lower')
                fig.colorbar(im, ax=ax, label='Height')
            
            # Add title with timestamp
            ax.set_title(f"{timestamp}")
            
            # Save if output directory provided
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                filename = os.path.join(output_dir, f"frame_{i+1}.png")
                plt.savefig(filename, dpi=kwargs.get('dpi', 100), bbox_inches='tight')
                logger.info(f"Saved visualization to {filename}")
            
            # Display if requested
            if show:
                plt.show()
            else:
                plt.close(fig)
                
            figures.append(fig)
            
        return figures
    
    def create_animation(
        self,
        frames_data: List[np.ndarray],
        timestamps: Optional[List[Any]] = None,
        filename: Optional[str] = None,
        fps: int = 10,
        width: int = 800,
        height: int = 600,
        colorscale: str = 'viridis',
        view_type: str = '3d',
        show: bool = False,
        **kwargs
    ) -> Any:
        """
        Create an animation of a sequence using Matplotlib.
        
        Args:
            frames_data: List of heightmap arrays
            timestamps: Optional list of timestamps or labels for each frame
            filename: Optional filename to save the output
            fps: Frames per second for animation
            width: Figure width in pixels (converted to inches for matplotlib)
            height: Figure height in pixels (converted to inches for matplotlib)
            colorscale: Matplotlib colormap name
            view_type: '3d' for surface, '2d' for heatmap
            show: Whether to display the animation
            **kwargs: Additional animation options
            
        Returns:
            Matplotlib animation object or None if failed
        """
        if not self._has_matplotlib:
            logger.error("Matplotlib is not installed. Cannot create animation.")
            return None
            
        if not frames_data:
            logger.warning("No frames to animate.")
            return None
            
        # Check for additional dependencies
        try:
            import matplotlib.pyplot as plt
            import matplotlib.animation as animation
            from matplotlib import cm
        except ImportError:
            logger.error("Required matplotlib modules not available.")
            return None
            
        # Use frame indices if no timestamps provided
        if timestamps is None:
            timestamps = list(range(len(frames_data)))
            
        # Convert pixels to inches (assuming 100 DPI)
        width_inches = width / 100
        height_inches = height / 100
        
        # Create figure
        fig = plt.figure(figsize=(width_inches, height_inches), dpi=100)
        
        # Choose visualization type
        if view_type == '3d':
            try:
                from mpl_toolkits.mplot3d import Axes3D
                
                # Create 3D plot
                ax = fig.add_subplot(111, projection='3d')
                
                # Create mesh grid for 3D plot
                rows, cols = frames_data[0].shape
                x = np.arange(0, cols)
                y = np.arange(0, rows)
                X, Y = np.meshgrid(x, y)
                
                # Initial plot
                surf = ax.plot_surface(
                    X, Y, frames_data[0],
                    cmap=colorscale,
                    linewidth=0,
                    antialiased=True
                )
                
                # Set labels
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Height')
                
                # Set title with timestamp
                title = ax.set_title(f"Frame: {timestamps[0]}")
                
                # Update function for animation
                def update(frame_idx):
                    ax.clear()
                    surf = ax.plot_surface(
                        X, Y, frames_data[frame_idx],
                        cmap=colorscale,
                        linewidth=0,
                        antialiased=True
                    )
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Height')
                    title = ax.set_title(f"Frame: {timestamps[frame_idx]}")
                    return surf,
                    
            except (ImportError, RuntimeError) as e:
                logger.error(f"Error creating 3D animation: {e}")
                logger.info("Falling back to 2D animation")
                
                # Fall back to 2D
                ax = fig.add_subplot(111)
                im = ax.imshow(frames_data[0], cmap=colorscale, origin='lower')
                plt.colorbar(im, label='Height')
                
                # Set title with timestamp
                title = ax.set_title(f"Frame: {timestamps[0]}")
                
                # Update function for animation
                def update(frame_idx):
                    im.set_array(frames_data[frame_idx])
                    title.set_text(f"Frame: {timestamps[frame_idx]}")
                    return im, title
        else:
            # Create 2D plot
            ax = fig.add_subplot(111)
            im = ax.imshow(frames_data[0], cmap=colorscale, origin='lower')
            plt.colorbar(im, label='Height')
            
            # Set title with timestamp
            title = ax.set_title(f"Frame: {timestamps[0]}")
            
            # Update function for animation
            def update(frame_idx):
                im.set_array(frames_data[frame_idx])
                title.set_text(f"Frame: {timestamps[frame_idx]}")
                return im, title
        
        # Create animation
        anim = animation.FuncAnimation(
            fig,
            update,
            frames=len(frames_data),
            interval=1000/fps, # Convert fps to interval in milliseconds
            blit=False
        )
        
        # Save if filename provided
        if filename:
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
            
            # Determine writer based on file extension
            if filename.lower().endswith('.mp4'):
                writer = 'ffmpeg'
            elif filename.lower().endswith('.gif'):
                writer = 'pillow'
            else:
                writer = animation.writers.list()[0]
                
            try:
                anim.save(
                    filename,
                    writer=writer,
                    fps=fps,
                    dpi=100,
                    **kwargs.get('save_kwargs', {})
                )
                logger.info(f"Saved animation to {filename}")
            except Exception as e:
                logger.error(f"Error saving animation: {e}")
                
        # Show if requested
        if show:
            plt.show()
            
        return anim
    
    def visualize_statistics(
        self,
        stats_data: Dict[str, Any],
        output_dir: Optional[str] = None,
        show: bool = True,
        **kwargs
    ) -> List[Any]:
        """
        Visualize statistics about a sequence using Matplotlib.
        
        Args:
            stats_data: Dictionary of statistical data
            output_dir: Optional directory to save visualizations
            show: Whether to display the plots
            **kwargs: Additional visualization options
            
        Returns:
            List of created figure objects
        """
        if not self._has_matplotlib:
            logger.error("Matplotlib is not installed. Cannot visualize statistics.")
            return []
            
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("Failed to import matplotlib")
            return []
            
        if not stats_data:
            logger.warning("No statistics data to visualize")
            return []
            
        figures = []
        
        # Create time series plot of min, max, mean
        if all(key in stats_data for key in ['timestamps', 'min', 'max', 'mean']):
            fig, ax = plt.subplots(figsize=kwargs.get('figsize', (12, 6)))
            
            timestamps = stats_data['timestamps']
            min_vals = stats_data['min']
            max_vals = stats_data['max']
            mean_vals = stats_data['mean']
            
            # Plot values
            ax.plot(timestamps, min_vals, 'b-', label='Min')
            ax.plot(timestamps, max_vals, 'r-', label='Max')
            ax.plot(timestamps, mean_vals, 'g-', label='Mean')
            
            # Fill between min and max
            ax.fill_between(timestamps, min_vals, max_vals, alpha=0.2, color='gray')
            
            # Add labels and legend
            ax.set_xlabel('Time')
            ax.set_ylabel('Height')
            ax.set_title('Height Statistics Over Time')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Rotate x-axis labels if they're strings
            if isinstance(timestamps[0], str):
                plt.xticks(rotation=45)
                
            plt.tight_layout()
            
            # Save if output directory provided
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                filename = os.path.join(output_dir, "height_stats_over_time.png")
                plt.savefig(filename, dpi=kwargs.get('dpi', 100), bbox_inches='tight')
                logger.info(f"Saved statistics visualization to {filename}")
                
            # Display if requested
            if show:
                plt.show()
            elif not output_dir:
                plt.close(fig)
                
            figures.append(fig)
            
        # Create histogram of height distribution if available
        if 'histogram_data' in stats_data and 'bin_edges' in stats_data:
            fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))
            
            hist_data = stats_data['histogram_data']
            bin_edges = stats_data['bin_edges']
            
            # Plot histogram
            ax.bar(bin_edges[:-1], hist_data, width=bin_edges[1] - bin_edges[0], alpha=0.7)
            
            # Add labels
            ax.set_xlabel('Height')
            ax.set_ylabel('Frequency')
            ax.set_title('Height Distribution')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            
            # Save if output directory provided
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                filename = os.path.join(output_dir, "height_distribution.png")
                plt.savefig(filename, dpi=kwargs.get('dpi', 100), bbox_inches='tight')
                logger.info(f"Saved distribution visualization to {filename}")
                
            # Display if requested
            if show:
                plt.show()
            elif not output_dir:
                plt.close(fig)
                
            figures.append(fig)
            
        return figures
    
    def save_figure(
        self, 
        figure: Any,
        filename: str,
        dpi: int = 300,
        **kwargs
    ) -> Optional[str]:
        """
        Save a matplotlib figure to a file.
        
        Args:
            figure: Matplotlib figure object to save
            filename: Output filename
            dpi: Resolution in dots per inch
            **kwargs: Additional save options
            
        Returns:
            Path to saved file or None if saving failed
        """
        if not self._has_matplotlib:
            logger.error("Matplotlib is not installed. Cannot save figure.")
            return None
            
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("Failed to import matplotlib")
            return None
            
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
            
            # Save the figure
            figure.savefig(filename, dpi=dpi, bbox_inches='tight', **kwargs)
            logger.info(f"Saved figure to {filename}")
            
            # Close the figure if specified
            if kwargs.get('close', True):
                plt.close(figure)
                
            return filename
        except Exception as e:
            logger.error(f"Error saving figure: {e}")
            return None
