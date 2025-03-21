"""
TrueMap & GelSight TMD Processor package.

A Python-based TMD file processor with visualization and export capabilities for height maps.
"""

import os
import logging

import numpy as np

from .plotters.matplotlib import plot_height_map_matplotlib
from .processor import TMDProcessor

# Check if plotly is available for interactive visualizations
try:
    from .plotters.plotly import (
        plot_height_map_3d, 
        plot_height_map_2d, 
        plot_cross_section_plotly,
        plot_multiple_profiles
    )
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    
# Check if polyscope is available for 3D visualizations
try:
    from .plotters.polyscope import PolyscopePlotter
    HAS_POLYSCOPE = True
except ImportError:
    HAS_POLYSCOPE = False

logger = logging.getLogger(__name__)


class TMD:
    def __init__(self, file_path: str):
        """
        Initialize a TMD object.

        Args:
            file_path: Path to the TMD file
        """
        self.file_path = file_path
        self.processor = TMDProcessor(file_path)
        self.height_map_data = None
        self.metadata_dict = None
        self.load()

    def load(self):
        """
        Load the TMD file.
        """
        data = self.processor.process()
        if data is None:
            logger.error(f"Failed to load TMD file: {self.file_path}")
            return False
            
        self.metadata_dict = self.processor.get_metadata()
        self.height_map_data = self.processor.get_height_map()
        return True

    def metadata(self):
        """
        Get the metadata of the TMD file.

        Returns:
            Metadata dictionary
        """
        return self.metadata_dict

    def height_map(self):
        """
        Get the height map of the TMD file.

        Returns:
            Height map as a 2D numpy array
        """
        return self.height_map_data

    def plot_3D(self, output_dir: str = ".", z_scale: float = 1.0, show: bool = False, colorbar_label: str = "Height (normalized)"):
        """
        Plot the height map in 3D using matplotlib.
        
        Args:
            output_dir: Directory where to save the output image
            z_scale: Vertical scaling factor
            show: Whether to show the plot interactively
            colorbar_label: Label for the color bar
            
        Returns:
            Path to the saved image file
        """
        filename = os.path.join(output_dir, "height_map_3d_matplotlib.png")
        return plot_height_map_matplotlib(
            height_map=self.height_map_data,
            colorbar_label=colorbar_label,
            filename=filename,
            z_scale=z_scale,
            show=show
        )
    
    def plot_interactive(self, output_dir: str = ".", plot_type: str = "3d", title: str = None, show: bool = True):
        """
        Create an interactive plot of the height map using Plotly.
        
        Args:
            output_dir: Directory where to save the output HTML
            plot_type: Type of plot ('3d', '2d', or 'profile')
            title: Plot title (default: derived from filename)
            show: Whether to show the plot in a browser
            
        Returns:
            Plotly figure object or None if Plotly is not available
        """
        if not HAS_PLOTLY:
            logger.warning("Plotly is not available. Install with: pip install plotly")
            return None
            
        if title is None:
            title = f"TMD: {os.path.basename(self.file_path)}"
            
        # Create output filename based on plot type
        if plot_type == '2d':
            filename = os.path.join(output_dir, "height_map_2d.html")
            return plot_height_map_2d(
                self.height_map_data,
                title=title,
                filename=filename
            )
        elif plot_type == 'profile':
            # Extract middle row profile
            profile_row = self.height_map_data.shape[0] // 2
            x_length = self.metadata_dict.get('x_length', self.height_map_data.shape[1])
            x_positions = np.linspace(0, x_length, self.height_map_data.shape[1])
            heights = self.height_map_data[profile_row, :]
            
            filename = os.path.join(output_dir, "profile.html")
            return plot_cross_section_plotly(
                x_positions,
                heights,
                title=f"{title} - Profile at Row {profile_row}",
                filename=filename
            )
        else:  # Default to 3D
            filename = os.path.join(output_dir, "height_map_3d.html")
            return plot_height_map_3d(
                self.height_map_data,
                title=title,
                filename=filename
            )
    
    def visualize_polyscope(self, min_scale: float = 0.0, max_scale: float = 100.0):
        """
        Create an interactive 3D visualization using Polyscope.
        
        Args:
            min_scale: Minimum vertical scale for the slider
            max_scale: Maximum vertical scale for the slider
            
        Returns:
            True if successful, False if Polyscope is not available
        """
        if not HAS_POLYSCOPE:
            logger.warning("Polyscope is not available. Install with: pip install polyscope")
            return False
            
        # Get dimensions from metadata if available
        try:
            x_range = (0, self.metadata_dict.get('x_length', self.height_map_data.shape[1] - 1))
            y_range = (0, self.metadata_dict.get('y_length', self.height_map_data.shape[0] - 1))
        except:
            x_range = (0, self.height_map_data.shape[1] - 1)
            y_range = (0, self.height_map_data.shape[0] - 1)
            
        # Create a Polyscope plotter
        plotter = PolyscopePlotter(backend=None)
        
        # Plot the height map
        name = os.path.basename(self.file_path)
        mesh = plotter.plot_height_map(
            self.height_map_data,
            x_range=x_range,
            y_range=y_range,
            name=name,
            enabled=True,
            add_height_slider=True,
            min_scale=min_scale,
            max_scale=max_scale
        )
        
        # Show the visualization
        plotter.show()
        return True

    def get_stats(self):
        """
        Get statistics about the height map.
        
        Returns:
            Dictionary with height map statistics
        """
        return self.processor.get_stats()
    
    def export_metadata(self, output_path=None):
        """
        Export metadata to a text file.
        
        Args:
            output_path: Path to the output file (default: auto-generate in same directory)
            
        Returns:
            Path to the exported metadata file
        """
        return self.processor.export_metadata(output_path)


def load(file_path: str):
    """
    Convenience function to load a TMD file.

    Args:
        file_path: Path to the TMD file

    Returns:
        Tuple of (metadata, height_map)
    """
    processor = TMDProcessor(file_path)
    data = processor.process()
    if data is None:
        return None, None
    return processor.get_metadata(), processor.get_height_map()
