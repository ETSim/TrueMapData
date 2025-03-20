"""
TrueMap & GelSight TMD Processor package.

A Python-based TMD file processor with visualization and export capabilities for height maps.
"""

import os

import numpy as np

from .plotters.matplotlib import plot_height_map_matplotlib
from .processor import TMDProcessor


class TMD:
    def __init__(self, file_path: str):
        """
        Initialize a TMD object.

        Args:
            file_path: Path to the TMD file
        """
        self.processor = TMDProcessor(file_path)
        self.height_map_data = None
        self.metadata_dict = None
        self.load()

    def load(self):
        """
        Load the TMD file.
        """
        self.processor.process()
        self.metadata_dict = self.processor.get_metadata()
        self.height_map_data = self.processor.get_height_map()

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

    def plot_3D(self, output_dir: str = ".", z: float = 1.0):
        """
        Plot the height map in 3D.
        """
        plot_height_map_matplotlib(
            height_map=self.height_map_data,
            colorbar_label="Height (normalized)",
            filename=os.path.join(output_dir, "height_map_3d_matplotlib.png"),
        )


def load(file_path: str):
    """
    Convenience function to load a TMD file.

    Args:
        file_path: Path to the TMD file

    Returns:
        Tuple of (metadata, height_map)
    """
    processor = TMDProcessor(file_path)
    processor.process()
    return processor.get_metadata(), processor.get_height_map()
