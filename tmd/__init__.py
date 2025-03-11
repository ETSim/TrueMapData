"""
TMD (TrueMap Data) processor package.

This package provides utilities for reading, processing, and visualizing
TMD files that contain height map data.
"""

from typing import Any, Dict
import numpy as np
from tmd.processor import TMDProcessor

class TMD:
    """
    Main class for working
    with TMD files.
    """

    def __init__(self, file_path: str) -> None:
        """
        Initialize the TMD class with a file path.
        """
        self.file_path: str = file_path
        self.height_map: np.ndarray = None
        self.metadata: Dict[str, Any] = None

    def load(self):
        """
        Load the TMD file and process the data.
        """
        processor = TMDProcessor(self.file_path)
        data = processor.process()
        if data is not None:
            self.height_map = data["height_map"]
            self.metadata = processor.get_metadata()
        else:
            self.height_map = None
            self.metadata = None
            raise ValueError(f"Failed to process TMD file: {self.file_path}")
        
    def height_map(self):
        """
        Return the height map data.
        """
        return self.height_map
    
    def metadata(self):
        """
        Return the metadata for the TMD file.
        """
        return self.metadata