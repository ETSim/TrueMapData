"""
Base exporter for TMD sequence data.

This module provides the base class for exporters that can save sequence data
to various formats.
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np

# Set up logging
logger = logging.getLogger(__name__)

class BaseExporter(ABC):
    """
    Base class for sequence exporters.
    
    All exporters should inherit from this class and implement their specific
    export methods.
    """
    
    def __init__(self):
        """Initialize the base exporter."""
        pass
    
    def ensure_output_dir(self, output_dir: str) -> bool:
        """
        Ensure output directory exists, creating it if necessary.
        
        Args:
            output_dir: Directory path to ensure exists
            
        Returns:
            True if directory exists or was created, False on error
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Error creating output directory {output_dir}: {e}")
            return False
    
    def sanitize_filename(self, filename: str) -> str:
        """
        Clean a filename to make it safe for file systems.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        # Replace problematic characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
            
        return filename

    @abstractmethod
    def export_sequence_differences(
        self,
        frames_data: List[np.ndarray],
        output_dir: str,
        timestamps: List[Any] = None,
        format: str = 'png',
        normalize: bool = True,
        colormap: str = 'RdBu',
        **kwargs
    ) -> List[str]:
        """
        Export differences between consecutive frames.
        
        Args:
            frames_data: List of heightmap difference arrays
            output_dir: Directory to save the output files
            timestamps: Optional list of timestamps or labels for each difference
            format: Output format (depends on exporter)
            normalize: Whether to normalize difference values before saving
            colormap: Color map for visualization
            **kwargs: Additional keyword arguments for exporter-specific options
            
        Returns:
            List of paths to saved files
        """
        pass
    
    @abstractmethod
    def export_normal_maps(
        self,
        frames_data: List[np.ndarray],
        output_dir: str,
        timestamps: List[Any] = None,
        format: str = 'png',
        z_scale: float = 10.0,
        **kwargs
    ) -> List[str]:
        """
        Export normal maps of the frames.
        
        Args:
            frames_data: List of heightmap arrays
            output_dir: Directory to save the output files
            timestamps: Optional list of timestamps or labels for each frame
            format: Output format (depends on exporter)
            z_scale: Z-scale factor for normal map generation
            **kwargs: Additional keyword arguments for exporter-specific options
            
        Returns:
            List of paths to saved files
        """
        pass
