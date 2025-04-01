"""
Factory Module for Sequence Exporters

This module provides a factory class for creating different exporters
for height map sequences (gif, video, PowerPoint) and centralizes export
functionality to reduce code duplication in the TMDSequence class.
"""

import os
import logging
from typing import Dict, Type, Optional, Any, List, Union
from pathlib import Path
import numpy as np

from .base import BaseExporter
from .gif import GifExporter
from .video import VideoExporter
from .powerpoint import PowerPointExporter

logger = logging.getLogger(__name__)

class SequenceExporterFactory:
    """
    Factory class for creating sequence exporters.
    
    This factory provides centralized creation of exporters and 
    export functionality, reducing code duplication in the TMDSequence class.
    """
    
    _exporters: Dict[str, Type[BaseExporter]] = {
        'gif': GifExporter,
        'video': VideoExporter,
        'mp4': VideoExporter,
        'avi': VideoExporter,
        'powerpoint': PowerPointExporter,
        'pptx': PowerPointExporter,
    }
    
    # Format aliases and extensions mapping
    _format_mapping: Dict[str, str] = {
        'gif': 'gif',
        'animated_gif': 'gif',
        'video': 'mp4',
        'mp4': 'mp4',
        'avi': 'avi',
        'powerpoint': 'pptx',
        'ppt': 'pptx',
        'pptx': 'pptx',
    }
    
    @classmethod
    def get_exporter(cls, format_type: str) -> Optional[BaseExporter]:
        """
        Get an exporter instance for the specified format.
        
        Args:
            format_type: The format type (gif, video, powerpoint, etc.)
            
        Returns:
            An instance of the appropriate exporter, or None if not found
        """
        format_type = format_type.lower()
        
        # Try to get the canonical format from mapping
        canonical_format = cls._format_mapping.get(format_type, format_type)
        exporter_class = cls._exporters.get(canonical_format)
        
        if exporter_class:
            return exporter_class()
        
        # Try all registered exporters if not found in mapping
        for exporter_class in cls._exporters.values():
            if exporter_class.supports_format(format_type):
                return exporter_class()
        
        logger.error(f"No exporter found for format: {format_type}")
        return None
    
    @classmethod
    def register_exporter(cls, format_type: str, exporter_class: Type[BaseExporter]) -> None:
        """
        Register a new exporter class for a specific format.
        
        Args:
            format_type: The format type (e.g., 'custom_format')
            exporter_class: The exporter class to register
        """
        format_type = format_type.lower()
        cls._exporters[format_type] = exporter_class
        cls._format_mapping[format_type] = format_type
        logger.debug(f"Registered exporter for format: {format_type}")
    
    @classmethod
    def supported_formats(cls) -> List[str]:
        """
        Get a list of supported export formats.
        
        Returns:
            List of supported format strings
        """
        return list(cls._format_mapping.keys())
    
    @classmethod
    def get_file_extension(cls, format_type: str) -> str:
        """
        Get the file extension for a given format type.
        
        Args:
            format_type: Format type (e.g., 'video', 'gif')
            
        Returns:
            File extension (e.g., 'mp4', 'gif')
        """
        format_type = format_type.lower()
        return cls._format_mapping.get(format_type, format_type)

    @classmethod
    def export_sequence(cls, 
                       frames: List[np.ndarray], 
                       output_path: str, 
                       format_type: str, 
                       **kwargs) -> Optional[str]:
        """
        Export frames using the appropriate exporter.
        
        Args:
            frames: List of 2D numpy arrays
            output_path: Path where the output should be saved
            format_type: Type of export (gif, video, powerpoint)
            **kwargs: Additional options for the specific exporter
            
        Returns:
            Path to the exported file if successful, None otherwise
        """
        # Get the appropriate exporter
        exporter = cls.get_exporter(format_type)
        if not exporter:
            logger.error(f"No exporter available for format '{format_type}'")
            return None
            
        # Validate frames
        if not frames or not isinstance(frames, list) or len(frames) == 0:
            logger.error("No frames provided for export")
            return None
            
        # Ensure the output path has the correct extension
        output_path = cls._ensure_extension(output_path, format_type)
            
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        try:
            # Perform the export
            return exporter.export(frames, output_path, **kwargs)
        except Exception as e:
            logger.error(f"Error during export to {format_type}: {e}", exc_info=True)
            return None
    
    @classmethod
    def export_gif(cls, 
                  frames: List[np.ndarray], 
                  output_path: str, 
                  fps: float = 10.0, 
                  **kwargs) -> Optional[str]:
        """
        Export frames as an animated GIF.
        
        Args:
            frames: List of 2D numpy arrays
            output_path: Path where the GIF should be saved
            fps: Frames per second (default: 10.0)
            **kwargs: Additional options for the GIF exporter
            
        Returns:
            Path to the exported GIF if successful, None otherwise
        """
        kwargs['fps'] = fps
        return cls.export_sequence(frames, output_path, 'gif', **kwargs)
    
    @classmethod
    def export_video(cls, 
                    frames: List[np.ndarray], 
                    output_path: str, 
                    fps: float = 30.0, 
                    **kwargs) -> Optional[str]:
        """
        Export frames as a video file.
        
        Args:
            frames: List of 2D numpy arrays
            output_path: Path where the video should be saved
            fps: Frames per second (default: 30.0)
            **kwargs: Additional options for the video exporter
            
        Returns:
            Path to the exported video if successful, None otherwise
        """
        kwargs['fps'] = fps
        return cls.export_sequence(frames, output_path, 'video', **kwargs)
    
    @classmethod
    def export_powerpoint(cls, 
                         frames: List[np.ndarray], 
                         output_path: str, 
                         **kwargs) -> Optional[str]:
        """
        Export frames as a PowerPoint presentation.
        
        Args:
            frames: List of 2D numpy arrays
            output_path: Path where the PowerPoint should be saved
            **kwargs: Additional options for the PowerPoint exporter
            
        Returns:
            Path to the exported PowerPoint if successful, None otherwise
        """
        return cls.export_sequence(frames, output_path, 'powerpoint', **kwargs)
    
    @classmethod
    def export_frames_as_images(cls,
                               frames: List[np.ndarray],
                               output_dir: str,
                               format_type: str = 'png',
                               base_filename: str = 'frame',
                               colormap: str = 'viridis',
                               **kwargs) -> List[str]:
        """
        Export individual frames as separate image files.
        
        Args:
            frames: List of 2D numpy arrays
            output_dir: Directory where images should be saved
            format_type: Image format ('png', 'jpg', 'tif', etc.)
            base_filename: Base name for frame files
            colormap: Matplotlib colormap to use
            **kwargs: Additional export options
            
        Returns:
            List of paths to saved image files
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
            
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Get DPI setting
            dpi = kwargs.get('dpi', 100)
            
            # Get optional frame timestamps
            timestamps = kwargs.get('timestamps', None)
            
            # Prepare output paths
            output_files = []
            
            # Export each frame
            for i, frame in enumerate(frames):
                # Create filename with padding
                filename = f"{base_filename}_{i:04d}.{format_type.lower()}"
                filepath = os.path.join(output_dir, filename)
                
                # Create figure and plot
                fig = Figure(figsize=(8, 6), dpi=dpi)
                canvas = FigureCanvas(fig)
                ax = fig.add_subplot(111)
                
                # Plot height map
                im = ax.imshow(frame, cmap=colormap)
                fig.colorbar(im, ax=ax)
                
                # Add timestamp if available
                if timestamps and i < len(timestamps):
                    ax.set_title(f"Frame: {timestamps[i]}")
                
                # Save figure
                fig.tight_layout()
                fig.savefig(filepath, format=format_type.lower(), dpi=dpi)
                plt.close(fig)
                
                output_files.append(filepath)
                
            logger.info(f"Exported {len(output_files)} frames as {format_type} images")
            return output_files
            
        except ImportError as e:
            logger.error(f"Missing dependency for image export: {e}")
            return []
        except Exception as e:
            logger.error(f"Error exporting frames as images: {e}", exc_info=True)
            return []
    
    @classmethod
    def _ensure_extension(cls, output_path: str, format_type: str) -> str:
        """
        Ensure the output path has the correct extension for the format.
        
        Args:
            output_path: Original output path
            format_type: Format type
            
        Returns:
            Output path with correct extension
        """
        path = Path(output_path)
        extension = cls.get_file_extension(format_type)
        
        if not extension:
            return output_path
            
        if path.suffix.lower() != f".{extension}":
            path = path.with_suffix(f".{extension}")
            
        return str(path)