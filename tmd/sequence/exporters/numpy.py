""".

NumPy exporter implementation for TMD sequence data.

This module provides functionality to export TMD sequence data as NumPy arrays.
"""

import os
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union

from .base import BaseExporter

logger = logging.getLogger(__name__)

class NumpyExporter(BaseExporter):
    """NumPy file exporter for TMD sequence data.."""
    
    def __init__(self):
        """Initialize the NumPy exporter.."""
        pass
    
    def export_sequence_as_npz(
        self,
        frames_data: List[np.ndarray],
        output_file: str,
        timestamps: List[Any] = None,
        metadata: Dict[str, Any] = None,
        compress: bool = True,
        **kwargs
    ) -> str:
        """.

        Export a sequence of frames as a NumPy .npz file.
        
        Args:
            frames_data: List of frame arrays to export
            output_file: Path to save the output file
            timestamps: Optional list of timestamps for each frame
            metadata: Optional dictionary of metadata to include
            compress: Whether to use compression
            **kwargs: Additional options
            
        Returns:
            Path to the saved file
        """
        # Ensure output directory exists
        if not self.ensure_output_dir(os.path.dirname(output_file)):
            return ""
            
        # Ensure the file has .npz extension
        if not output_file.lower().endswith('.npz'):
            output_file = f"{output_file}.npz"
            
        # Prepare data dictionary
        data_dict = {}
        
        # Add frames
        for i, frame in enumerate(frames_data):
            data_dict[f"frame_{i}"] = frame
            
        # Add timestamps if provided
        if timestamps is not None:
            data_dict["timestamps"] = np.array(timestamps)
            
        # Add metadata if provided
        if metadata is not None:
            # Convert metadata to arrays where possible
            for key, value in metadata.items():
                try:
                    data_dict[f"meta_{key}"] = np.array(value)
                except:
                    logger.warning(f"Could not convert metadata key '{key}' to NumPy array")
        
        # Save data
        try:
            if compress:
                np.savez_compressed(output_file, **data_dict)
            else:
                np.savez(output_file, **data_dict)
                
            logger.info(f"Sequence exported to {output_file}")
            return output_file
        except Exception as e:
            logger.error(f"Error exporting to NPZ: {e}")
            return ""
            
    def export_sequence_differences(
        self,
        frames_data: List[np.ndarray],
        output_file: str,
        timestamps: List[Any] = None,
        metadata: Dict[str, Any] = None,
        compress: bool = True,
        **kwargs
    ) -> str:
        """.

        Export differences between frames as a NumPy .npz file.
        
        Args:
            frames_data: List of difference arrays
            output_file: Path to save the output file
            timestamps: Optional list of timestamps for each frame
            metadata: Optional dictionary of metadata to include
            compress: Whether to use compression
            **kwargs: Additional options
            
        Returns:
            Path to the saved file
        """
        # This function is identical to export_sequence_as_npz for now
        # as we're just saving the difference arrays directly
        return self.export_sequence_as_npz(
            frames_data=frames_data,
            output_file=output_file,
            timestamps=timestamps,
            metadata=metadata,
            compress=compress,
            **kwargs
        )

# Create convenience functions that match the module interface pattern
def export_sequence_as_npz(
    frames_data: List[np.ndarray],
    output_file: str,
    timestamps: List[Any] = None,
    metadata: Dict[str, Any] = None,
    compress: bool = True,
    **kwargs
) -> str:
    """.

    Export a sequence of frames as a NumPy .npz file.
    
    Args:
        frames_data: List of frame arrays to export
        output_file: Path to save the output file
        timestamps: Optional list of timestamps for each frame
        metadata: Optional dictionary of metadata to include
        compress: Whether to use compression
        **kwargs: Additional options
        
    Returns:
        Path to the saved file
    """
    exporter = NumpyExporter()
    return exporter.export_sequence_as_npz(
        frames_data=frames_data,
        output_file=output_file,
        timestamps=timestamps,
        metadata=metadata,
        compress=compress,
        **kwargs
    )

def export_sequence_differences(
    frames_data: List[np.ndarray],
    output_file: str,
    timestamps: List[Any] = None,
    metadata: Dict[str, Any] = None,
    compress: bool = True,
    **kwargs
) -> str:
    """.

    Export differences between frames as a NumPy .npz file.
    
    Args:
        frames_data: List of difference arrays
        output_file: Path to save the output file
        timestamps: Optional list of timestamps for each frame
        metadata: Optional dictionary of metadata to include
        compress: Whether to use compression
        **kwargs: Additional options
        
    Returns:
        Path to the saved file
    """
    exporter = NumpyExporter()
    return exporter.export_sequence_differences(
        frames_data=frames_data,
        output_file=output_file,
        timestamps=timestamps,
        metadata=metadata,
        compress=compress,
        **kwargs
    )
