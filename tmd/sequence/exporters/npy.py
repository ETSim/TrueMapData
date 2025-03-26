"""
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

"""
NumPy exporter for height map sequences.

This module provides functionality to export height map sequences as NumPy
array files (.npy) for easy loading in other Python applications.
"""

import os
import numpy as np
import logging
from typing import List, Optional, Union, Tuple

# Set up logger
logger = logging.getLogger(__name__)


def export_sequence_to_npy(
    frames: List[np.ndarray],
    output_file: str,
    compress: bool = True,
    metadata: Optional[dict] = None
) -> Optional[str]:
    """
    Export a sequence of height maps as a NumPy .npy file.
    
    Args:
        frames: List of 2D numpy arrays representing height maps
        output_file: Path to save the numpy file
        compress: Whether to use compression
        metadata: Optional metadata to include with the array
        
    Returns:
        Path to the created file or None if failed
    """
    try:
        # Check frames
        if not frames or len(frames) == 0:
            logger.error("No frames provided for NumPy export")
            return None
            
        # Ensure all frames have the same shape
        first_shape = frames[0].shape
        if not all(frame.shape == first_shape for frame in frames):
            logger.error("All frames must have the same shape for NumPy array export")
            return None
            
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        # Ensure file has .npy extension
        if not output_file.lower().endswith('.npy'):
            output_file += '.npy'
            
        # Convert list to 3D numpy array (frames, height, width)
        array = np.array(frames)
        
        # Save array
        if compress:
            np.savez_compressed(output_file, data=array, metadata=metadata or {})
            # Rename to .npy if needed (savez always adds .npz extension)
            if output_file.lower().endswith('.npy'):
                # Rename the output file
                output_file_actual = output_file + 'z'  # .npyz
                if os.path.exists(output_file_actual):
                    os.replace(output_file_actual, output_file)
        else:
            np.save(output_file, array)
        
        logger.info(f"Sequence saved to NumPy file: {output_file}")
        return output_file
        
    except Exception as e:
        logger.error(f"Error exporting to NumPy: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_sequence_from_npy(file_path: str) -> Optional[np.ndarray]:
    """
    Load a sequence of height maps from a NumPy .npy file.
    
    Args:
        file_path: Path to the numpy file
        
    Returns:
        3D numpy array of shape (frames, height, width) or None if failed
    """
    try:
        # Check file extension
        if file_path.lower().endswith('.npz'):
            # Load compressed file
            with np.load(file_path) as data:
                if 'data' in data:
                    return data['data']
                else:
                    logger.error("NPZ file does not contain 'data' array")
                    return None
        else:
            # Load uncompressed file
            return np.load(file_path)
            
    except Exception as e:
        logger.error(f"Error loading NumPy sequence: {e}")
        return None
