"""
TMDSequence: Core class for managing sequences of height maps.

This module defines the TMDSequence class that supports adding frames,
applying transformations, computing statistics, and exporting the sequence
to various formats using a centralized factory-based approach.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

import numpy as np

from tmd.core.tmd import TMD, TMDProcessor, TMDProcessingError
from tmd.utils.files import TMDFileUtilities
from tmd.surface.processing import threshold_height_map
from tmd.sequence.factory import SequenceExporterFactory
from tmd.exporters.TMDDataIOFactory import TMDDataIOFactory

logger = logging.getLogger(__name__)

class TMDSequence:
    """
    Class representing a sequence of TMD files.

    Provides methods for adding frames (from TMD files or arrays),
    managing timestamps and transformations, computing statistics, and exporting
    the sequence to different file formats using a factory-based approach.
    """

    def __init__(self, name: str = "Unnamed Sequence"):
        """
        Initialize a new TMDSequence.
        
        Args:
            name: Name of the sequence for identification
        """
        self.name = name
        self.frames: List[np.ndarray] = []
        self.frame_timestamps: List[Any] = []
        self.metadata: Dict[str, Any] = {}
        self.transformations: Dict[int, Dict[str, Any]] = {}
        self.frame_metadata: List[Dict[str, Any]] = []
        self.tmd_objects: List[Optional[TMD]] = []  # Store TMD objects if available

    def add_frame(
        self,
        height_map: np.ndarray,
        timestamp: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        transformation: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Add a single frame to the sequence.
        
        Args:
            height_map: 2D numpy array with height map data
            timestamp: Timestamp identifier for the frame
            metadata: Associated metadata dictionary
            transformation: Dictionary of transformations to apply
            
        Returns:
            Index of the added frame, or -1 if failed
        """
        if height_map is None or height_map.size == 0:
            logger.warning("Attempted to add empty height map to sequence")
            return -1

        frame_data = height_map.copy()
        if timestamp is None:
            timestamp = f"Frame {len(self.frames) + 1}"
        if metadata is None:
            metadata = {}

        self.frames.append(frame_data)
        self.frame_timestamps.append(timestamp)
        self.frame_metadata.append(metadata)
        self.transformations[len(self.frames) - 1] = transformation if transformation else {}
        self.tmd_objects.append(None)  # No TMD object for raw frame
        return len(self.frames) - 1

    def add_tmd_file(self, filepath: Union[str, Path], timestamp: Any = None) -> int:
        """
        Add a single TMD file to the sequence.
        
        Args:
            filepath: Path to the TMD file
            timestamp: Optional timestamp (defaults to filename)
            
        Returns:
            Index of the added frame, or -1 if failed
        """
        try:
            # Use the TMD class for better integration
            tmd_obj = TMD(filepath)
            
            if timestamp is None:
                timestamp = Path(filepath).stem
                
            # Get the height map and metadata from the TMD object
            height_map = tmd_obj.height_map()
            metadata = tmd_obj.metadata()
            
            # Add the frame
            frame_idx = self.add_frame(height_map, timestamp, metadata)
            
            # Store the TMD object reference
            if frame_idx >= 0:
                self.tmd_objects[frame_idx] = tmd_obj
                logger.info(f"Added TMD file '{filepath}' as frame {frame_idx}")
            
            return frame_idx
            
        except (FileNotFoundError, TMDProcessingError) as e:
            logger.error(f"Error adding TMD file '{filepath}': {str(e)}")
            return -1
        except Exception as e:
            logger.error(f"Unexpected error adding TMD file '{filepath}': {str(e)}")
            return -1

    def add_frames_from_folder(
        self, 
        folder_path: Union[str, Path], 
        extension: str = "tmd",
        sort_method: str = "name",
        recursive: bool = True
    ) -> int:
        """
        Add all TMD files from a folder to the sequence.
        
        Args:
            folder_path: Path to the folder containing TMD files
            extension: File extension to match (default: "tmd")
            sort_method: How to sort files ("name", "time", "none")
            recursive: Whether to search subdirectories
            
        Returns:
            Number of successfully added frames
        """
        try:
            # Get list of all files with matching extension
            file_list = TMDFileUtilities.list_files_with_extension(
                str(folder_path), extension, recursive=recursive
            )
            
            if not file_list:
                logger.warning(f"No files with extension '.{extension}' found in {folder_path}")
                return 0
                
            # Sort files if requested
            if sort_method.lower() == "name":
                file_list.sort()
            elif sort_method.lower() == "time":
                file_list.sort(key=lambda x: Path(x).stat().st_mtime)
                
            # Add each file to the sequence
            count = 0
            for filepath in file_list:
                result = self.add_tmd_file(filepath)
                if result >= 0:
                    count += 1
                    
            logger.info(f"Added {count} frames from folder {folder_path}")
            return count
            
        except Exception as e:
            logger.error(f"Error adding frames from folder '{folder_path}': {str(e)}")
            return 0

    def get_frame(self, index: int) -> Optional[np.ndarray]:
        """Get a specific frame by index."""
        if 0 <= index < len(self.frames):
            return self.frames[index]
        logger.warning(f"Invalid frame index: {index}")
        return None

    def get_frame_count(self) -> int:
        """Get the total number of frames in the sequence."""
        return len(self.frames)

    def get_timestamp(self, index: int) -> Optional[Any]:
        """Get the timestamp for a specific frame."""
        if 0 <= index < len(self.frame_timestamps):
            return self.frame_timestamps[index]
        logger.warning(f"Invalid frame index: {index}")
        return None

    def get_all_timestamps(self) -> List[Any]:
        """Get all frame timestamps."""
        return self.frame_timestamps.copy()

    def get_all_frames(self) -> List[np.ndarray]:
        """Get all frames in the sequence."""
        return self.frames.copy()

    def get_frame_metadata(self, index: int) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific frame."""
        if 0 <= index < len(self.frame_metadata):
            return self.frame_metadata[index]
        logger.warning(f"Invalid frame index: {index}")
        return None

    def get_tmd_object(self, index: int) -> Optional[TMD]:
        """Get the original TMD object for a frame if available."""
        if 0 <= index < len(self.tmd_objects):
            return self.tmd_objects[index]
        return None

    def get_transformation(self, index: int) -> Optional[Dict[str, Any]]:
        """Get transformation parameters for a specific frame."""
        return self.transformations.get(index, {})

    def set_transformation(self, index: int, transformation: Dict[str, Any]) -> bool:
        """Set transformation parameters for a specific frame."""
        if 0 <= index < len(self.frames):
            self.transformations[index] = transformation
            return True
        logger.warning(f"Invalid frame index: {index}")
        return False

    def apply_transformations(self) -> List[np.ndarray]:
        """
        Apply all defined transformations to frames.
        
        Returns:
            List of transformed frame arrays
        """
        transformed_frames = []
        for i, frame in enumerate(self.frames):
            transform = self.get_transformation(i) or {}
            transformed = frame.copy()
            
            # Apply scaling if specified
            if 'scaling' in transform:
                scaling = transform['scaling']
                if isinstance(scaling, (list, tuple)) and len(scaling) >= 3:
                    transformed = transformed * scaling[2]  # Z-scale
                elif isinstance(scaling, (int, float)):
                    transformed = transformed * scaling
                    
            # Apply thresholding if specified
            if 'threshold' in transform:
                threshold = transform['threshold']
                if isinstance(threshold, dict):
                    transformed = threshold_height_map(
                        transformed,
                        min_height=threshold.get('min'),
                        max_height=threshold.get('max'),
                        replacement=threshold.get('replacement'),
                    )
                    
            # Apply offset if specified
            if 'offset' in transform:
                offset = transform['offset']
                if isinstance(offset, (int, float)):
                    transformed = transformed + offset
                    
            transformed_frames.append(transformed)
        return transformed_frames

    def calculate_statistics(self) -> Dict[str, List[Any]]:
        """
        Calculate statistics for all frames in the sequence.
        
        Returns:
            Dictionary of statistical measures across all frames
        """
        stats = {
            'timestamps': self.frame_timestamps.copy(),
            'min': [],
            'max': [],
            'mean': [],
            'median': [],
            'std': [],
            'range': [],
            'sum': [],
            'valid_pixels': []
        }
        
        transformed_frames = self.apply_transformations()
        for frame in transformed_frames:
            # Handle NaN values appropriately
            valid_mask = ~np.isnan(frame)
            valid_data = frame[valid_mask]
            
            if valid_data.size > 0:
                stats['min'].append(float(np.min(valid_data)))
                stats['max'].append(float(np.max(valid_data)))
                stats['mean'].append(float(np.mean(valid_data)))
                stats['median'].append(float(np.median(valid_data)))
                stats['std'].append(float(np.std(valid_data)))
                stats['range'].append(float(np.max(valid_data) - np.min(valid_data)))
                stats['sum'].append(float(np.sum(valid_data)))
                stats['valid_pixels'].append(int(np.sum(valid_mask)))
            else:
                # Handle empty/all-NaN frames
                stats['min'].append(float('nan'))
                stats['max'].append(float('nan'))
                stats['mean'].append(float('nan'))
                stats['median'].append(float('nan'))
                stats['std'].append(float('nan'))
                stats['range'].append(float('nan'))
                stats['sum'].append(float('nan'))
                stats['valid_pixels'].append(0)
                
        return stats

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the sequence into a dictionary representation suitable for export.
        
        Returns:
            Dictionary containing all sequence data
        """
        return {
            "name": self.name,
            "metadata": self.metadata,
            "frames": self.frames,
            "timestamps": self.frame_timestamps,
            "frame_metadata": self.frame_metadata,
            "transformations": self.transformations,
        }

    # -------------------------------------------------------------------------
    # Simplified Export Methods using the Centralized Factory
    # -------------------------------------------------------------------------
    
    def export(self, output_path: str, format_type: str, **kwargs) -> Optional[str]:
        """
        Generic export method using the SequenceExporterFactory.
        
        Args:
            output_path: Path to the output file
            format_type: Export format type ('gif', 'video', 'powerpoint', etc.)
            **kwargs: Format-specific export options
            
        Returns:
            Path to the exported file if successful, None otherwise
        """
        # Apply transformations to get the frames to export
        frames = self.apply_transformations()
        
        if not frames:
            logger.error("No frames available to export")
            return None
            
        # Add sequence metadata if not provided
        if 'title' not in kwargs and format_type.lower() in ['powerpoint', 'pptx']:
            kwargs['title'] = self.name
            
        # Add timestamps if available and not provided
        if 'timestamps' not in kwargs and self.frame_timestamps:
            kwargs['timestamps'] = self.frame_timestamps
            
        # Use the factory to perform the export
        return SequenceExporterFactory.export_sequence(
            frames, output_path, format_type, **kwargs
        )
    
    def export_to_gif(self, output_path: str, fps: float = 10.0, **kwargs) -> Optional[str]:
        """
        Export the sequence to an animated GIF.
        
        Args:
            output_path: Path for the output GIF file
            fps: Frames per second (default: 10.0)
            **kwargs: Additional export options
            
        Returns:
            Path to the exported GIF if successful, None otherwise
        """
        # Use the factory's specialized method
        frames = self.apply_transformations()
        return SequenceExporterFactory.export_gif(frames, output_path, fps, **kwargs)
        
    def export_to_video(self, output_path: str, fps: float = 30.0, **kwargs) -> Optional[str]:
        """
        Export the sequence to a video file (MP4).
        
        Args:
            output_path: Path for the output video file
            fps: Frames per second (default: 30.0)
            **kwargs: Additional export options
            
        Returns:
            Path to the exported video if successful, None otherwise
        """
        # Use the factory's specialized method
        frames = self.apply_transformations()
        return SequenceExporterFactory.export_video(frames, output_path, fps, **kwargs)
        
    def export_to_powerpoint(self, output_path: str, **kwargs) -> Optional[str]:
        """
        Export the sequence to a PowerPoint presentation.
        
        Args:
            output_path: Path for the output PPTX file
            **kwargs: Additional export options
            
        Returns:
            Path to the exported presentation if successful, None otherwise
        """
        # Use the factory's specialized method
        frames = self.apply_transformations()
        
        # Add sequence name as title if not provided
        if 'title' not in kwargs:
            kwargs['title'] = self.name
            
        return SequenceExporterFactory.export_powerpoint(frames, output_path, **kwargs)
    
    def export_frames_as_images(self, 
                               output_dir: str, 
                               format_type: str = 'png', 
                               **kwargs) -> List[str]:
        """
        Export individual frames as separate image files.
        
        Args:
            output_dir: Directory where images should be saved
            format_type: Image format ('png', 'jpg', 'tif', etc.)
            **kwargs: Additional export options
            
        Returns:
            List of paths to saved image files
        """
        frames = self.apply_transformations()
        
        # Add timestamps if available and not provided
        if 'timestamps' not in kwargs and self.frame_timestamps:
            kwargs['timestamps'] = self.frame_timestamps
            
        # Use base filename from sequence name if not provided
        if 'base_filename' not in kwargs:
            kwargs['base_filename'] = self.name.replace(' ', '_').lower()
            
        return SequenceExporterFactory.export_frames_as_images(
            frames, output_dir, format_type, **kwargs
        )
    
    def get_supported_export_formats(self) -> List[str]:
        """
        Get a list of supported export formats.
        
        Returns:
            List of supported format names
        """
        return SequenceExporterFactory.supported_formats()
    
    # -------------------------------------------------------------------------
    # Data Storage Methods
    # -------------------------------------------------------------------------
    
    def save_to_npz(self, filepath: str) -> bool:
        """
        Save the sequence to a compressed NPZ file.
        
        Args:
            filepath: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert to dictionary
            data = self.to_dict()
            
            # NPZ doesn't handle dictionaries directly, so convert each frame
            for i, frame in enumerate(data['frames']):
                data[f'frame_{i}'] = frame
                
            # Remove the frames list to avoid duplication
            frames_data = data.pop('frames')
            
            # Save to NPZ file
            np.savez_compressed(filepath, **data)
            logger.info(f"Sequence saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving sequence: {e}")
            return False

    @classmethod
    def load_from_npz(cls, filepath: str) -> Optional['TMDSequence']:
        """
        Load a sequence from a compressed NPZ file.
        
        Args:
            filepath: Path to the NPZ file
            
        Returns:
            TMDSequence object or None if loading failed
        """
        try:
            # Load the NPZ file
            data = np.load(filepath, allow_pickle=True)
            
            # Create a new sequence with the saved name
            sequence = cls(name=str(data['name']))
            
            # Load metadata if available
            if 'metadata' in data:
                sequence.metadata = data['metadata'].item() if data['metadata'].dtype == np.object_ else {}
                
            # Get timestamps and transformations
            timestamps = data['timestamps'] if 'timestamps' in data else []
            transformations = data['transformations'].item() if 'transformations' in data else {}
            
            # Get frame metadata if available
            frame_metadata = data['frame_metadata'] if 'frame_metadata' in data else []
            
            # Find all frame keys
            frame_keys = [k for k in data.keys() if k.startswith('frame_')]
            
            # Add each frame to the sequence
            for key in sorted(frame_keys, key=lambda k: int(k.split('_')[1])):
                idx = int(key.split('_')[1])
                
                # Get timestamp for this frame
                timestamp = timestamps[idx] if idx < len(timestamps) else None
                
                # Get transformation for this frame
                transform = transformations.get(str(idx), {}) if isinstance(transformations, dict) else {}
                
                # Get metadata for this frame
                metadata = frame_metadata[idx] if idx < len(frame_metadata) else {}
                
                # Add the frame to the sequence
                sequence.add_frame(data[key], timestamp=timestamp, 
                                   metadata=metadata, transformation=transform)
                
            logger.info(f"Sequence loaded from {filepath} with {len(frame_keys)} frames")
            return sequence
            
        except Exception as e:
            logger.error(f"Error loading sequence from NPZ: {e}")
            return None
            
    def __str__(self) -> str:
        """Return a string representation of the TMDSequence."""
        return f"TMDSequence('{self.name}', {len(self.frames)} frames)"
    
    def __repr__(self) -> str:
        """Return a detailed string representation for debugging."""
        return f"TMDSequence(name='{self.name}', frames={len(self.frames)}, metadata_keys={list(self.metadata.keys