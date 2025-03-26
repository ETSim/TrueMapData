"""
TMD Sequence module for time series data.

This module provides the TMDSequence class for working with sequences of height maps,
such as time series data or multiple scans of the same object.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Import local modules
from ..utils.processing import threshold_height_map

# Set up logging
logger = logging.getLogger(__name__)

class TMDSequence:
    """
    Class representing a sequence of height maps.
    
    This class provides functionality for working with sequences of height maps,
    such as time series data or multiple scans of the same object.
    """
    
    def __init__(self, name: str = "Unnamed Sequence"):
        """
        Initialize a TMD sequence.
        
        Args:
            name: Name of the sequence
        """
        self.name = name
        self.frames = []
        self.frame_timestamps = []
        self.metadata = {}
        self.transformations = []
    
    def add_frame(
        self, 
        height_map: np.ndarray,
        timestamp: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        transformation: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Add a frame to the sequence.
        
        Args:
            height_map: 2D numpy array of height values
            timestamp: Timestamp or label for the frame
            metadata: Optional metadata for the frame
            transformation: Optional transformation to apply to the frame
            
        Returns:
            Index of the added frame
        """
        if height_map is None or height_map.size == 0:
            logger.warning("Attempted to add empty height map to sequence")
            return -1
            
        # Make a copy of the height map to avoid modifying the original
        frame_data = height_map.copy()
        
        # Use frame index as timestamp if none provided
        if timestamp is None:
            timestamp = f"Frame {len(self.frames) + 1}"
            
        # Use empty dict if no metadata provided
        if metadata is None:
            metadata = {}
            
        # Store the frame
        self.frames.append(frame_data)
        self.frame_timestamps.append(timestamp)
        
        # Store optional transformation
        if transformation is not None:
            while len(self.transformations) < len(self.frames):
                self.transformations.append({})
            self.transformations[-1] = transformation
        else:
            self.transformations.append({})
            
        return len(self.frames) - 1
    
    def get_frame(self, index: int) -> Optional[np.ndarray]:
        """
        Get a frame by index.
        
        Args:
            index: Index of the frame to get
            
        Returns:
            Height map for the requested frame or None if index is invalid
        """
        if 0 <= index < len(self.frames):
            return self.frames[index]
        logger.warning(f"Invalid frame index: {index}")
        return None
    
    def get_timestamp(self, index: int) -> Optional[Any]:
        """
        Get the timestamp for a frame.
        
        Args:
            index: Index of the frame
            
        Returns:
            Timestamp for the frame or None if index is invalid
        """
        if 0 <= index < len(self.frame_timestamps):
            return self.frame_timestamps[index]
        logger.warning(f"Invalid frame index: {index}")
        return None
    
    def get_all_timestamps(self) -> List[Any]:
        """
        Get all timestamps in the sequence.
        
        Returns:
            List of timestamps
        """
        return self.frame_timestamps.copy()
    
    def get_all_frames(self) -> List[np.ndarray]:
        """
        Get all frames in the sequence.
        
        Returns:
            List of height maps
        """
        return self.frames.copy()
    
    def get_transformation(self, index: int) -> Optional[Dict[str, Any]]:
        """
        Get the transformation for a frame.
        
        Args:
            index: Index of the frame
            
        Returns:
            Transformation dictionary for the frame or None if index is invalid
        """
        if 0 <= index < len(self.transformations):
            return self.transformations[index]
        logger.warning(f"Invalid frame index: {index}")
        return None
    
    def set_transformation(self, index: int, transformation: Dict[str, Any]) -> bool:
        """
        Set the transformation for a frame.
        
        Args:
            index: Index of the frame
            transformation: Transformation dictionary
            
        Returns:
            True if successful, False otherwise
        """
        if 0 <= index < len(self.frames):
            while len(self.transformations) <= index:
                self.transformations.append({})
            self.transformations[index] = transformation
            return True
        logger.warning(f"Invalid frame index: {index}")
        return False
    
    def apply_transformations(self) -> List[np.ndarray]:
        """
        Apply transformations to all frames.
        
        Returns:
            List of transformed frames
        """
        transformed_frames = []
        
        for i, frame in enumerate(self.frames):
            # Get transformation for this frame
            transform = self.get_transformation(i) or {}
            
            # Create a copy of the frame
            transformed = frame.copy()
            
            # Apply scaling if specified
            if 'scaling' in transform:
                scaling = transform['scaling']
                if len(scaling) >= 3:  # [x_scale, y_scale, z_scale]
                    # Apply z-scaling
                    transformed = transformed * scaling[2]
            
            # Apply threshold if specified
            if 'threshold' in transform:
                threshold = transform['threshold']
                if isinstance(threshold, dict):
                    min_height = threshold.get('min')
                    max_height = threshold.get('max')
                    replacement = threshold.get('replacement')
                    
                    transformed = threshold_height_map(
                        transformed, 
                        min_height=min_height,
                        max_height=max_height,
                        replacement=replacement
                    )
            
            # Add the transformed frame
            transformed_frames.append(transformed)
            
        return transformed_frames