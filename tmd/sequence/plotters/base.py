"""
Base plotter module for TMD sequence data.

This module provides the abstract base class for all sequence plotters.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np

# Set up logging
logger = logging.getLogger(__name__)

class BasePlotter(ABC):
    """
    Abstract base class for sequence plotters.
    
    All sequence plotting classes should inherit from this class and implement
    its abstract methods.
    """
    
    def __init__(self):
        """Initialize the base plotter."""
        self._has_dependencies = self.check_dependencies()
    
    def check_dependencies(self) -> bool:
        """
        Check if required dependencies are available.
        
        Returns:
            True if dependencies are available, False otherwise
        """
        # Base implementation always reports success
        # Subclasses should override this method to check their specific dependencies
        return True
    
    @abstractmethod
    def visualize_sequence(
        self,
        frames_data: List[np.ndarray],
        timestamps: Optional[List[Any]] = None,
        **kwargs
    ) -> Any:
        """
        Visualize a sequence of height maps.
        
        Args:
            frames_data: List of height map arrays
            timestamps: Optional list of timestamps or labels
            **kwargs: Additional visualization options
            
        Returns:
            Visualization object or None if visualization failed
        """
        pass
    
    @abstractmethod
    def create_animation(
        self,
        frames_data: List[np.ndarray],
        timestamps: Optional[List[Any]] = None,
        filename: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Create an animation from a sequence of height maps.
        
        Args:
            frames_data: List of height map arrays
            timestamps: Optional list of timestamps or labels
            filename: Optional filename to save the animation
            **kwargs: Additional animation options
            
        Returns:
            Animation object or None if animation creation failed
        """
        pass
    
    @abstractmethod
    def visualize_statistics(
        self,
        stats_data: Dict[str, Any],
        **kwargs
    ) -> Any:
        """
        Visualize statistics about a sequence.
        
        Args:
            stats_data: Dictionary of statistical data
            **kwargs: Additional visualization options
            
        Returns:
            Visualization object or None if visualization failed
        """
        pass
    
    def save_figure(
        self, 
        figure: Any,
        filename: str,
        **kwargs
    ) -> Optional[str]:
        """
        Save a figure to a file.
        
        Args:
            figure: Figure object to save
            filename: Output filename
            **kwargs: Additional save options
            
        Returns:
            Path to saved file or None if saving failed
        """
        # Default implementation - subclasses should override as needed
        logger.warning("save_figure not implemented in base class")
        return None

class TestingPlotter(BasePlotter):
    """
    Concrete implementation of BasePlotter for testing purposes.
    
    This class implements all abstract methods from BasePlotter to allow for testing.
    """
    
    def visualize_sequence(
        self,
        frames_data: List[np.ndarray],
        timestamps: Optional[List[Any]] = None,
        **kwargs
    ) -> Any:
        """Implementation for testing."""
        return {"frames": frames_data, "timestamps": timestamps}
    
    def create_animation(
        self,
        frames_data: List[np.ndarray],
        timestamps: Optional[List[Any]] = None,
        filename: Optional[str] = None,
        **kwargs
    ) -> Any:
        """Implementation for testing."""
        return {"frames": frames_data, "timestamps": timestamps, "filename": filename}
    
    def visualize_statistics(
        self,
        stats_data: Dict[str, Any],
        **kwargs
    ) -> Any:
        """Implementation for testing."""
        return {"stats": stats_data}
