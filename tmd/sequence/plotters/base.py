"""Base class for sequence plotters in TMD."""

import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

class BasePlotter(ABC):
    """Base class for all sequence plotters."""
    
    def __init__(self):
        """Initialize the plotter."""
        self._has_dependencies = self._check_dependencies()
        
    def _check_dependencies(self) -> bool:
        """
        Check if required dependencies are available.
        
        Returns:
            True if all dependencies are available, False otherwise
        """
        # Base implementation always passes
        # Override in subclasses to check specific dependencies
        return True
    
    @abstractmethod
    def create_animation(self, frames_data: List[np.ndarray], **kwargs) -> Any:
        """
        Create an animation from sequence data.
        
        Args:
            frames_data: List of 2D arrays containing frame data
            **kwargs: Additional visualization options
            
        Returns:
            Animation object (implementation-specific)
        """
        pass
    
    @abstractmethod
    def visualize_sequence(self, frames_data: List[np.ndarray], **kwargs) -> Any:
        """
        Visualize sequence data.
        
        Args:
            frames_data: List of 2D arrays containing frame data
            **kwargs: Additional visualization options
            
        Returns:
            Visualization object (implementation-specific)
        """
        pass
    
    @abstractmethod
    def visualize_statistics(self, stats_data: Dict[str, List[float]], **kwargs) -> Any:
        """
        Visualize statistical data from a sequence.
        
        Args:
            stats_data: Dictionary of statistical data
            **kwargs: Additional visualization options
            
        Returns:
            Visualization object (implementation-specific)
        """
        pass
    
    def has_dependencies(self) -> bool:
        """
        Check if the plotter has all required dependencies.
        
        Returns:
            True if all dependencies are available, False otherwise
        """
        return self._has_dependencies
    
    def save_figure(self, fig: Any, filename: str, **kwargs) -> Optional[str]:
        """
        Save a figure to disk.
        
        Args:
            fig: Figure object to save
            filename: Output filename
            **kwargs: Additional saving options
            
        Returns:
            Path to saved file or None if saving failed
        """
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
            
            # Implementation needs to be provided in subclasses
            logger.warning("save_figure method not implemented in subclass")
            return None
            
        except Exception as e:
            logger.error(f"Error saving figure: {e}")
            return None


class TestingPlotter(BasePlotter):
    """
    Concrete implementation of BasePlotter for testing purposes.
    
    This class implements all abstract methods from BasePlotter to allow for testing.
    """
    
    def visualize_sequence(
        self,
        frames_data: List[np.ndarray],
        **kwargs
    ) -> Any:
        """Implementation for testing."""
        return {"frames": frames_data}
    
    def create_animation(
        self,
        frames_data: List[np.ndarray],
        **kwargs
    ) -> Any:
        """Implementation for testing."""
        return {"frames": frames_data}
    
    def visualize_statistics(
        self,
        stats_data: Dict[str, List[float]],
        **kwargs
    ) -> Any:
        """Implementation for testing."""
        return {"stats": stats_data}
