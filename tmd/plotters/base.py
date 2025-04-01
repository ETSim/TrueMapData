#!/usr/bin/env python3
"""
Base classes for TMD visualization.

This module defines the base classes that all TMD visualization backends must implement:
  - BasePlotter: Interface for plotting single height maps
  - BaseSequencePlotter: Interface for plotting sequences of height maps
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np

# Set up logger
logger = logging.getLogger(__name__)


class BasePlotter(ABC):
    """
    Base class for plotters that visualize TMD height maps.
    
    This abstract class defines the interface that all TMD visualization
    backends must implement for displaying single height maps.
    """
    
    NAME = "base"  # Should be overridden by subclasses
    DEFAULT_COLORMAP = "viridis"  # Default colormap name
    SUPPORTED_MODES = []  # List of supported plotting modes
    REQUIRED_DEPENDENCIES = []  # List of required dependencies
    
    @abstractmethod
    def plot(self, height_map: np.ndarray, **kwargs) -> Any:
        """
        Plot the TMD height map data.
        
        Args:
            height_map: 2D numpy array representing the height map
            **kwargs: Additional options specific to the backend
            
        Returns:
            A plot object specific to the backend
        """
        pass
    
    def plot_2d(self, height_map: np.ndarray, **kwargs) -> Any:
        """
        Create a 2D heatmap representation of the height map.
        
        Args:
            height_map: 2D numpy array representing the height map
            **kwargs: Additional options specific to the backend
            
        Returns:
            A plot object specific to the backend
        """
        kwargs["mode"] = "2d"
        return self.plot(height_map, **kwargs)
    
    def plot_3d(self, height_map: np.ndarray, **kwargs) -> Any:
        """
        Create a 3D representation of the height map.
        
        Args:
            height_map: 2D numpy array representing the height map
            **kwargs: Additional options specific to the backend
            
        Returns:
            A plot object specific to the backend
        """
        kwargs["mode"] = "3d"
        return self.plot(height_map, **kwargs)
    
    def plot_profile(self, height_map: np.ndarray, profile_row: int = None, **kwargs) -> Any:
        """
        Create a cross-section profile visualization.
        
        Args:
            height_map: 2D numpy array representing the height map
            profile_row: Row index to use for the profile
            **kwargs: Additional options specific to the backend
            
        Returns:
            A plot object specific to the backend
        """
        kwargs["mode"] = "profile"
        if profile_row is not None:
            kwargs["profile_row"] = profile_row
        return self.plot(height_map, **kwargs)
    
    def show(self, plot_obj: Any) -> None:
        """
        Display the plot.
        
        Args:
            plot_obj: Plot object returned by plot()
        """
        pass
    
    @abstractmethod
    def save(self, plot_obj: Any, filename: str, **kwargs) -> Optional[str]:
        """
        Save the plot to a file.
        
        Args:
            plot_obj: Plot object returned by plot()
            filename: Output filename
            **kwargs: Additional options specific to the backend
            
        Returns:
            Path to saved file if successful, None otherwise
        """
        pass


class BaseSequencePlotter(ABC):
    """
    Base class for plotters that visualize TMD height map sequences.
    
    This abstract class defines the interface that all TMD visualization
    backends must implement for displaying sequences of height maps.
    """
    
    NAME = "base"  # Should be overridden by subclasses
    DEFAULT_COLORMAP = "viridis"  # Default colormap name
    SUPPORTED_MODES = []  # List of supported modes
    REQUIRED_DEPENDENCIES = []  # List of required dependencies
    
    @abstractmethod
    def visualize_sequence(self, frames: List[np.ndarray], **kwargs) -> Any:
        """
        Visualize a sequence of TMD height maps.
        
        Args:
            frames: List of 2D numpy arrays representing the sequence
            **kwargs: Additional options specific to the backend
            
        Returns:
            A plot object specific to the backend
        """
        pass
    
    @abstractmethod
    def create_animation(self, frames: List[np.ndarray], **kwargs) -> Any:
        """
        Create an animation from a sequence of TMD height maps.
        
        Args:
            frames: List of 2D numpy arrays representing the sequence
            **kwargs: Additional options specific to the backend
            
        Returns:
            An animation object specific to the backend
        """
        pass
    
    @abstractmethod
    def visualize_statistics(self, stats_data: Dict[str, List[float]], **kwargs) -> Any:
        """
        Visualize statistical data from the sequence.
        
        Args:
            stats_data: Dictionary with metric names as keys and lists of values
            **kwargs: Additional options specific to the backend
            
        Returns:
            A plot object specific to the backend
        """
        pass
    
    @abstractmethod
    def save_figure(self, fig: Any, filename: str, **kwargs) -> Optional[str]:
        """
        Save a visualization to a file.
        
        Args:
            fig: Figure object from one of the visualization methods
            filename: Output filename
            **kwargs: Additional options specific to the backend
            
        Returns:
            Path to saved file if successful, None otherwise
        """
        pass