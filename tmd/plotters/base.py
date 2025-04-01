#!/usr/bin/env python3
"""
Base Plotter Abstract Classes and Factory

This module defines the base abstract classes for TMD plotters and the factory
classes for creating plotter instances.

Classes:
  - BasePlotter: Abstract base class for all TMD height map plotters.
  - BaseSequencePlotter: Abstract base class for all TMD sequence plotters.
  - BasePlotterFactory: Base factory class with registration mechanism.
  - TMDPlotterFactory: Factory for creating height map plotters.
  - TMDSequencePlotterFactory: Factory for creating sequence plotters.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Union, ClassVar

import numpy as np

logger = logging.getLogger(__name__)


class BasePlotter(ABC):
    """Abstract base class for all TMD height map plotters."""

    def __init__(self):
        """Initialize plotter."""
        pass

    @abstractmethod
    def plot(self, height_map: np.ndarray, **kwargs) -> Any:
        """
        Plot a TMD height map using the concrete plotter implementation.
        
        Args:
            height_map: 2D numpy array representing the height map.
            **kwargs: Additional options specific to the concrete implementation.
            
        Returns:
            Implementation-specific plot object.
        """
        pass

    @abstractmethod
    def save(self, plot_obj: Any, filename: str, **kwargs) -> Optional[str]:
        """
        Save a plot to a file.
        
        Args:
            plot_obj: Implementation-specific plot object.
            filename: Output filename.
            **kwargs: Additional save options specific to the implementation.
            
        Returns:
            Filename if saved successfully, None otherwise.
        """
        pass


class BaseSequencePlotter(ABC):
    """Abstract base class for all TMD sequence plotters."""

    def __init__(self):
        """Initialize sequence plotter."""
        pass

    @abstractmethod
    def visualize_sequence(self, frames: List[np.ndarray], **kwargs) -> Any:
        """
        Visualize a sequence of TMD height maps.
        
        Args:
            frames: List of 2D numpy arrays representing the sequence.
            **kwargs: Additional options specific to the concrete implementation.
            
        Returns:
            Implementation-specific visualization object.
        """
        pass

    @abstractmethod
    def create_animation(self, frames: List[np.ndarray], **kwargs) -> Any:
        """
        Create an animation from a sequence of TMD height maps.
        
        Args:
            frames: List of 2D numpy arrays representing the sequence.
            **kwargs: Additional options specific to the concrete implementation.
            
        Returns:
            Implementation-specific animation object.
        """
        pass

    @abstractmethod
    def visualize_statistics(self, stats_data: Dict[str, List[float]], **kwargs) -> Any:
        """
        Visualize statistical data from the sequence.
        
        Args:
            stats_data: Dictionary with metric names as keys and lists of values.
            **kwargs: Additional options specific to the concrete implementation.
            
        Returns:
            Implementation-specific visualization object.
        """
        pass

    @abstractmethod
    def save_figure(self, fig: Any, filename: str, **kwargs) -> Optional[str]:
        """
        Save a figure to a file.
        
        Args:
            fig: Implementation-specific figure object.
            filename: Output filename.
            **kwargs: Additional save options specific to the implementation.
            
        Returns:
            Filename if saved successfully, None otherwise.
        """
        pass


class BasePlotterFactory:
    """Base plotter factory with registration mechanism."""
    
    _registry = {}  # Will be overridden by subclasses
    
    @classmethod
    def register(cls, name: str, plotter_class: Type) -> None:
        """
        Register a plotter implementation with the factory.
        
        Args:
            name: Name to identify the plotter.
            plotter_class: Plotter class to register.
        """
        cls._registry[name.lower()] = plotter_class
        logger.debug(f"Registered {plotter_class.__name__} with key '{name.lower()}'")
    
    @classmethod
    def get_registered_plotters(cls) -> List[str]:
        """
        Get list of registered plotters.
        
        Returns:
            List of registered plotter names.
        """
        return list(cls._registry.keys())


class TMDPlotterFactory(BasePlotterFactory):
    """Factory for creating TMD height map plotters."""
    
    _registry = {}  # Separate registry for height map plotters
    
    @classmethod
    def create_plotter(cls, name: str) -> BasePlotter:
        """
        Create a plotter based on the given name.
        
        Args:
            name: Name of the registered plotter.
            
        Returns:
            Instance of the requested plotter.
            
        Raises:
            ValueError: If the requested plotter is not registered.
        """
        name = name.lower()
        registered = cls.get_registered_plotters()
        
        if name not in registered:
            raise ValueError(f"Unknown plotter: {name}. Available: {', '.join(registered)}")
        
        plotter_class = cls._registry[name]
        return plotter_class()
    
    @classmethod
    def get_available_plotters(cls) -> List[str]:
        """
        Get list of available plotters.
        
        This differs from get_registered_plotters in that it checks if
        the dependencies for each plotter are available.
        
        Returns:
            List of available plotter names.
        """
        available = []
        for name, plotter_class in cls._registry.items():
            try:
                # Try to instantiate to see if dependencies are met
                plotter_class()
                available.append(name)
            except ImportError:
                logger.debug(f"Plotter '{name}' is registered but dependencies are not met")
        
        return available


class TMDSequencePlotterFactory(BasePlotterFactory):
    """Factory for creating TMD sequence plotters."""
    
    _registry = {}  # Separate registry for sequence plotters
    
    @classmethod
    def create_plotter(cls, name: str) -> BaseSequencePlotter:
        """
        Create a sequence plotter based on the given name.
        
        Args:
            name: Name of the registered plotter.
            
        Returns:
            Instance of the requested sequence plotter.
            
        Raises:
            ValueError: If the requested plotter is not registered.
        """
        name = name.lower()
        registered = cls.get_registered_plotters()
        
        if name not in registered:
            raise ValueError(f"Unknown sequence plotter: {name}. Available: {', '.join(registered)}")
        
        plotter_class = cls._registry[name]
        return plotter_class()
    
    @classmethod
    def get_available_plotters(cls) -> List[str]:
        """
        Get list of available sequence plotters.
        
        Returns:
            List of available plotter names.
        """
        available = []
        for name, plotter_class in cls._registry.items():
            try:
                # Try to instantiate to see if dependencies are met
                plotter_class()
                available.append(name)
            except ImportError:
                logger.debug(f"Sequence plotter '{name}' is registered but dependencies are not met")
        
        return available