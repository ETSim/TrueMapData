"""
Factory classes for TMD visualization.

This module provides the factory classes for creating appropriate plotters
based on the requested visualization backend. It implements the Factory Method
pattern to decouple plotter creation from the client code.
"""

import logging
from typing import Dict, List, Type, Optional, Any, ClassVar, Union
import inspect

# Import base classes with a TYPE_CHECKING check to avoid circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from tmd.plotters.base import BasePlotter, BaseSequencePlotter

# Set up logger
logger = logging.getLogger(__name__)

class TMDPlotterFactory:
    """
    Factory class for creating plotters for TMD height maps.
    
    This class implements the Factory Method pattern to create appropriate
    plotter instances based on the requested backend (e.g., matplotlib, plotly).
    """
    
    # Class variable to store registered plotter classes
    _plotter_registry: ClassVar[Dict[str, Type['BasePlotter']]] = {}
    
    @classmethod
    def register(cls, name: str, plotter_class: Type['BasePlotter']) -> None:
        """
        Register a plotter class with the factory.
        
        Args:
            name: Name of the plotter (e.g., "matplotlib", "plotly")
            plotter_class: Plotter class to register
        """
        # Dynamic check to avoid circular imports
        from tmd.plotters.base import BasePlotter
        
        if not inspect.isclass(plotter_class) or not issubclass(plotter_class, BasePlotter):
            raise TypeError(f"Plotter class must be a subclass of BasePlotter")
        
        cls._plotter_registry[name.lower()] = plotter_class
        logger.debug(f"Registered plotter '{name}' with class {plotter_class.__name__}")
    
    @classmethod
    def create_plotter(cls, name: str) -> 'BasePlotter':
        """
        Create a plotter instance for the specified backend.
        
        Args:
            name: Name of the plotter to create (e.g., "matplotlib", "plotly")
            
        Returns:
            Instance of the requested plotter
            
        Raises:
            ValueError: If the requested plotter is not registered or available
        """
        name = name.lower()
        
        if name not in cls._plotter_registry:
            raise ValueError(f"Plotter '{name}' not registered")
        
        plotter_class = cls._plotter_registry[name]
        
        # Check if the plotter has the required dependencies
        if hasattr(plotter_class, 'REQUIRED_DEPENDENCIES'):
            try:
                # Try to instantiate the plotter
                plotter = plotter_class()
                return plotter
            except ImportError as e:
                # Handle missing dependencies
                dependencies = getattr(plotter_class, 'REQUIRED_DEPENDENCIES', [])
                raise ValueError(
                    f"Plotter '{name}' requires dependencies that are not available: {dependencies}. "
                    f"Error: {str(e)}"
                )
        else:
            # If no dependencies specified, just instantiate
            return plotter_class()
    
    @classmethod
    def list_available_strategies(cls) -> Dict[str, bool]:
        """
        List all registered plotters and their availability.
        
        Returns:
            Dictionary mapping plotter names to their availability status (bool)
        """
        available_plotters = {}
        
        for name, plotter_class in cls._plotter_registry.items():
            try:
                # Check if dependencies are available
                if hasattr(plotter_class, 'REQUIRED_DEPENDENCIES'):
                    # Try to import each dependency
                    for dependency in plotter_class.REQUIRED_DEPENDENCIES:
                        # Simple import check
                        parts = dependency.split('.')
                        __import__(parts[0])
                    available_plotters[name] = True
                else:
                    # If no dependencies specified, assume available
                    available_plotters[name] = True
            except ImportError:
                available_plotters[name] = False
        
        return available_plotters
    

class TMDSequencePlotterFactory:
    """
    Factory class for creating plotters for TMD sequences.
    
    This class implements the Factory Method pattern to create appropriate
    sequence plotter instances based on the requested backend.
    """
    
    # Class variable to store registered plotter classes
    _plotter_registry: ClassVar[Dict[str, Type['BaseSequencePlotter']]] = {}
    
    @classmethod
    def register(cls, name: str, plotter_class: Type['BaseSequencePlotter']) -> None:
        """
        Register a sequence plotter class with the factory.
        
        Args:
            name: Name of the plotter (e.g., "matplotlib", "plotly")
            plotter_class: Sequence plotter class to register
        """
        # Dynamic check to avoid circular imports
        from tmd.plotters.base import BaseSequencePlotter
        
        if not inspect.isclass(plotter_class) or not issubclass(plotter_class, BaseSequencePlotter):
            raise TypeError(f"Sequence plotter class must be a subclass of BaseSequencePlotter")
        
        cls._plotter_registry[name.lower()] = plotter_class
        logger.debug(f"Registered sequence plotter '{name}' with class {plotter_class.__name__}")
    
    @classmethod
    def create_plotter(cls, name: str) -> 'BaseSequencePlotter':
        """
        Create a sequence plotter instance for the specified backend.
        
        Args:
            name: Name of the plotter to create (e.g., "matplotlib", "plotly")
            
        Returns:
            Instance of the requested sequence plotter
            
        Raises:
            ValueError: If the requested plotter is not registered or available
        """
        name = name.lower()
        
        if name not in cls._plotter_registry:
            raise ValueError(f"Sequence plotter '{name}' not registered")
        
        plotter_class = cls._plotter_registry[name]
        
        # Check if the plotter has the required dependencies
        if hasattr(plotter_class, 'REQUIRED_DEPENDENCIES'):
            try:
                # Try to instantiate the plotter
                plotter = plotter_class()
                return plotter
            except ImportError as e:
                # Handle missing dependencies
                dependencies = getattr(plotter_class, 'REQUIRED_DEPENDENCIES', [])
                raise ValueError(
                    f"Sequence plotter '{name}' requires dependencies that are not available: {dependencies}. "
                    f"Error: {str(e)}"
                )
        else:
            # If no dependencies specified, just instantiate
            return plotter_class()
    
    @classmethod
    def list_available_strategies(cls) -> Dict[str, bool]:
        """
        List all registered sequence plotters and their availability.
        
        Returns:
            Dictionary mapping plotter names to their availability status (bool)
        """
        available_plotters = {}
        
        for name, plotter_class in cls._plotter_registry.items():
            try:
                # Check if dependencies are available
                if hasattr(plotter_class, 'REQUIRED_DEPENDENCIES'):
                    # Try to import each dependency
                    for dependency in plotter_class.REQUIRED_DEPENDENCIES:
                        # Simple import check
                        parts = dependency.split('.')
                        __import__(parts[0])
                    available_plotters[name] = True
                else:
                    # If no dependencies specified, assume available
                    available_plotters[name] = True
            except ImportError:
                available_plotters[name] = False
        
        return available_plotters


def _register_all_plotters():
    """Register all available plotters with the factories."""
    # Try to register matplotlib plotters
    try:
        from tmd.plotters.matplotlib import MatplotlibHeightMapPlotter
        TMDPlotterFactory.register("matplotlib", MatplotlibHeightMapPlotter)
    except ImportError:
        logger.debug("Matplotlib plotter not available")
    
    try:
        from tmd.plotters.matplotlib import MatplotlibSequencePlotter
        TMDSequencePlotterFactory.register("matplotlib", MatplotlibSequencePlotter)
    except ImportError:
        logger.debug("Matplotlib sequence plotter not available")
    
    # Try to register plotly plotters
    try:
        from tmd.plotters.plotly import PlotlyHeightMapVisualizer
        TMDPlotterFactory.register("plotly", PlotlyHeightMapVisualizer)
    except ImportError:
        logger.debug("Plotly plotter not available")
    
    try:
        from tmd.plotters.plotly import PlotlySequenceVisualizer
        TMDSequencePlotterFactory.register("plotly", PlotlySequenceVisualizer)
    except ImportError:
        logger.debug("Plotly sequence plotter not available")
    
    # Try to register seaborn plotters
    try:
        from tmd.plotters.seaborn import SeabornHeightMapPlotter
        TMDPlotterFactory.register("seaborn", SeabornHeightMapPlotter)
    except ImportError:
        logger.debug("Seaborn plotter not available")
    
    try:
        from tmd.plotters.seaborn import SeabornSequencePlotter
        TMDSequencePlotterFactory.register("seaborn", SeabornSequencePlotter)
    except ImportError:
        logger.debug("Seaborn sequence plotter not available")
    
    # Try to register polyscope plotters
    try:
        from tmd.plotters.polyscope import PolyscopePlotter, PolyscopeSequencePlotter
        TMDPlotterFactory.register("polyscope", PolyscopePlotter)
        TMDSequencePlotterFactory.register("polyscope", PolyscopeSequencePlotter)
    except (ImportError, SyntaxError) as e:
        logger.debug(f"Polyscope plotter not available: {e}")


# Add utility functions that can be imported directly from the module

def get_registered_plotters() -> Dict[str, bool]:
    """
    Get a dictionary of available plotters and their status.
    
    Returns:
        Dict[str, bool]: Dictionary with plotter names as keys and 
                        availability status as values
    """
    return TMDPlotterFactory.list_available_strategies()

def get_available_plotters() -> List[str]:
    """
    Get list of available plotter names.
    
    Returns:
        List[str]: Names of all available plotters
    """
    plotters = get_registered_plotters()
    return [name for name, available in plotters.items() if available]

def get_best_plotter(preference_order=None) -> Optional['BasePlotter']:
    """
    Get the best available plotter based on preference order.
    
    Args:
        preference_order: List of plotter names in order of preference
                        (default: ["plotly", "matplotlib", "polyscope", "seaborn"])
    
    Returns:
        Plotter instance or None if no plotters are available
    """
    if preference_order is None:
        preference_order = ["plotly", "matplotlib", "polyscope", "seaborn"]
    
    available = get_registered_plotters()
    
    for plotter in preference_order:
        if plotter in available and available[plotter]:
            return TMDPlotterFactory.create_plotter(plotter)
    
    return None


# Register all available plotters when the module is imported
_register_all_plotters()