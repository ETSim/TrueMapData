"""
TMD Plotter Factories

This module defines two factory classes:
  - TMDPlotterFactory: Creates plotters for single TMD height maps.
  - TMDSequencePlotterFactory: Creates plotters for TMD sequences.
  
Design Patterns:
  - Factory: Instantiate the appropriate plotter based on a strategy string.
  
Usage Example:
    from tmd.plotters.factory import TMDPlotterFactory, TMDSequencePlotterFactory

    # Create a single TMD plotter using Matplotlib
    plotter = TMDPlotterFactory.create_plotter("matplotlib")
    fig = plotter.plot(height_map, title="My TMD Height Map")
    fig.show()

    # Create a sequence plotter using Matplotlib for sequences
    seq_plotter = TMDSequencePlotterFactory.create_plotter("matplotlib")
    fig_seq = seq_plotter.visualize_sequence(frames_data, n_frames=5, mode="2d")
    fig_seq.show()
"""

import logging
import importlib
from typing import Any, Optional, Union, Type, Dict, List, ClassVar

# Import base classes
from tmd.plotters.base import BasePlotter, BaseSequencePlotter, BasePlotterFactory
from tmd.utils.files import TMDFileUtilities

logger = logging.getLogger(__name__)

class PlotterFactoryBase:
    """Base class for plotter factories implementing common functionality."""
    
    # Constants for dependency mapping - to be overridden by subclasses
    STRATEGY_DEPENDENCIES: ClassVar[Dict[str, List[str]]] = {}
    STRATEGY_CLASSES: ClassVar[Dict[str, str]] = {}
    DEFAULT_STRATEGY: ClassVar[str] = "matplotlib"
    
    @classmethod
    def check_strategy_availability(cls, strategy: str) -> bool:
        """
        Check if the requested plotting strategy is available.
        
        Args:
            strategy: The strategy name to check
            
        Returns:
            Boolean indicating if all dependencies for the strategy are available
            
        Raises:
            ValueError: If the strategy is not supported
        """
        if strategy not in cls.STRATEGY_DEPENDENCIES:
            raise ValueError(f"Plotting strategy '{strategy}' not supported. "
                            f"Available options: {', '.join(cls.STRATEGY_DEPENDENCIES.keys())}")
            
        deps = cls.STRATEGY_DEPENDENCIES[strategy]
        dep_status = cls._check_dependencies(deps)
        return all(dep_status.values())
    
    @classmethod
    def _check_dependencies(cls, dependencies: List[str]) -> Dict[str, bool]:
        """Check if all dependencies are available."""
        status = {}
        for dep in dependencies:
            module = TMDFileUtilities.import_optional_dependency(dep)
            status[dep] = module is not None
        return status
    
    @classmethod
    def get_missing_dependencies(cls, strategy: str) -> List[str]:
        """
        Get a list of missing dependencies for a strategy.
        
        Args:
            strategy: The strategy name to check
            
        Returns:
            List of missing dependency names
        """
        if strategy not in cls.STRATEGY_DEPENDENCIES:
            raise ValueError(f"Plotting strategy '{strategy}' not supported. "
                            f"Available options: {', '.join(cls.STRATEGY_DEPENDENCIES.keys())}")
            
        deps = cls.STRATEGY_DEPENDENCIES[strategy]
        dep_status = cls._check_dependencies(deps)
        return [k for k, v in dep_status.items() if not v]
    
    @classmethod
    def list_available_strategies(cls) -> Dict[str, bool]:
        """
        List all available plotting strategies and their availability status.
        
        Returns:
            Dictionary with strategy names as keys and availability (bool) as values
        """
        strategies = {strategy: False for strategy in cls.STRATEGY_DEPENDENCIES}
        
        for strategy, deps in cls.STRATEGY_DEPENDENCIES.items():
            dep_status = cls._check_dependencies(deps)
            strategies[strategy] = all(dep_status.values())
            
        return strategies

    @classmethod
    def _import_class(cls, class_path: str) -> Type:
        """
        Import a class from a dotted path.
        
        Args:
            class_path: String in the format "package.module.Class"
            
        Returns:
            The imported class
            
        Raises:
            ImportError: If the class cannot be imported
        """
        try:
            module_path, class_name = class_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to import {class_path}: {e}")
            raise ImportError(f"Failed to import {class_path}: {e}") from e


class TMDPlotterFactory(PlotterFactoryBase, BasePlotterFactory):
    """
    Factory class for creating TMD plotters for single height maps.

    Strategies:
      - "matplotlib": Returns a MatplotlibTMDPlotter instance.
      - "plotly": Returns a PlotlyTMDPlotter instance.
      - "polyscope": Returns a PolyscopePlotter instance.
      - "seaborn": Returns a SeabornTMDPlotter instance.
    """
    # Define strategy dependencies and class mappings as class variables
    STRATEGY_DEPENDENCIES = {
        "matplotlib": ["matplotlib.pyplot"],
        "plotly": ["plotly", "plotly.graph_objects"],
        "polyscope": ["polyscope"],
        "seaborn": ["seaborn"]
    }
    
    STRATEGY_CLASSES = {
        "matplotlib": "tmd.plotters.matplotlib.MatplotlibHeightMapPlotter",
        "plotly": "tmd.plotters.plotly.PlotlyHeightMapVisualizer",
        "polyscope": "tmd.plotters.polyscope.PolyscopePlotter",
        "seaborn": "tmd.plotters.seaborn.SeabornHeightMapPlotter"
    }
    
    DEFAULT_STRATEGY = "matplotlib"
    
    # Initialize registry to avoid AttributeError
    _registry = {}

    @classmethod
    def create_plotter(cls, strategy: str = None) -> BasePlotter:
        """
        Create a plotter instance based on the specified strategy.
        
        Args:
            strategy: The strategy name (e.g., "matplotlib", "plotly").
                     If None, tries to use the default strategy.
        
        Returns:
            A plotter instance for the requested strategy.
            
        Raises:
            ValueError: If no strategy is available.
        """
        # If no strategy provided, use default
        if strategy is None:
            strategy = cls.DEFAULT_STRATEGY
        
        # Convert to lowercase for case-insensitive matching
        strategy = strategy.lower() if strategy else cls.DEFAULT_STRATEGY.lower()
        
        # Try direct import if matplotlib is requested
        if strategy == "matplotlib":
            try:
                import matplotlib.pyplot as plt
                # If we got here, matplotlib is available
                if "matplotlib" not in cls._registry:
                    # Try to import the specific plotter class
                    try:
                        from tmd.plotters.matplotlib import MatplotlibHeightMapPlotter
                        cls.register("matplotlib", MatplotlibHeightMapPlotter)
                    except ImportError:
                        # Fallback to direct import via class path
                        plotter_class = cls._import_class(cls.STRATEGY_CLASSES["matplotlib"])
                        cls.register("matplotlib", plotter_class)
                
                # At this point, matplotlib should be in the registry
                if "matplotlib" in cls._registry:
                    return cls._registry["matplotlib"]()
            except ImportError:
                logger.warning("Matplotlib is not available")
                # Fall through to try other strategies
        
        # Try to use the strategy from the registry first
        if strategy in cls._registry:
            plotter_class = cls._registry[strategy]
            try:
                logger.debug(f"Creating plotter from registry: {strategy}")
                return plotter_class()
            except Exception as e:
                logger.warning(f"Failed to create plotter from registry: {e}")
                # Fall through to dynamic imports if registry fails
        
        # Check which strategies are actually available
        available_strategies = {}
        for name, deps in cls.STRATEGY_DEPENDENCIES.items():
            try:
                # Try importing the first dependency as a quick check
                if deps:
                    __import__(deps[0])
                    available_strategies[name] = True
            except ImportError:
                available_strategies[name] = False
        
        # Filter to only the available ones
        truly_available = [s for s, status in available_strategies.items() if status]
        
        # If requested strategy is available, try to create it
        if strategy in truly_available:
            try:
                plotter_class = cls._import_class(cls.STRATEGY_CLASSES[strategy])
                cls.register(strategy, plotter_class)
                return plotter_class()
            except (ImportError, KeyError) as e:
                logger.error(f"Failed to create plotter for '{strategy}': {e}")
        
        # If requested strategy is not available, try to find an alternative
        if truly_available:
            alt_strategy = truly_available[0]
            logger.warning(f"Strategy '{strategy}' not available. Using '{alt_strategy}' instead.")
            try:
                plotter_class = cls._import_class(cls.STRATEGY_CLASSES[alt_strategy])
                cls.register(alt_strategy, plotter_class)
                return plotter_class()
            except (ImportError, KeyError) as e:
                logger.error(f"Failed to create plotter for '{alt_strategy}': {e}")
        
        # If all else fails, raise an informative error
        registered = list(cls._registry.keys())
        if registered:
            raise ValueError(f"Could not create plotter for '{strategy}'. "
                           f"Registered but unavailable: {', '.join(registered)}")
        else:
            raise ValueError("No plotting backends available. Please install at least one of: "
                           f"{', '.join(cls.STRATEGY_DEPENDENCIES.keys())}")


class TMDSequencePlotterFactory(PlotterFactoryBase, BasePlotterFactory):
    """
    Factory class for creating TMD sequence plotters.

    Strategies:
      - "matplotlib": Returns a MatplotlibSequencePlotter instance.
      - "plotly": Returns a PlotlySequenceVisualizer instance for sequences.
      - "polyscope": Returns a PolyscopePlotter instance configured for sequences.
      - "seaborn": Returns a SeabornSequencePlotter for sequence analysis.
    """
    # Define strategy dependencies and class mappings as class variables
    STRATEGY_DEPENDENCIES = {
        "matplotlib": ["matplotlib.pyplot", "matplotlib.animation"],
        "plotly": ["plotly", "plotly.graph_objects"],
        "polyscope": ["polyscope"],
        "seaborn": ["seaborn"]
    }
    
    STRATEGY_CLASSES = {
        "matplotlib": "tmd.plotters.matplotlib.MatplotlibSequencePlotter",
        "plotly": "tmd.plotters.plotly.PlotlySequenceVisualizer",
        "polyscope": "tmd.plotters.polyscope.PolyscopePlotter",
        "seaborn": "tmd.plotters.seaborn.SeabornSequencePlotter"
    }
    
    DEFAULT_STRATEGY = "matplotlib"
    
    # Initialize registry to avoid AttributeError
    _registry = {}

    @classmethod
    def create_plotter(cls, strategy: str = None) -> BaseSequencePlotter:
        """
        Create a sequence plotter based on the specified strategy.
        
        Args:
            strategy: The plotting library to use. Options:
                      "matplotlib" (default), "plotly", "polyscope", or "seaborn"
                      
        Returns:
            An instance of a concrete plotter implementing the BaseSequencePlotter interface
            
        Raises:
            ValueError: If the strategy is not supported
            ImportError: If the required dependencies for a strategy are not available
        """
        if strategy is None:
            strategy = cls.DEFAULT_STRATEGY
        else:
            strategy = strategy.lower()
            
        # First check if the plotter is already in the registry
        if strategy in cls._registry:
            plotter_class = cls._registry[strategy]
            
            # Special case for polyscope
            if strategy == "polyscope" and plotter_class.__name__ == "PolyscopePlotter":
                return plotter_class(is_sequence=True)
                
            return plotter_class()
            
        # Check if strategy is supported
        if strategy not in cls.STRATEGY_DEPENDENCIES:
            raise ValueError(f"Unsupported sequence plotter strategy: {strategy}. "
                             f"Available strategies: {', '.join(cls.STRATEGY_CLASSES.keys())}")
        
        # Check if the required dependencies are available
        missing = cls.get_missing_dependencies(strategy)
        if missing:
            logger.error(f"Missing dependencies for {strategy}: {', '.join(missing)}")
            raise ImportError(f"Missing dependencies for {strategy}: {', '.join(missing)}")
        
        # Import the appropriate plotter class
        try:
            # Try to get the class and register it
            plotter_class = cls._import_class(cls.STRATEGY_CLASSES[strategy])
            cls.register(strategy, plotter_class)
            
            # Special case for polyscope
            if strategy == "polyscope" and plotter_class.__name__ == "PolyscopePlotter":
                return plotter_class(is_sequence=True)
                
            return plotter_class()
        except ImportError as e:
            logger.error(f"Failed to import plotter for {strategy}: {e}")
            raise


# Register all available plotters on module import
def _register_all_plotters():
    """Register all available plotters with the factories."""
    # Try to register matplotlib plotters
    try:
        from tmd.plotters.matplotlib import MatplotlibHeightMapPlotter, MatplotlibSequencePlotter
        TMDPlotterFactory.register("matplotlib", MatplotlibHeightMapPlotter)
        TMDSequencePlotterFactory.register("matplotlib", MatplotlibSequencePlotter)
        logger.debug("Successfully registered matplotlib plotters")
    except ImportError:
        logger.debug("Matplotlib plotters not available")

    # Try to register plotly plotters
    try:
        from tmd.plotters.plotly import PlotlyHeightMapVisualizer, PlotlySequenceVisualizer
        TMDPlotterFactory.register("plotly", PlotlyHeightMapVisualizer)
        TMDSequencePlotterFactory.register("plotly", PlotlySequenceVisualizer)
        logger.debug("Successfully registered plotly plotters")
    except ImportError:
        logger.debug("Plotly plotters not available")

    # Try to register polyscope plotters
    try:
        from tmd.plotters.polyscope import PolyscopePlotter
        TMDPlotterFactory.register("polyscope", PolyscopePlotter)
        TMDSequencePlotterFactory.register("polyscope", PolyscopePlotter)
        logger.debug("Successfully registered polyscope plotters")
    except ImportError:
        logger.debug("Polyscope plotters not available")

    # Try to register seaborn plotters
    try:
        from tmd.plotters.seaborn import SeabornHeightMapPlotter, SeabornSequencePlotter
        TMDPlotterFactory.register("seaborn", SeabornHeightMapPlotter)
        TMDSequencePlotterFactory.register("seaborn", SeabornSequencePlotter)
        logger.debug("Successfully registered seaborn plotters")
    except ImportError:
        logger.debug("Seaborn plotters not available")

# Register all available plotters
_register_all_plotters()