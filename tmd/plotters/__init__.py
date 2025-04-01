"""
TMD Plotter Packages

This module exposes the TMD visualization components:
  - Base classes: BasePlotter, BaseSequencePlotter
  - Factory classes: TMDPlotterFactory, TMDSequencePlotterFactory
  - Concrete plotters for different backends (matplotlib, plotly, seaborn, polyscope)
  - Visualization utilities for enhanced plotting
"""

# Import base classes
from tmd.plotters.base import BasePlotter, BaseSequencePlotter

# Import factory classes
from tmd.plotters.factory import (
    TMDPlotterFactory, 
    TMDSequencePlotterFactory, 
)

# Make get_registered_plotters function available directly
def get_registered_plotters():
    """
    Get a dictionary of available plotters and their status.
    
    Returns:
        Dict[str, bool]: Dictionary with plotter names as keys and 
                        availability status as values
    """
    # Create an empty result dict
    plotters = {}
    
    # Directly check for dependencies to determine availability
    # (more reliable than using class-based methods)
    
    # Check for matplotlib
    try:
        import matplotlib.pyplot as plt
        plotters["matplotlib"] = True
    except ImportError:
        plotters["matplotlib"] = False
        
    # Check for plotly
    try:
        import plotly.graph_objects as go
        plotters["plotly"] = True
    except ImportError:
        plotters["plotly"] = False
        
    # Check for seaborn (requires matplotlib)
    try:
        import seaborn as sns
        plotters["seaborn"] = plotters.get("matplotlib", False) and True
    except ImportError:
        plotters["seaborn"] = False
        
    # Check for polyscope
    try:
        import polyscope
        plotters["polyscope"] = True
    except ImportError:
        plotters["polyscope"] = False
    
    return plotters

# Get available plotters
def get_available_plotters():
    """
    Get list of available plotter names.
    
    Returns:
        List[str]: Names of all available plotters
    """
    plotters = get_registered_plotters()
    return [name for name, available in plotters.items() if available]

# Make get_best_plotter function available for auto-selection
def get_best_plotter(preference_order=None):
    """
    Get the best available plotter based on preference order.
    
    Args:
        preference_order: List of plotter names in order of preference
                        (default: ["plotly", "polyscope", "matplotlib", "seaborn"])
    
    Returns:
        Plotter instance or None if no plotters are available
    """
    if preference_order is None:
        preference_order = ["plotly", "polyscope", "matplotlib", "seaborn"]
    
    available = get_registered_plotters()
    
    for plotter in preference_order:
        if plotter in available and available[plotter]:
            return TMDPlotterFactory.create_plotter(plotter)
    
    return None

# Import built-in plotting backends when available
try:
    from tmd.plotters.matplotlib import (
        MatplotlibHeightMapPlotter, 
        MatplotlibSequencePlotter
    )
except ImportError:
    pass

try:
    from tmd.plotters.plotly import (
        PlotlyHeightMapVisualizer, 
        PlotlySequenceVisualizer
    )
except ImportError:
    pass

try:
    from tmd.plotters.polyscope import PolyscopePlotter
except ImportError:
    pass

try:
    from tmd.plotters.seaborn import (
        SeabornHeightMapPlotter, 
        SeabornSequencePlotter, 
        SeabornProfilePlotter
    )
except ImportError:
    pass

# Import visualization utilities
try:
    from tmd.plotters.visualization_utils import (
        ColorMapRegistry,
        HeightMapAnalyzer,
        TMDVisualizationUtils
    )
except ImportError:
    pass

# Define __all__ for explicit exports
__all__ = [
    'BasePlotter', 
    'BaseSequencePlotter',
    'TMDPlotterFactory', 
    'TMDSequencePlotterFactory',
    'get_registered_plotters',
    'get_available_plotters',
    'get_best_plotter',
    'ColorMapRegistry',
    'HeightMapAnalyzer',
    'TMDVisualizationUtils'
]