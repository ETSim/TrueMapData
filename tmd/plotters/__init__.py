"""
TMD Plotter Packages

This module exposes the TMD visualization components:
  - Base classes: BasePlotter, BaseSequencePlotter
  - Factory classes: TMDPlotterFactory, TMDSequencePlotterFactory
  - Concrete plotters for different backends (matplotlib, plotly, seaborn, polyscope)
  - Visualization utilities for enhanced plotting
"""

# Import and export classes in the correct order to avoid circular imports
from tmd.plotters.base import BasePlotter, BaseSequencePlotter
from tmd.plotters.factory import TMDPlotterFactory, TMDSequencePlotterFactory

# Export these functions directly from the module
from tmd.plotters.factory import get_registered_plotters, get_available_plotters, get_best_plotter

# Import concrete plotters (optional, will handle ImportError gracefully)
try:
    from tmd.plotters.matplotlib import MatplotlibHeightMapPlotter, MatplotlibSequencePlotter
except ImportError:
    pass

try:
    from tmd.plotters.plotly import PlotlyHeightMapVisualizer, PlotlySequenceVisualizer
except ImportError:
    pass

try:
    from tmd.plotters.seaborn import SeabornHeightMapPlotter, SeabornSequencePlotter, SeabornProfilePlotter
except ImportError:
    pass

try:
    from tmd.plotters.polyscope import PolyscopePlotter
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
    'get_best_plotter'
]