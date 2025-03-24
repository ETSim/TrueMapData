"""
TMD Sequence package for working with time series data and sequence operations.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union, Tuple

# Set up logging
logger = logging.getLogger(__name__)

# Import core components
from .sequence import TMDSequence
from .compare import compare_heightmaps, calculate_difference_metrics, create_comparison_visualizations
from .alignement import align_heightmaps, align_sequence_to_reference, visualize_alignment

# Import plotters - update to match the actual function names available in the module
try:
    from .plotters.plotly import (
        visualize_sequence_3d as plot_sequence_3d,
        visualize_sequence_2d as plot_sequence_2d,
        visualize_sequence_stats as plot_sequence_stats,
        create_sequence_animation
    )
except ImportError:
    # Fallback to define stub functions if the actual imports fail
    def plot_sequence_3d(*args, **kwargs): 
        raise NotImplementedError("plot_sequence_3d is not properly implemented")
    def plot_sequence_2d(*args, **kwargs): 
        raise NotImplementedError("plot_sequence_2d is not properly implemented")
    def plot_sequence_stats(*args, **kwargs): 
        raise NotImplementedError("plot_sequence_stats is not properly implemented")
    def create_sequence_animation(*args, **kwargs): 
        raise NotImplementedError("create_sequence_animation is not properly implemented")

# Import exporters
from .exporters.powerpoint import export_sequence_to_pptx, export_sequence_by_parameter
from .exporters.gif import (
    export_sequence_to_gif, 
    export_grouped_sequences_to_gifs,
    export_normal_maps_to_gif
)

__all__ = [
    'TMDSequence',
    'compare_heightmaps',
    'calculate_difference_metrics',
    'create_comparison_visualizations',
    'align_heightmaps',
    'align_sequence_to_reference',
    'visualize_alignment',
    'plot_sequence_3d',
    'plot_sequence_2d', 
    'plot_sequence_stats',
    'create_sequence_animation',
    'export_sequence_to_pptx',
    'export_sequence_by_parameter',
    'export_sequence_to_gif',
    'export_grouped_sequences_to_gifs',
    'export_normal_maps_to_gif'
]
