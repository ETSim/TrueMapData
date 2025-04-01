"""
TMD class for working with True Map Data files.

This module provides a high-level interface for processing, visualizing,
and exporting TMD files.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path

import numpy as np

# Import main classes and functions from tmd.py
from tmd.core.tmd import TMD, TMDProcessor, TMDProcessingError, load, get_registered_plotters

# Set up logging
logger = logging.getLogger(__name__)

# Re-export the necessary items for package-level access
__all__ = [
    'TMD',
    'TMDProcessor', 
    'TMDProcessingError', 
    'load',
    'get_registered_plotters',
]

# Additional helper functions at package level can be added here
def list_available_plotters() -> List[str]:
    """
    Get a list of all available plotting backends.
    
    Returns:
        List of available plotting backend names.
    """
    return get_registered_plotters()['available']