"""
Base generator module that defines the interface for all map generators.

This module provides the abstract base class that all map generators must implement.
"""
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

import numpy as np

from ..core.base_types import MapParams
from ..core.image_utils import prepare_height_map

logger = logging.getLogger(__name__)

class MapGenerator(ABC):
    """
    Abstract base class for all map generators.
    
    Each concrete map generator should implement the generate() method
    to produce its specific type of map from a height map.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the map generator with default parameters.
        
        Args:
            **kwargs: Default parameters for this generator
        """
        self.default_params = kwargs
    
    @abstractmethod
    def generate(self, height_map: np.ndarray, **kwargs) -> np.ndarray:
        """
        Generate a specific type of map from a height map.
        
        Args:
            height_map: Input height map as numpy array
            **kwargs: Parameters for map generation
            
        Returns:
            Generated map as numpy array
        """
        pass
    
    def _prepare_height_map(self, height_map: np.ndarray, **kwargs) -> np.ndarray:
        """
        Prepare the height map for processing.
        
        Args:
            height_map: Input height map
            **kwargs: Preparation parameters
            
        Returns:
            Prepared height map
        """
        # Use core utility function to prepare height map
        return prepare_height_map(height_map, **kwargs)
    
    def _get_params(self, **kwargs) -> MapParams:
        """
        Get the effective parameters by merging defaults with provided kwargs.
        
        Args:
            **kwargs: Parameters provided for this generation
            
        Returns:
            Merged parameters
        """
        params = self.default_params.copy()
        params.update(kwargs)
        return self._validate_params(params)
    
    def _validate_params(self, params: MapParams) -> MapParams:
        """
        Validate and adjust parameters as needed.
        
        Override this method in subclasses to perform specific validation.
        
        Args:
            params: Parameters to validate
            
        Returns:
            Validated parameters
        """
        return params
