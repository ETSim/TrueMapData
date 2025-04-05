"""
Base triangulator module for heightmap triangulation.

This module provides an abstract base class for triangulation algorithms,
defining a common interface for different triangulation implementations.
"""

import time
import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional, Callable, Union

# Set up logging
logger = logging.getLogger(__name__)


class BaseTriangulator(ABC):
    """Abstract base class for heightmap triangulation algorithms."""
    
    def __init__(
        self, 
        height_map: np.ndarray, 
        z_scale: float = 1.0, 
        max_triangles: int = 100000, 
        error_threshold: float = 0.001,
        progress_callback: Optional[Callable[[float], None]] = None
    ):
        """
        Initialize the base triangulator.
        
        Args:
            height_map: 2D array of height values
            z_scale: Scaling factor for height values
            max_triangles: Maximum number of triangles to generate
            error_threshold: Error threshold for subdivision or simplification
            progress_callback: Optional callback function for progress reporting
        """
        self.height_map = height_map
        self.z_scale = z_scale
        self.max_triangles = max_triangles
        self.error_threshold = error_threshold
        self.progress_callback = progress_callback
        self.stats = self._init_stats()
        self.start_time = time.time()
    
    def _init_stats(self) -> Dict[str, Any]:
        """Initialize statistics dictionary."""
        return {
            "original_points": self.height_map.size,
            "max_triangles": self.max_triangles,
            "error_threshold": self.error_threshold,
            "final_triangles": 0,
            "final_vertices": 0,
            "processing_time": 0.0,
            "compression_ratio": 0.0
        }
    
    @abstractmethod
    def calculate_complexity_map(self) -> np.ndarray:
        """
        Calculate terrain complexity map to guide triangulation.
        
        Returns:
            2D array representing local terrain complexity
        """
        pass
    
    @abstractmethod
    def triangulate(self) -> Tuple[List[List[float]], List[List[int]]]:
        """
        Run triangulation algorithm.
        
        Returns:
            Tuple of (vertices, triangles) where vertices is a list of [x, y, z] coordinates
            and triangles is a list of [a, b, c] indices.
        """
        pass
    
    def finalize_stats(self, vertices, triangles) -> None:
        """Update statistics after triangulation is complete."""
        self.stats["final_triangles"] = len(triangles)
        self.stats["final_vertices"] = len(vertices)
        self.stats["processing_time"] = time.time() - self.start_time
        
        if self.height_map.size > 0:
            # Calculate compression ratio: input size / output size
            input_size = self.height_map.size
            output_size = len(vertices) + len(triangles) * 3  # Vertices + triangle indices
            self.stats["compression_ratio"] = input_size / max(1, output_size)
        
        logger.info(
            f"Triangulation complete. Generated {len(triangles)} triangles from "
            f"{self.height_map.size} height points in {self.stats['processing_time']:.2f}s. "
            f"Compression ratio: {self.stats['compression_ratio']:.2f}x"
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the triangulation.
        
        Returns:
            Dictionary with statistics
        """
        return self.stats.copy()
    
    def report_progress(self, progress: float) -> None:
        """
        Report progress to callback if provided.
        
        Args:
            progress: Progress value between 0.0 and 1.0
        """
        if self.progress_callback:
            self.progress_callback(max(0.0, min(1.0, progress)))