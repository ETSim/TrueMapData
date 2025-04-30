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


def to_16bit_grayscale(self, height_map: np.ndarray) -> np.ndarray:
        """
        Convert a heightmap to 16-bit grayscale format.
        
        Args:
            height_map: Input heightmap array
        Returns:
            16-bit normalized heightmap
        """
        
        # Ensure floating point for calculations
        height_map = height_map.astype(np.float32)
        
        # Normalize to [0, 1] range
        min_val = np.min(height_map)
        max_val = np.max(height_map)
        height_range = max_val - min_val
        
        if height_range > 0:
            height_map = (height_map - min_val) / height_range
        else:
            height_map = np.zeros_like(height_map)
        
        # Convert to 16-bit integer range [0, 65535]
        height_map = (height_map * 65535).astype(np.uint16)
        
        # Convert back to float32 but preserve 16-bit precision
        height_map = height_map.astype(np.float32) / 65535.0
        
        logger.debug(f"Converted heightmap: shape={height_map.shape}, dtype={height_map.dtype}, range=[{height_map.min():.3f}, {height_map.max():.3f}]")
        
        return height_map


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
        """Initialize the base triangulator."""
        # Convert heightmap to 16-bit grayscale
        self.height_map = to_16bit_grayscale(self, height_map)
        
        # Ensure heightmap is in correct format
        self.height_map = self._validate_and_convert_heightmap(height_map)
        self.z_scale = z_scale
        self.max_triangles = max_triangles
        self.error_threshold = error_threshold
        self.progress_callback = progress_callback
        self.stats = self._init_stats()
        self.start_time = time.time()
    
    def _validate_and_convert_heightmap(self, height_map: np.ndarray) -> np.ndarray:
        """
        Validate and ensure heightmap is in correct format for triangulation.
        
        Args:
            height_map: Input heightmap array
            
        Returns:
            Validated and converted heightmap
        """
        if height_map.dtype == np.uint16:
            # Already in 16-bit format, just normalize to float32 [0,1]
            return height_map.astype(np.float32) / 65535.0
            
        # Convert to 16-bit precision while maintaining [0,1] range
        height_map = height_map.astype(np.float32)
        min_val = np.min(height_map)
        max_val = np.max(height_map)
        height_range = max_val - min_val
        
        if height_range > 0:
            height_map = (height_map - min_val) / height_range
            
        # Convert through 16-bit to ensure consistent precision
        height_map = (height_map * 65535).astype(np.uint16).astype(np.float32) / 65535.0
        
        return height_map
    
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