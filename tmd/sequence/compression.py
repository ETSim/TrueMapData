"""
Compression Strategy for TMD Sequences

This module provides strategies for compressing sequence data using different formats
(NPZ, Pickle, NPY) and a factory for creating the appropriate strategy.
"""

import os
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Optional, Type, Union

import numpy as np

from tmd.compression.factory import TMDDataIOFactory
from tmd.compression.base import TMDDataExporter, TMDDataImporter

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Compression Strategy Interface
# ------------------------------------------------------------------------------

class CompressionStrategy(ABC):
    """Abstract base class for all compression strategies."""
    
    @abstractmethod
    def get_exporter(self, **kwargs) -> TMDDataExporter:
        """Get the exporter for this compression strategy."""
        pass
    
    @abstractmethod
    def get_importer(self) -> TMDDataImporter:
        """Get the importer for this compression strategy."""
        pass
    
    def compress(self, data: Dict[str, Any], output_path: str, **kwargs) -> str:
        """
        Compress the provided data and save to output_path.
        
        Args:
            data: Dictionary containing data to compress
            output_path: Path where to save the compressed data
            **kwargs: Additional compression options
            
        Returns:
            Path to the compressed file
        """
        exporter = self.get_exporter(**kwargs)
        return exporter.export(data, output_path)
    
    def decompress(self, input_path: str, **kwargs) -> Dict[str, Any]:
        """
        Decompress data from the specified input path.
        
        Args:
            input_path: Path to the compressed file
            **kwargs: Additional decompression options
            
        Returns:
            Dictionary containing decompressed data
        """
        importer = self.get_importer()
        return importer.load(input_path)

# ------------------------------------------------------------------------------
# Concrete Compression Strategies
# ------------------------------------------------------------------------------

class NPZCompressionStrategy(CompressionStrategy):
    """Compression strategy using NumPy's NPZ format."""
    
    def __init__(self, compress: bool = True):
        self.compress = compress
        
    def get_exporter(self, **kwargs) -> TMDDataExporter:
        return TMDDataIOFactory.get_exporter('npz', compress=self.compress)
        
    def get_importer(self) -> TMDDataImporter:
        return TMDDataIOFactory.get_importer('npz')


class PickleCompressionStrategy(CompressionStrategy):
    """Compression strategy using Python's pickle format."""
    
    def get_exporter(self, **kwargs) -> TMDDataExporter:
        return TMDDataIOFactory.get_exporter('pickle')
        
    def get_importer(self) -> TMDDataImporter:
        return TMDDataIOFactory.get_importer('pickle')


class NPYCompressionStrategy(CompressionStrategy):
    """Compression strategy using NumPy's NPY format (for single arrays only)."""
    
    def get_exporter(self, **kwargs) -> TMDDataExporter:
        return TMDDataIOFactory.get_exporter('npy')
        
    def get_importer(self) -> TMDDataImporter:
        return TMDDataIOFactory.get_importer('npy')

# ------------------------------------------------------------------------------
# Compression Strategy Factory
# ------------------------------------------------------------------------------

class CompressionStrategyFactory:
    """Factory for creating compression strategies."""
    
    _strategies: Dict[str, Type[CompressionStrategy]] = {
        'npz': NPZCompressionStrategy,
        'pickle': PickleCompressionStrategy,
        'npy': NPYCompressionStrategy,
    }
    
    @classmethod
    def get_strategy(cls, format_type: str, **kwargs) -> CompressionStrategy:
        """
        Get a compression strategy for the specified format.
        
        Args:
            format_type: The format type (npz, pickle, npy)
            **kwargs: Additional options passed to the strategy constructor
            
        Returns:
            An instance of CompressionStrategy
            
        Raises:
            ValueError: If format_type is not supported
        """
        format_type = format_type.lower()
        strategy_class = cls._strategies.get(format_type)
        
        if not strategy_class:
            supported = ", ".join(cls._strategies.keys())
            raise ValueError(f"Unsupported compression format '{format_type}'. "
                            f"Supported formats: {supported}")
            
        return strategy_class(**kwargs)
    
    @classmethod
    def register_strategy(cls, format_type: str, strategy_class: Type[CompressionStrategy]) -> None:
        """
        Register a new compression strategy.
        
        Args:
            format_type: The format identifier
            strategy_class: The strategy class to register
        """
        cls._strategies[format_type.lower()] = strategy_class
        logger.debug(f"Registered compression strategy for format: {format_type}")
        
    @classmethod
    def supported_formats(cls) -> List[str]:
        """Get a list of supported compression formats."""
        return list(cls._strategies.keys())


# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------

def compress_sequence(
    sequence_data: Dict[str, Any], 
    output_path: str, 
    format_type: str = 'npz', 
    **kwargs
) -> str:
    """
    Compress sequence data using the specified format.
    
    Args:
        sequence_data: Dictionary containing sequence data
        output_path: Where to save the compressed data
        format_type: Format to use (npz, pickle, npy)
        **kwargs: Additional format-specific options
        
    Returns:
        Path to the compressed file
    """
    try:
        strategy = CompressionStrategyFactory.get_strategy(format_type, **kwargs)
        result = strategy.compress(sequence_data, output_path)
        logger.info(f"Sequence compressed to {result} using {format_type} format")
        return result
    except Exception as e:
        logger.error(f"Error compressing sequence data: {e}", exc_info=True)
        raise


def decompress_sequence(
    input_path: str, 
    format_type: Optional[str] = None, 
    **kwargs
) -> Dict[str, Any]:
    """
    Decompress sequence data from a file.
    
    Args:
        input_path: Path to the compressed file
        format_type: Format to use (npz, pickle, npy), inferred from extension if None
        **kwargs: Additional format-specific options
        
    Returns:
        Dictionary containing sequence data
    """
    if format_type is None:
        # Infer format from file extension
        ext = Path(input_path).suffix.lower()[1:]  # Remove the leading dot
        if ext in CompressionStrategyFactory.supported_formats():
            format_type = ext
        else:
            raise ValueError(f"Could not infer compression format from extension '{ext}'")
    
    try:
        strategy = CompressionStrategyFactory.get_strategy(format_type, **kwargs)
        result = strategy.decompress(input_path)
        logger.info(f"Sequence decompressed from {input_path} using {format_type} format")
        return result
    except Exception as e:
        logger.error(f"Error decompressing sequence data: {e}", exc_info=True)
        raise


def get_appropriate_strategy(file_path: str) -> CompressionStrategy:
    """
    Get the appropriate compression strategy based on file extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        An instance of CompressionStrategy
    """
    ext = Path(file_path).suffix.lower()[1:]  # Remove the leading dot
    return CompressionStrategyFactory.get_strategy(ext)
