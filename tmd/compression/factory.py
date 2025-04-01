#!/usr/bin/env python3
"""
TMD Data I/O Factory

This module provides a factory to get the appropriate exporter or importer
for TMD data based on a format string.
"""

from typing import Any, Dict, Type
from .base import TMDDataExporter, TMDDataImporter
from .npz import NPZExporter, NPZImporter
from .pickle import PickleExporter, PickleImporter
from .npy import NPYExporter, NPYImporter
from .zip import ZIPExporter, ZIPImporter

class TMDDataIOFactory:
    """Factory for creating TMD data exporters and importers."""
    
    # Class dictionaries for registered exporters and importers
    _exporters: Dict[str, Type[TMDDataExporter]] = {
        'npz': NPZExporter,
        'pickle': PickleExporter,
        'npy': NPYExporter,
        'zip': ZIPExporter
    }
    
    _importers: Dict[str, Type[TMDDataImporter]] = {
        'npz': NPZImporter,
        'pickle': PickleImporter,
        'npy': NPYImporter,
        'zip': ZIPImporter
    }
    
    @classmethod
    def register_exporter(cls, format_type: str, exporter_class: Type[TMDDataExporter]) -> None:
        """
        Register a new exporter class.
        
        Args:
            format_type: Format type identifier
            exporter_class: Exporter class to register
        """
        cls._exporters[format_type.lower()] = exporter_class
    
    @classmethod
    def register_importer(cls, format_type: str, importer_class: Type[TMDDataImporter]) -> None:
        """
        Register a new importer class.
        
        Args:
            format_type: Format type identifier
            importer_class: Importer class to register
        """
        cls._importers[format_type.lower()] = importer_class
    
    @classmethod
    def get_exporter(cls, format_type: str, **kwargs) -> TMDDataExporter:
        """
        Get an exporter for the specified format.

        Args:
            format_type: Format type as a string.
            **kwargs: Additional parameters (e.g., compress for NPZ).

        Returns:
            An instance of TMDDataExporter.
            
        Raises:
            ValueError: If format_type is not supported
        """
        format_type = format_type.lower()
        exporter_class = cls._exporters.get(format_type)
        
        if not exporter_class:
            supported = ", ".join(cls._exporters.keys())
            raise ValueError(f"Unsupported export format: {format_type}. "
                           f"Supported formats: {supported}")
        
        return exporter_class(**kwargs)

    @classmethod
    def get_importer(cls, format_type: str) -> TMDDataImporter:
        """
        Get an importer for the specified format.

        Args:
            format_type: Format type as a string.

        Returns:
            An instance of TMDDataImporter.
            
        Raises:
            ValueError: If format_type is not supported
        """
        format_type = format_type.lower()
        importer_class = cls._importers.get(format_type)
        
        if not importer_class:
            supported = ", ".join(cls._importers.keys())
            raise ValueError(f"Unsupported import format: {format_type}. "
                           f"Supported formats: {supported}")
        
        return importer_class()
    
    @classmethod
    def supported_export_formats(cls) -> list:
        """Get a list of supported export formats."""
        return list(cls._exporters.keys())
    
    @classmethod
    def supported_import_formats(cls) -> list:
        """Get a list of supported import formats."""
        return list(cls._importers.keys())
