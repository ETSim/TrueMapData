#!/usr/bin/env python3
"""
Abstract base classes for TMD Data Exporters and Importers.

These classes define the common interface for exporting and importing TMD data.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

class TMDDataExporter(ABC):
    @abstractmethod
    def export(self, data: Dict[str, Any], output_path: str) -> str:
        """
        Export TMD data to a file.

        Args:
            data: Dictionary containing TMD data.
            output_path: Destination file path.

        Returns:
            The output path if successful.
        """
        pass

class TMDDataImporter(ABC):
    @abstractmethod
    def load(self, file_path: str) -> Dict[str, Any]:
        """
        Load TMD data from a file.

        Args:
            file_path: Path to the input file.

        Returns:
            A dictionary containing TMD data.
        """
        pass
