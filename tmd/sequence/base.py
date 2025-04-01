"""
Exporters Base and Factory Module

This module defines a common interface for exporting height map sequences and
provides concrete implementations for GIF, PowerPoint (PPTX), and Video exporters.
The factory class returns an exporter instance based on the specified format.
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List

import numpy as np

# Setup module logger
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Base Exporter Interface
# ------------------------------------------------------------------------------

class BaseExporter(ABC):
    """
    Abstract base class for all exporters.
    """
    
    @abstractmethod
    def export(self, **kwargs) -> Optional[str]:
        """
        Export the height map sequence.
        
        Args:
            **kwargs: Arbitrary keyword arguments containing export options.
        
        Returns:
            The path to the exported file if successful, or None if failed.
        """
        pass