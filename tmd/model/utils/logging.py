"""Logging utilities for model operations."""

import logging
import json
from typing import Dict, Any

class StructuredLogger:
    """Logger that adds structured context to messages."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        
    def info(self, msg: str, **context):
        """Log info message with structured context."""
        self._log(logging.INFO, msg, **context)
        
    def error(self, msg: str, **context):
        """Log error message with structured context."""
        self._log(logging.ERROR, msg, **context)
        
    def _log(self, level: int, msg: str, **context):
        """Internal method to format and log messages."""
        if context:
            msg = f"{msg} | {json.dumps(context)}"
        self.logger.log(level, msg)

# Create default logger instance
mesh_logger = StructuredLogger("tmd.model.mesh")
