"""
Base classes for TMD sequence exporters.

This module provides the BaseExporter class that defines common functionality
for all TMD sequence exporters.
"""

import os
import re
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

class BaseExporter:
    """
    Base class for all TMD sequence exporters.
    
    This class provides common utility methods used by various exporters.
    """
    
    def ensure_output_dir(self, directory: str) -> bool:
        """
        Ensure that the output directory exists, creating it if necessary.
        
        Args:
            directory: Path to the directory
            
        Returns:
            True if directory exists or was created, False otherwise
        """
        if not directory:
            return True  # No directory specified (using current directory)
        
        try:
            os.makedirs(directory, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Failed to create output directory {directory}: {e}")
            return False
    
    def sanitize_filename(self, filename: str) -> str:
        """
        Sanitize a string to be used as a filename.
        
        Args:
            filename: Original filename or string
            
        Returns:
            Sanitized filename safe for filesystem use
        """
        # Replace invalid characters with underscores
        sanitized = re.sub(r'[\\/*?:"<>|]', "_", filename)
        
        # Replace multiple spaces/underscores with single ones
        sanitized = re.sub(r'[ _]+', "_", sanitized)
        
        # Trim leading/trailing underscores and spaces
        sanitized = sanitized.strip("_ ")
        
        # Ensure we have at least some valid characters
        if not sanitized:
            sanitized = "unnamed"
            
        return sanitized
    
    def get_file_extension(self, format_name: str) -> str:
        """
        Get the appropriate file extension for a given format name.
        
        Args:
            format_name: Name of the format (e.g., 'png', 'jpg')
            
        Returns:
            File extension with leading dot
        """
        format_name = format_name.lower()
        if not format_name.startswith('.'):
            return f".{format_name}"
        return format_name
    
    def validate_output_path(self, path: str, required_extension: Optional[str] = None) -> str:
        """
        Validate and normalize an output path.
        
        Args:
            path: Output file path
            required_extension: Optional extension to enforce
            
        Returns:
            Normalized output path
        """
        if not path:
            raise ValueError("Output path cannot be empty")
            
        # Ensure directory exists
        directory = os.path.dirname(path)
        if directory and not self.ensure_output_dir(directory):
            raise ValueError(f"Could not create directory: {directory}")
            
        # Add/replace extension if required
        if required_extension:
            ext = self.get_file_extension(required_extension)
            base, current_ext = os.path.splitext(path)
            if not current_ext or current_ext.lower() != ext.lower():
                path = f"{base}{ext}"
                
        return path
