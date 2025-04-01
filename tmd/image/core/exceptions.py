#!/usr/bin/env python3
"""
Custom exception classes for the image processing package.
"""
from tmd.utils.exceptions import TMDException

class ImageException(TMDException):
    """Base exception for all image-related errors."""
    pass

class ImageIOError(ImageException):
    """Exception raised when there's an issue with image I/O operations."""
    pass

class ImageProcessingError(ImageException):
    """Exception raised when there's an error during image processing."""
    pass

class ImageConversionError(ImageException):
    """Exception raised when there's an issue converting between image formats."""
    pass

class ImageResizeError(ImageException):
    """Exception raised when there's an issue resizing an image."""
    pass

class ImageMetadataError(ImageException):
    """Exception raised when there's an issue with image metadata."""
    pass

class MapGenerationError(ImageException):
    """Exception raised when map generation fails."""
    pass

class MapExportError(ImageException):
    """Exception raised when map export fails."""
    pass

class InvalidHeightMapError(ImageException):
    """Exception raised when an invalid height map is provided."""
    pass

class MapGeneratorNotFoundError(ImageException):
    """Exception raised when a requested map generator is not found."""
    pass
