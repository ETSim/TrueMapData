#!/usr/bin/env python3
"""
Exceptions for the TMD Image Processing Package.

This module contains custom exceptions used by the TMD image processing modules.
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
