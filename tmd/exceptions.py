#!/usr/bin/env python3
"""
TMD Exceptions

This module defines custom exceptions used throughout the TMD library.
"""

class TMDException(Exception):
    """Base class for all TMD exceptions."""
    pass

class TMDFileError(TMDException):
    """Exception raised when there is an error processing a TMD file."""
    pass

class TMDVersionError(TMDFileError):
    """Exception raised when a TMD version is not supported."""
    pass

class TMDDataError(TMDException):
    """Exception raised when there is a problem with TMD data."""
    pass

class TMDProcessingError(Exception):
    """Custom exception for TMD processing errors."""
    pass