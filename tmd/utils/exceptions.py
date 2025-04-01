#!/usr/bin/env python3
"""
Exceptions for the TMD Utils Package.

This module contains custom exceptions used by the TMD utils module.
"""

class TMDException(Exception):
    """Base exception for all TMD-related errors."""
    pass


class TMDFileError(TMDException):
    """Base exception for TMD file processing errors."""
    pass


class TMDVersionError(TMDFileError):
    """Exception raised when there's an issue with the TMD file version."""
    pass


class TMDDataError(TMDFileError):
    """Exception raised when there's an issue with TMD data processing."""
    pass


class TMDImportError(TMDException):
    """Exception raised when there's an issue with optional dependencies."""
    pass


class TMDEnvironmentError(TMDException):
    """Exception raised when there's an issue with the environment setup."""
    pass
