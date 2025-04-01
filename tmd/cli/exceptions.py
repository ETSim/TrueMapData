#!/usr/bin/env python3
"""
Custom exceptions for TMD CLI tools.

This module defines exception classes that are specific to the TMD CLI tools,
providing more precise error handling and user feedback.
"""

class TMDCliException(Exception):
    """Base exception for all TMD CLI errors."""
    pass

class CommandError(TMDCliException):
    """Exception raised when a command fails to execute properly."""
    pass

class InputError(TMDCliException):
    """Exception raised for invalid input parameters."""
    pass

class FileError(TMDCliException):
    """Exception raised for file-related errors (not found, permission denied, etc.)."""
    pass

class DependencyError(TMDCliException):
    """Exception raised when a required dependency is missing."""
    pass

class ConfigError(TMDCliException):
    """Exception raised for configuration-related errors."""
    pass
