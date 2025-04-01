#!/usr/bin/env python3
"""
Tests for TMD Exceptions.

This module contains unit tests for all exception classes defined in the
TMD exceptions module.
"""

import pytest
import sys
from typing import Type

# Import all exceptions from the module
from tmd.utils.exceptions import (
    TMDException,
    TMDFileError,
    TMDVersionError,
    TMDDataError,
    TMDImportError,
    TMDEnvironmentError
)


class TestTMDExceptions:
    """Test suite for TMD exception classes."""

    def test_exception_inheritance(self):
        """Test the inheritance hierarchy of all TMD exceptions."""
        # Check that all exceptions inherit from the base TMDException
        assert issubclass(TMDFileError, TMDException)
        assert issubclass(TMDVersionError, TMDFileError)
        assert issubclass(TMDDataError, TMDFileError)
        assert issubclass(TMDImportError, TMDException)
        assert issubclass(TMDEnvironmentError, TMDException)
        
        # Check that all exceptions ultimately inherit from Python's Exception
        assert issubclass(TMDException, Exception)
        
        # Verify appropriate parent-child relationships
        assert issubclass(TMDVersionError, TMDFileError)
        assert issubclass(TMDDataError, TMDFileError)
        assert not issubclass(TMDImportError, TMDFileError)
        assert not issubclass(TMDEnvironmentError, TMDFileError)

    def test_exception_instances(self):
        """Test creating and using TMD exception instances."""
        # Test creating instances with messages
        base_exc = TMDException("Base exception message")
        file_exc = TMDFileError("File error message")
        version_exc = TMDVersionError("Version error message")
        data_exc = TMDDataError("Data error message")
        import_exc = TMDImportError("Import error message")
        env_exc = TMDEnvironmentError("Environment error message")
        
        # Check that messages are stored correctly
        assert str(base_exc) == "Base exception message"
        assert str(file_exc) == "File error message"
        assert str(version_exc) == "Version error message"
        assert str(data_exc) == "Data error message"
        assert str(import_exc) == "Import error message"
        assert str(env_exc) == "Environment error message"
        
        # Check instance types
        assert isinstance(base_exc, TMDException)
        assert isinstance(file_exc, TMDFileError)
        assert isinstance(version_exc, TMDVersionError)
        assert isinstance(data_exc, TMDDataError)
        assert isinstance(import_exc, TMDImportError)
        assert isinstance(env_exc, TMDEnvironmentError)
        
        # Check inheritance of instances
        assert isinstance(file_exc, TMDException)
        assert isinstance(version_exc, TMDFileError)
        assert isinstance(version_exc, TMDException)
        assert isinstance(data_exc, TMDFileError)
        assert isinstance(data_exc, TMDException)
        assert isinstance(import_exc, TMDException)
        assert isinstance(env_exc, TMDException)

    def test_exception_catching(self):
        """Test catching exceptions through the inheritance hierarchy."""
        # Define a helper function to test exception catching
        def assert_caught_by(raised_exc: Exception, caught_by: Type[Exception]) -> bool:
            """
            Test if an exception would be caught by an except clause.
            
            Args:
                raised_exc: The exception instance to raise
                caught_by: The exception class in the except clause
                
            Returns:
                True if exception would be caught, False otherwise
            """
            try:
                raise raised_exc
            except caught_by:
                return True
            except Exception:
                return False
            
        # Test that all TMD exceptions are caught by TMDException
        assert assert_caught_by(TMDFileError("File error"), TMDException)
        assert assert_caught_by(TMDVersionError("Version error"), TMDException)
        assert assert_caught_by(TMDDataError("Data error"), TMDException)
        assert assert_caught_by(TMDImportError("Import error"), TMDException)
        assert assert_caught_by(TMDEnvironmentError("Environment error"), TMDException)
        
        # Test that file-related exceptions are caught by TMDFileError
        assert assert_caught_by(TMDVersionError("Version error"), TMDFileError)
        assert assert_caught_by(TMDDataError("Data error"), TMDFileError)
        
        # Test that non-file exceptions are not caught by TMDFileError
        assert not assert_caught_by(TMDImportError("Import error"), TMDFileError)
        assert not assert_caught_by(TMDEnvironmentError("Environment error"), TMDFileError)
        
        # Test that all exceptions are caught by Python's built-in Exception
        assert assert_caught_by(TMDException("Base error"), Exception)
        assert assert_caught_by(TMDFileError("File error"), Exception)
        assert assert_caught_by(TMDVersionError("Version error"), Exception)
        assert assert_caught_by(TMDDataError("Data error"), Exception)
        assert assert_caught_by(TMDImportError("Import error"), Exception)
        assert assert_caught_by(TMDEnvironmentError("Environment error"), Exception)

    def test_exception_with_cause(self):
        """Test exceptions with a cause (using from keyword)."""
        original_error = ValueError("Original error")
        
        # Create exceptions with a cause
        try:
            try:
                raise original_error
            except ValueError as e:
                raise TMDVersionError("Version error due to value error") from e
        except TMDVersionError as exc:
            # Check the exception chain
            assert isinstance(exc.__cause__, ValueError)
            assert exc.__cause__ is original_error
            assert str(exc.__cause__) == "Original error"
            
        # Test nested causes
        try:
            try:
                try:
                    raise original_error
                except ValueError as e:
                    raise TMDDataError("Data error") from e
            except TMDDataError as e:
                raise TMDFileError("File error") from e
        except TMDFileError as exc:
            # Check the exception chain
            assert isinstance(exc.__cause__, TMDDataError)
            assert isinstance(exc.__cause__.__cause__, ValueError)
            assert exc.__cause__.__cause__ is original_error


if __name__ == "__main__":
    pytest.main(["-v", __file__])