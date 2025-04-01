#!/usr/bin/env python3
"""
Tests for TMD Exceptions module.

This module contains unit tests for all exception classes defined in the 
TMD exceptions module.
"""

import pytest
from tmd.exceptions import (
    TMDException, 
    TMDFileError, 
    TMDVersionError, 
    TMDDataError
)


class TestTMDExceptions:
    """Test cases for TMD exception classes."""

    def test_tmd_exception_base(self):
        """Test that TMDException can be raised and caught properly."""
        # Test raising the exception with a message
        error_msg = "Base TMD exception"
        with pytest.raises(TMDException) as excinfo:
            raise TMDException(error_msg)
        
        # Verify the error message
        assert str(excinfo.value) == error_msg
        
        # Test that it's derived from Exception
        assert isinstance(excinfo.value, Exception)

    def test_tmd_file_error(self):
        """Test TMDFileError exception functionality."""
        # Test raising with a message
        error_msg = "File processing error"
        with pytest.raises(TMDFileError) as excinfo:
            raise TMDFileError(error_msg)
        
        # Verify the error message
        assert str(excinfo.value) == error_msg
        
        # Test inheritance
        assert isinstance(excinfo.value, TMDException)
        assert isinstance(excinfo.value, Exception)

    def test_tmd_version_error(self):
        """Test TMDVersionError exception functionality."""
        # Test raising with a message
        error_msg = "Version 2.1 is not supported"
        with pytest.raises(TMDVersionError) as excinfo:
            raise TMDVersionError(error_msg)
        
        # Verify the error message
        assert str(excinfo.value) == error_msg
        
        # Test inheritance chain
        assert isinstance(excinfo.value, TMDFileError)
        assert isinstance(excinfo.value, TMDException)
        assert isinstance(excinfo.value, Exception)

    def test_tmd_data_error(self):
        """Test TMDDataError exception functionality."""
        # Test raising with a message
        error_msg = "Invalid data format in TMD structure"
        with pytest.raises(TMDDataError) as excinfo:
            raise TMDDataError(error_msg)
        
        # Verify the error message
        assert str(excinfo.value) == error_msg
        
        # Test inheritance
        assert isinstance(excinfo.value, TMDException)
        assert isinstance(excinfo.value, Exception)
        assert not isinstance(excinfo.value, TMDFileError)

    def test_exception_hierarchy(self):
        """Test catching exceptions through the hierarchy."""
        
        # Test that TMDFileError can be caught as TMDException
        try:
            raise TMDFileError("File error")
        except TMDException as e:
            assert str(e) == "File error"
        
        # Test that TMDVersionError can be caught as TMDFileError
        try:
            raise TMDVersionError("Version error")
        except TMDFileError as e:
            assert str(e) == "Version error"
        
        # Test that all can be caught as Exception
        exceptions = [
            TMDException("Base error"),
            TMDFileError("File error"),
            TMDVersionError("Version error"),
            TMDDataError("Data error")
        ]
        
        for exc in exceptions:
            try:
                raise exc
            except Exception as e:
                assert str(e) == str(exc)


if __name__ == "__main__":
    pytest.main(["-v", __file__])