#!/usr/bin/env python3
"""
Tests for TMDFileUtilities class.

This module contains unit tests for the TMDFileUtilities class methods
including dependency management, imports, and environment setup.
"""

import os
import sys
import importlib
import logging
from unittest import mock
import pytest

# Import the class to test
from tmd.utils.files import TMDFileUtilities
from tmd.utils.exceptions import TMDImportError
from tmd.cli.core.ui import HAS_RICH



class TestTMDFileUtilities:
    """Test cases for TMDFileUtilities class."""

    def setup_method(self):
        """Set up test environment."""
        # Clear module cache before each test
        TMDFileUtilities._module_cache = {}
        
        # Create a simple mock module for testing
        self.mock_module = mock.MagicMock()
        self.mock_module.__version__ = "1.0.0"

    def test_import_optional_dependency_success(self):
        """Test successful import of optional dependency."""
        # Mock successful import
        with mock.patch.dict('sys.modules', {'testmodule': self.mock_module}):
            # Mock __import__ to return our mock module
            with mock.patch('builtins.__import__', return_value=self.mock_module):
                result = TMDFileUtilities.import_optional_dependency('testmodule')
                
                # Check that the module was imported correctly
                assert result is self.mock_module
                
                # Check that it was cached
                assert 'testmodule' in TMDFileUtilities._module_cache
                assert TMDFileUtilities._module_cache['testmodule'] is self.mock_module
                
                # Call again to test cache lookup
                cached_result = TMDFileUtilities.import_optional_dependency('testmodule')
                assert cached_result is self.mock_module

    def test_import_optional_dependency_failure(self):
        """Test failed import of optional dependency."""
        # Mock failed import
        with mock.patch('builtins.__import__', side_effect=ImportError("Module not found")):
            result = TMDFileUtilities.import_optional_dependency('nonexistentmodule')
            
            # Check that the result is None
            assert result is None
            
            # Check that the failure was cached
            assert 'nonexistentmodule' in TMDFileUtilities._module_cache
            assert TMDFileUtilities._module_cache['nonexistentmodule'] is None

    def test_import_error_decorator_no_raise(self):
        """Test the import error decorator when not raising errors."""
        # Create a function that will raise ImportError
        @TMDFileUtilities.import_error_decorator(raise_error=False)
        def test_function():
            raise ImportError("Test import error")
            
        # Function should return None without raising
        assert test_function() is None

    def test_import_error_decorator_with_raise(self):
        """Test the import error decorator when raising errors."""
        # Create a function that will raise ImportError
        @TMDFileUtilities.import_error_decorator(raise_error=True)
        def test_function():
            raise ImportError("Test import error")
            
        # Function should raise TMDImportError
        with pytest.raises(TMDImportError):
            test_function()

    def test_import_error_decorator_success(self):
        """Test the import error decorator with successful function."""
        # Create a function that succeeds
        @TMDFileUtilities.import_error_decorator(raise_error=True)
        def test_function():
            return "success"
            
        # Function should return its normal result
        assert test_function() == "success"

    def test_check_and_install_dependencies(self):
        """Test dependency checking functionality."""
        # Mock dependencies for testing
        dependencies = {
            "numpy": ">=1.20.0",
            "nonexistent": ">=1.0.0"
        }
        
        # Mock the import function to succeed for numpy and fail for nonexistent
        def mock_import(name):
            if name == "numpy":
                return self.mock_module
            raise ImportError(f"No module named '{name}'")
        
        # Mock print_message to avoid actual printing
        with mock.patch('builtins.__import__', side_effect=mock_import):
            with mock.patch.object(TMDFileUtilities, 'print_message'):
                # Without auto-install
                results = TMDFileUtilities.check_and_install_dependencies(
                    dependencies, auto_install=False, quiet=True
                )
                
                assert results["numpy"] is True
                assert results["nonexistent"] is False
                
                # With auto-install (but mock the subprocess.run)
                with mock.patch('subprocess.run') as mock_run:
                    # Make subprocess.run return success for installation
                    mock_process = mock.MagicMock()
                    mock_process.returncode = 0
                    mock_run.return_value = mock_process
                    
                    results = TMDFileUtilities.check_and_install_dependencies(
                        dependencies, auto_install=True, quiet=True
                    )
                    
                    # Check that pip install was called
                    mock_run.assert_called_once()
                    assert "pip" in mock_run.call_args[0][0]
                    assert "install" in mock_run.call_args[0][0]
                    assert "nonexistent>=1.0.0" in mock_run.call_args[0][0]

    def test_check_tmd_dependencies(self):
        """Test TMD dependency checking functionality."""
        # Mock successful dependency checks
        with mock.patch.object(
            TMDFileUtilities, 'check_and_install_dependencies', 
            return_value={"numpy": True, "scipy": True}
        ):
            # Mock the TMD core imports
            with mock.patch.dict('sys.modules', {
                'tmd': mock.MagicMock(),
                'tmd.utils.utils': mock.MagicMock()
            }):
                result = TMDFileUtilities.check_tmd_dependencies(quiet=True)
                assert result is True
                
        # Mock failed dependency checks
        with mock.patch.object(
            TMDFileUtilities, 'check_and_install_dependencies', 
            return_value={"numpy": True, "scipy": False}
        ):
            # Mock the TMD core imports
            with mock.patch.dict('sys.modules', {
                'tmd': mock.MagicMock(),
                'tmd.utils.utils': mock.MagicMock()
            }):
                # Without exit_on_failure
                result = TMDFileUtilities.check_tmd_dependencies(quiet=True)
                assert result is False
                
                # With exit_on_failure, but mock sys.exit
                with mock.patch('sys.exit') as mock_exit:
                    result = TMDFileUtilities.check_tmd_dependencies(
                        quiet=True, exit_on_failure=True
                    )
                    mock_exit.assert_called_once_with(1)

    def test_import_with_error_handling(self):
        """Test module import with error handling."""
        # Mock successful import
        with mock.patch.object(
            TMDFileUtilities, 'import_optional_dependency', 
            return_value=self.mock_module
        ):
            result = TMDFileUtilities.import_with_error_handling(
                'testmodule', 'Test Module', 'pip install testmodule'
            )
            assert result is self.mock_module
            
        # Mock failed import
        with mock.patch.object(
            TMDFileUtilities, 'import_optional_dependency', 
            return_value=None
        ):
            # Without exit_on_failure
            with mock.patch('rich.print' if HAS_RICH else 'builtins.print'):
                result = TMDFileUtilities.import_with_error_handling(
                    'nonexistent', 'Nonexistent Module', 'pip install nonexistent'
                )
                assert result is None
                
            # With exit_on_failure, but mock sys.exit
            with mock.patch('sys.exit') as mock_exit:
                with mock.patch('rich.print' if HAS_RICH else 'builtins.print'):
                    result = TMDFileUtilities.import_with_error_handling(
                        'nonexistent', 'Nonexistent Module', 
                        'pip install nonexistent', exit_on_failure=True
                    )
                    mock_exit.assert_called_once_with(1)

    def test_check_visualization_capabilities(self):
        """Test visualization capabilities check."""
        # We can't easily test the actual imports, so we'll just verify the function returns a tuple
        result = _check_visualization_capabilities()
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert all(isinstance(x, bool) for x in result)


if __name__ == "__main__":
    pytest.main(["-v", __file__])