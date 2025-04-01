#!/usr/bin/env python3
"""
Tests for TMD version and author information.

This module contains unit tests that verify the version, author, and license
information for the TMD (True Map Data) Toolkit.
"""

import pytest
import re
import sys
import importlib.util


class TestVersionInfo:
    """Test cases for TMD version and author information."""
    
    def setup_method(self):
        """Set up the test by importing the version module."""
        # Assuming the module is named 'version_info.py' - adjust as needed
        try:
            # Try direct import first
            import version_info
            self.module = version_info
        except ImportError:
            # If that fails, try to load the module from the file path
            # This approach is useful in CI/CD environments
            try:
                spec = importlib.util.spec_from_file_location("version_info", "version_info.py")
                self.module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(self.module)
            except (ImportError, FileNotFoundError):
                pytest.skip("Could not import version_info module")
    
    def test_version_defined(self):
        """Test that __version__ is defined and has the correct format."""
        assert hasattr(self.module, "__version__"), "__version__ attribute is missing"
        assert isinstance(self.module.__version__, str), "__version__ should be a string"
        
        # Check version format (semantic versioning)
        version_pattern = r'^\d+\.\d+\.\d+(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$'
        assert re.match(version_pattern, self.module.__version__), f"Version '{self.module.__version__}' doesn't follow semantic versioning"
        
        # More specific check for the current version
        assert self.module.__version__ == "0.1.4", f"Expected version 0.1.4, got {self.module.__version__}"
    
    def test_author_defined(self):
        """Test that __author__ is defined and has the expected value."""
        assert hasattr(self.module, "__author__"), "__author__ attribute is missing"
        assert isinstance(self.module.__author__, str), "__author__ should be a string"
        assert self.module.__author__ == "TMD Contributors", f"Expected author 'TMD Contributors', got {self.module.__author__}"
    
    def test_license_defined(self):
        """Test that __license__ is defined and has the expected value."""
        assert hasattr(self.module, "__license__"), "__license__ attribute is missing"
        assert isinstance(self.module.__license__, str), "__license__ should be a string"
        assert self.module.__license__ == "MIT", f"Expected license 'MIT', got {self.module.__license__}"
    
    def test_module_docstring(self):
        """Test that the module has an appropriate docstring."""
        assert self.module.__doc__ is not None, "Module is missing a docstring"
        assert "Version and Author Information" in self.module.__doc__, "Docstring should mention Version and Author Information"
        assert "TMD" in self.module.__doc__, "Docstring should mention TMD"
        

if __name__ == "__main__":
    pytest.main(["-v", __file__])