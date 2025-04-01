#!/usr/bin/env python3
"""
Tests for TMD utility functions.

This module provides unit tests for the TMDUtils class and its methods,
ensuring proper functionality for TMD file processing and analysis.
"""

import os
import struct
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

# Import the TMDUtils class from your module
from tmd.utils.utils import TMDUtils
from tmd.utils.exceptions import TMDVersionError


class TestTMDUtils:
    """Test suite for TMDUtils class."""

    def setup_method(self):
        """Set up test data and temporary files for each test."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
        
        # Create a simple test height map
        self.height_map = np.array([
            [0.0, 0.1, 0.2],
            [0.3, 0.4, 0.5],
            [0.6, 0.7, 0.8]
        ], dtype=np.float32)
        
        # Create test TMD file paths
        self.v1_file_path = self.test_dir / "test_v1.tmd"
        self.v2_file_path = self.test_dir / "test_v2.tmd"
        
        # Create test TMD files
        self._create_test_tmd_files()

    def teardown_method(self):
        """Clean up temporary files after each test."""
        self.temp_dir.cleanup()

    def _create_test_tmd_files(self):
        """Create test TMD files for v1 and v2 formats."""
        # Create a v1 TMD file
        with open(self.v1_file_path, "wb") as f:
            # Write v1 header
            header = "Binary TrueMap Data File\r\n"
            header_bytes = header.encode("ascii")
            remaining_header = 28 - len(header_bytes)
            if remaining_header > 0:
                header_bytes += b"\0" * remaining_header
            f.write(header_bytes)
            
            # Write dimensions
            width, height = self.height_map.shape[1], self.height_map.shape[0]
            f.write(struct.pack("<II", width, height))
            
            # Write spatial info (x_length, y_length only in v1)
            f.write(struct.pack("<ff", 10.0, 10.0))
            
            # Write height map data
            f.write(self.height_map.tobytes())
        
        # Create a v2 TMD file
        with open(self.v2_file_path, "wb") as f:
            # Write v2 header
            header = "Binary TrueMap Data File v2.0\n"
            header_bytes = header.encode("ascii")
            remaining_header = 32 - len(header_bytes)
            if remaining_header > 0:
                header_bytes += b"\0" * remaining_header
            f.write(header_bytes)
            
            # Write comment
            comment = "Test Comment\n"
            comment_bytes = comment.encode("ascii")
            remaining_comment = 24 - len(comment_bytes)
            if remaining_comment > 0:
                comment_bytes += b"\0" * remaining_comment
            f.write(comment_bytes)
            
            # Write dimensions
            width, height = self.height_map.shape[1], self.height_map.shape[0]
            f.write(struct.pack("<II", width, height))
            
            # Write spatial info (x_length, y_length, x_offset, y_offset in v2)
            f.write(struct.pack("<ffff", 10.0, 10.0, 1.0, 1.0))
            
            # Write height map data
            f.write(self.height_map.tobytes())

    def test_hexdump(self):
        """Test hexdump functionality."""
        test_bytes = b"Hello, World!"
        result = TMDUtils.hexdump(test_bytes, width=8)
        
        # Check that the result is a non-empty string
        assert isinstance(result, str)
        assert len(result) > 0
        
        # Check that the hexdump contains the expected hex values
        assert "48 65 6c 6c 6f 2c 20 57" in result  # "Hello, W" in hex
        assert "6f 72 6c 64 21" in result  # "orld!" in hex
        
        # Test empty input
        assert TMDUtils.hexdump(b"") == "(empty)"
        
        # Test invalid start offset
        assert "(invalid start offset)" in TMDUtils.hexdump(b"Hello", start=10)

    def test_read_null_terminated_string(self):
        """Test reading null-terminated strings."""
        test_data = b"Test String\0Extra data"
        with tempfile.TemporaryFile() as f:
            f.write(test_data)
            f.seek(0)
            
            result = TMDUtils.read_null_terminated_string(f)
            assert result == "Test String"
            
            # Check that the file pointer is positioned after the null
            assert f.tell() == len(b"Test String\0")
        
        # Test string without null terminator
        test_data = b"No Null Terminator"
        with tempfile.TemporaryFile() as f:
            f.write(test_data)
            f.seek(0)
            
            result = TMDUtils.read_null_terminated_string(f)
            assert result == "No Null Terminator"

    def test_detect_tmd_version(self):
        """Test TMD version detection."""
        # Test v1 detection
        assert TMDUtils.detect_tmd_version(self.v1_file_path) == 1
        
        # Test v2 detection
        assert TMDUtils.detect_tmd_version(self.v2_file_path) == 2
        
        # Test file not found
        with pytest.raises(FileNotFoundError):
            TMDUtils.detect_tmd_version(self.test_dir / "nonexistent.tmd")
        
        # Test invalid file (create a tiny file that can't be a valid TMD)
        invalid_file = self.test_dir / "invalid.tmd"
        with open(invalid_file, "wb") as f:
            f.write(b"ABC")
        
        with pytest.raises(TMDVersionError):
            TMDUtils.detect_tmd_version(invalid_file)

    def test_process_tmd_file(self):
        """Test processing of TMD files."""
        # Test v1 file processing
        metadata, height_map = TMDUtils.process_tmd_file(self.v1_file_path)
        
        assert metadata["version"] == 1
        assert metadata["width"] == 3
        assert metadata["height"] == 3
        assert metadata["x_length"] == 10.0
        assert metadata["y_length"] == 10.0
        assert np.array_equal(height_map, self.height_map)
        
        # Test v2 file processing
        metadata, height_map = TMDUtils.process_tmd_file(self.v2_file_path)
        
        assert metadata["version"] == 2
        assert metadata["width"] == 3
        assert metadata["height"] == 3
        assert metadata["x_length"] == 10.0
        assert metadata["y_length"] == 10.0
        assert metadata["x_offset"] == 1.0
        assert metadata["y_offset"] == 1.0
        assert metadata["comment"] == "Test Comment"
        
        # Check if height map data is as expected
        assert np.array_equal(height_map, self.height_map)
        
        # Test force_offset parameter
        metadata, _ = TMDUtils.process_tmd_file(
            self.v2_file_path, force_offset=(2.0, 2.0)
        )
        assert metadata["x_offset"] == 2.0
        assert metadata["y_offset"] == 2.0
        
        # Test file not found
        with pytest.raises(FileNotFoundError):
            TMDUtils.process_tmd_file(self.test_dir / "nonexistent.tmd")

    def test_write_tmd_file(self):
        """Test writing TMD files."""
        # Test writing v1 file
        output_path_v1 = self.test_dir / "output_v1.tmd"
        result_path = TMDUtils.write_tmd_file(
            self.height_map,
            output_path_v1,
            x_length=10.0,
            y_length=10.0,
            version=1
        )
        
        assert result_path == str(output_path_v1)
        assert os.path.exists(output_path_v1)
        
        # Verify the written file by reading it back
        metadata, height_map = TMDUtils.process_tmd_file(output_path_v1)
        assert metadata["version"] == 1
        assert metadata["width"] == 3
        assert metadata["height"] == 3
        assert np.array_equal(height_map, self.height_map)
        
        # Test writing v2 file
        output_path_v2 = self.test_dir / "output_v2.tmd"
        result_path = TMDUtils.write_tmd_file(
            self.height_map,
            output_path_v2,
            comment="Test Output",
            x_length=15.0,
            y_length=15.0,
            x_offset=2.0,
            y_offset=2.0,
            version=2
        )
        
        assert result_path == str(output_path_v2)
        assert os.path.exists(output_path_v2)
        
        # Verify the written file by reading it back
        metadata, height_map = TMDUtils.process_tmd_file(output_path_v2)
        assert metadata["version"] == 2
        assert metadata["width"] == 3
        assert metadata["height"] == 3
        assert metadata["x_length"] == 15.0
        assert metadata["y_length"] == 15.0
        assert metadata["x_offset"] == 2.0
        assert metadata["y_offset"] == 2.0
        assert "Test Output" in metadata["comment"]
        assert np.array_equal(height_map, self.height_map)
        
        # Test invalid inputs
        with pytest.raises(ValueError):
            # Invalid height map (not a 2D array)
            TMDUtils.write_tmd_file(
                np.array([1, 2, 3]),  # 1D array
                self.test_dir / "invalid.tmd"
            )
        
        with pytest.raises(ValueError):
            # Invalid version
            TMDUtils.write_tmd_file(
                self.height_map,
                self.test_dir / "invalid.tmd",
                version=3  # Only versions 1 and 2 are supported
            )

    def test_downsample_array(self):
        """Test array downsampling."""
        # Create a larger test array
        test_array = np.ones((10, 10)) * np.arange(10).reshape(10, 1)
        
        # Test downsampling to smaller dimensions
        result = TMDUtils.downsample_array(test_array, 5, 5)
        assert result.shape == (5, 5)
        
        # Test with different methods
        for method in ["nearest", "bilinear", "bicubic"]:
            result = TMDUtils.downsample_array(test_array, 5, 5, method=method)
            assert result.shape == (5, 5)
        
        # Test fallback path (when scipy is not available)
        with mock.patch("tmd.utils.core.TMDUtils.get_scipy_or_fallback", 
                       return_value=(None, False)):
            result = TMDUtils.downsample_array(test_array, 5, 5)
            assert result.shape == (5, 5)

    def test_quantize_array(self):
        """Test array quantization."""
        # Test array with continuous values
        test_array = np.linspace(0, 1, 100).reshape(10, 10)
        
        # Quantize to 5 levels
        result = TMDUtils.quantize_array(test_array, levels=5)
        
        # Check that the result has the same shape
        assert result.shape == test_array.shape
        
        # Check that we have at most 5 unique values
        assert len(np.unique(result)) <= 5
        
        # Test with single value array
        single_value = np.ones((5, 5))
        result = TMDUtils.quantize_array(single_value)
        assert np.array_equal(result, single_value)
        
        # Test with invalid levels
        with pytest.raises(ValueError):
            TMDUtils.quantize_array(test_array, levels=1)  # At least 2 levels required

    def test_print_message(self):
        """Test print message formatting."""
        # Use mock to capture print output
        with mock.patch("builtins.print") as mock_print:
            # Test with rich formatting disabled
            TMDUtils.print_message("Test message", "info", use_rich=False)
            mock_print.assert_called_with("Info: Test message")
            
            TMDUtils.print_message("Test warning", "warning", use_rich=False)
            mock_print.assert_called_with("Warning: Test warning")
            
            TMDUtils.print_message("Test error", "error", use_rich=False)
            mock_print.assert_called_with("Error: Test error")
            
            TMDUtils.print_message("Test success", "success", use_rich=False)
            mock_print.assert_called_with("Success: Test success")
            
            # Test unknown message type
            TMDUtils.print_message("Unknown type", "unknown", use_rich=False)
            mock_print.assert_called_with("Unknown type")

    def test_get_scipy_or_fallback(self):
        """Test scipy import or fallback logic."""
        # This tests the function directly - actual behavior depends on environment
        scipy, has_scipy = TMDUtils.get_scipy_or_fallback()
        
        # If scipy is available, check that it's returned correctly
        if has_scipy:
            assert scipy is not None
            assert hasattr(scipy, "ndimage")
        
        # Test the fallback path with a mock
        with mock.patch("importlib.import_module", side_effect=ImportError):
            with mock.patch("tmd.utils.core.TMDUtils.print_message") as mock_print:
                scipy, has_scipy = TMDUtils.get_scipy_or_fallback()
                assert not has_scipy
                assert scipy is None
                # Check that a warning was printed
                mock_print.assert_called_once()


if __name__ == "__main__":
    pytest.main(["-v", __file__])