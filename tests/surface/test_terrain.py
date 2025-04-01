#!/usr/bin/env python3
"""
Tests for TMDTerrain class.

This module contains unit tests for the TMDTerrain class,
which provides functions to generate synthetic height maps and TMD files.
"""

import os
import tempfile
import numpy as np
import pytest
from pathlib import Path
from unittest import mock

# Import the class to test
from tmd.surface.terrain import TMDTerrain
from tmd.utils.utils import TMDUtils


class TestTMDTerrain:
    """Test suite for TMDTerrain class."""

    def setup_method(self):
        """Set up test environment."""
        # Create a temporary directory for file outputs
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)

    def teardown_method(self):
        """Clean up after each test."""
        self.temp_dir.cleanup()

    def test_create_sample_height_map_default(self):
        """Test creating a sample height map with default parameters."""
        # Create a height map with default parameters
        height_map = TMDTerrain.create_sample_height_map()
        
        # Check dimensions
        assert height_map.shape == (100, 100)
        
        # Check data type
        assert height_map.dtype == np.float32
        
        # Check range (should be normalized to [0, 1])
        assert np.min(height_map) >= 0.0
        assert np.max(height_map) <= 1.0

    def test_create_sample_height_map_custom_size(self):
        """Test creating a sample height map with custom dimensions."""
        # Create a height map with custom size
        width, height = 50, 75
        height_map = TMDTerrain.create_sample_height_map(width=width, height=height)
        
        # Check dimensions
        assert height_map.shape == (height, width)

    def test_create_sample_height_map_patterns(self):
        """Test creating height maps with different patterns."""
        # Test all available patterns
        patterns = ["waves", "peak", "dome", "ramp", "combined"]
        
        for pattern in patterns:
            height_map = TMDTerrain.create_sample_height_map(
                width=50, height=50, pattern=pattern
            )
            
            # Basic checks for each pattern
            assert height_map.shape == (50, 50)
            assert height_map.dtype == np.float32
            assert np.min(height_map) >= 0.0
            assert np.max(height_map) <= 1.0
            
            # Verify that height maps with different patterns are different
            if pattern != "combined":  # Skip combined as it contains waves
                waves_map = TMDTerrain.create_sample_height_map(
                    width=50, height=50, pattern="waves"
                )
                # The maps should have different values
                assert not np.allclose(height_map, waves_map)

    def test_create_sample_height_map_noise(self):
        """Test the effect of noise on height maps."""
        # Create maps with different noise levels
        no_noise = TMDTerrain.create_sample_height_map(
            width=50, height=50, pattern="waves", noise_level=0.0
        )
        
        low_noise = TMDTerrain.create_sample_height_map(
            width=50, height=50, pattern="waves", noise_level=0.01
        )
        
        high_noise = TMDTerrain.create_sample_height_map(
            width=50, height=50, pattern="waves", noise_level=0.1
        )
        
        # Check that more noise leads to higher variance
        assert np.var(low_noise - no_noise) > 0  # Low noise has some effect
        assert np.var(high_noise - no_noise) > np.var(low_noise - no_noise)  # Higher noise has more effect

    def test_create_sample_height_map_invalid_pattern(self):
        """Test creating a height map with an invalid pattern."""
        # Invalid pattern should return zeros
        height_map = TMDTerrain.create_sample_height_map(pattern="invalid_pattern")
        
        # Should return zeros array
        assert np.all(height_map == 0.0)
        assert height_map.shape == (100, 100)

    def test_generate_synthetic_tmd(self):
        """Test generating a synthetic TMD file."""
        # Set up output path
        output_path = self.test_dir / "test_synthetic.tmd"
        
        # Mock TMDUtils.write_tmd_file to avoid actual file operations
        with mock.patch.object(TMDUtils, 'write_tmd_file') as mock_write:
            mock_write.return_value = str(output_path)
            
            # Generate a synthetic TMD file
            result_path = TMDTerrain.generate_synthetic_tmd(
                output_path=str(output_path),
                width=50,
                height=50,
                pattern="waves",
                comment="Test Comment",
                version=2
            )
            
            # Check that result path matches expected
            assert result_path == str(output_path)
            
            # Check that write_tmd_file was called with correct parameters
            mock_write.assert_called_once()
            
            # Extract the height_map argument from the call
            height_map_arg = mock_write.call_args[1]['height_map']
            assert height_map_arg.shape == (50, 50)
            assert height_map_arg.dtype == np.float32
            
            # Check other arguments
            assert mock_write.call_args[1]['output_path'] == str(output_path)
            assert mock_write.call_args[1]['comment'] == "Test Comment"
            assert mock_write.call_args[1]['version'] == 2

    def test_generate_synthetic_tmd_default_path(self):
        """Test generating a synthetic TMD file with default output path."""
        # Mock os.makedirs to avoid directory creation
        with mock.patch('os.makedirs') as mock_makedirs:
            # Mock TMDUtils.write_tmd_file to avoid actual file operations
            with mock.patch.object(TMDUtils, 'write_tmd_file') as mock_write:
                mock_write.return_value = "output/synthetic.tmd"
                
                # Generate a synthetic TMD file with default path
                result_path = TMDTerrain.generate_synthetic_tmd()
                
                # Check that os.makedirs was called to create the output directory
                mock_makedirs.assert_called_once_with("output", exist_ok=True)
                
                # Check that write_tmd_file was called
                mock_write.assert_called_once()
                
                # Check result path
                assert result_path == "output/synthetic.tmd"

    def test_generate_synthetic_tmd_integration(self):
        """Integration test for generating and reading a synthetic TMD file."""
        # Skip this test if we're in a CI environment or don't want to create actual files
        # pytest.skip("Skip to avoid creating actual files")
        
        # Set up output path
        output_path = self.test_dir / "integration_test.tmd"
        
        # Generate actual TMD file
        result_path = TMDTerrain.generate_synthetic_tmd(
            output_path=str(output_path),
            width=25,
            height=25,
            pattern="dome",
            comment="Integration Test",
            version=2
        )
        
        # Check that file was created
        assert os.path.exists(result_path)
        
        # Try reading the file back using TMDUtils
        try:
            metadata, height_map = TMDUtils.process_tmd_file(result_path)
            
            # Check basic metadata
            assert metadata["width"] == 25
            assert metadata["height"] == 25
            assert metadata["version"] == 2
            assert "Integration Test" in metadata.get("comment", "")
            
            # Check height map properties
            assert height_map.shape == (25, 25)
            assert height_map.dtype == np.float32
            assert np.min(height_map) >= 0.0
            assert np.max(height_map) <= 1.0
            
        except Exception as e:
            pytest.fail(f"Failed to read generated TMD file: {e}")


if __name__ == "__main__":
    pytest.main(["-v", __file__])