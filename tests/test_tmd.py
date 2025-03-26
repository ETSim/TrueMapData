"""Unit tests for TMD main package module."""

import unittest
import os
import numpy as np
import tempfile
import shutil
from unittest.mock import patch, mock_open

from tmd.utils.utils import (
    detect_tmd_version,
    process_tmd_file,
    write_tmd_file,
    create_sample_height_map,
    generate_synthetic_tmd
)


class TestMainPackage(unittest.TestCase):
    """Test class for main package functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create some height maps for testing
        self.test_height_map = create_sample_height_map(
            width=50, height=40, pattern='waves', noise_level=0.0
        )

        # Create paths for test files
        self.v1_test_file = os.path.join(self.temp_dir, 'test_v1.tmd')
        self.v2_test_file = os.path.join(self.temp_dir, 'test_v2.tmd')
        
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary directory and its contents
        shutil.rmtree(self.temp_dir)
    
    def test_tmd_read_write_v2(self):
        """Test writing and reading TMD v2 files."""
        # Write a test file
        write_tmd_file(
            self.test_height_map,
            self.v2_test_file,
            comment="Test TMD v2",
            x_length=10.0,
            y_length=8.0,
            version=2
        )
        
        # Check file exists
        self.assertTrue(os.path.exists(self.v2_test_file))
        
        # Read it back
        metadata, height_map = process_tmd_file(self.v2_test_file)
        
        # Verify metadata
        self.assertEqual(metadata['version'], 2)
        self.assertEqual(metadata['width'], 50)
        self.assertEqual(metadata['height'], 40)
        self.assertEqual(metadata['x_length'], 10.0)
        self.assertEqual(metadata['y_length'], 8.0)
        
        # Verify height map data - compare shapes and sample points rather than full array
        self.assertEqual(height_map.shape, (40, 50))
        
        # Sample some points to compare
        sample_points = [(0,0), (0,49), (39,0), (39,49), (20,25)]
        for row, col in sample_points:
            self.assertAlmostEqual(height_map[row, col], self.test_height_map[row, col], places=5)
    
    def test_tmd_read_write_v1(self):
        """Test writing and reading TMD v1 files."""
        # Use a simpler test map for v1 testing to avoid large array issues
        simple_map = np.ones((10, 10), dtype=np.float32) * 0.5
        
        # Add identifiable corner values
        simple_map[0, 0] = 0.1  # Top-left
        simple_map[0, 9] = 0.2  # Top-right
        simple_map[9, 0] = 0.3  # Bottom-left
        simple_map[9, 9] = 0.4  # Bottom-right
        
        # Write a test file with v1 format
        write_tmd_file(
            simple_map,
            self.v1_test_file,
            comment="Test TMD v1",
            x_length=5.0,
            y_length=4.0,
            version=1
        )
        
        # Check file exists
        self.assertTrue(os.path.exists(self.v1_test_file))
        
        # Read it back
        metadata, height_map = process_tmd_file(self.v1_test_file)
        
        # Verify metadata
        self.assertEqual(metadata['version'], 1)
        self.assertEqual(metadata['width'], 10)
        self.assertEqual(metadata['height'], 10)
        self.assertEqual(metadata['x_length'], 5.0)
        self.assertEqual(metadata['y_length'], 4.0)
        
        # Verify corner values to check orientation
        self.assertAlmostEqual(height_map[0, 0], simple_map[0, 0], places=5)  # Top-left
        self.assertAlmostEqual(height_map[0, 9], simple_map[0, 9], places=5)  # Top-right
        self.assertAlmostEqual(height_map[9, 0], simple_map[9, 0], places=5)  # Bottom-left
        self.assertAlmostEqual(height_map[9, 9], simple_map[9, 9], places=5)  # Bottom-right
    
    def test_detect_version(self):
        """Test version detection from file headers."""
        # Create test files with different versions
        write_tmd_file(
            self.test_height_map, self.v1_test_file, version=1
        )
        write_tmd_file(
            self.test_height_map, self.v2_test_file, version=2
        )
        
        # Test version detection
        self.assertEqual(detect_tmd_version(self.v1_test_file), 1)
        self.assertEqual(detect_tmd_version(self.v2_test_file), 2)
    
    def test_generate_synthetic_tmd(self):
        """Test generation of synthetic TMD files."""
        # Generate a synthetic file
        output_path = os.path.join(self.temp_dir, "synthetic.tmd")
        result_path = generate_synthetic_tmd(
            output_path=output_path,
            width=60,
            height=50,
            pattern="peak",
            comment="Synthetic test",
            version=2
        )
        
        # Verify file was created at the right path
        self.assertEqual(result_path, output_path)
        self.assertTrue(os.path.exists(output_path))
        
        # Read it back and verify metadata
        metadata, height_map = process_tmd_file(output_path)
        self.assertEqual(metadata['width'], 60)
        self.assertEqual(metadata['height'], 50)
        self.assertEqual(metadata['version'], 2)
    
    def test_process_file_with_offsets(self):
        """Test processing TMD files with spatial offsets."""
        # Create test file with offsets
        x_offset = 2.0
        y_offset = 1.5
        
        test_file = os.path.join(self.temp_dir, 'offset_test.tmd')
        write_tmd_file(
            self.test_height_map,
            test_file,
            x_offset=x_offset,
            y_offset=y_offset
        )
        
        # Process with original offsets
        metadata, height_map = process_tmd_file(test_file)
        self.assertEqual(metadata['x_offset'], x_offset)
        self.assertEqual(metadata['y_offset'], y_offset)
        
        # Process with force_offset
        new_offsets = (3.0, 2.5)
        metadata_forced, _ = process_tmd_file(
            test_file, force_offset=new_offsets
        )
        self.assertEqual(metadata_forced['x_offset'], new_offsets[0])
        self.assertEqual(metadata_forced['y_offset'], new_offsets[1])
    
    def test_file_not_found_error(self):
        """Test appropriate errors are raised for missing files."""
        non_existent_file = os.path.join(self.temp_dir, 'doesnt_exist.tmd')
        
        with self.assertRaises(FileNotFoundError):
            process_tmd_file(non_existent_file)
            
        with self.assertRaises(FileNotFoundError):
            detect_tmd_version(non_existent_file)
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists', return_value=True)
    def test_invalid_file_handling(self, mock_exists, mock_file):
        """Test handling of invalid or corrupt TMD files."""
        # Setup mock to return invalid data
        mock_file.return_value.read.return_value = b'Not a TMD file' + b'\0' * 100
        
        # Test with invalid file
        metadata, height_map = process_tmd_file('invalid.tmd')
        
        # Should return default values for invalid file
        self.assertEqual(metadata['width'], 1)
        self.assertEqual(metadata['height'], 1)
        self.assertEqual(height_map.shape, (1, 1))


if __name__ == '__main__':
    unittest.main()
