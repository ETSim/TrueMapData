"""Unit tests for TMD utils module."""

import unittest
from unittest.mock import patch, mock_open, MagicMock
import os
import numpy as np
import tempfile
import shutil

from tmd.utils.utils import (
    hexdump,
    read_null_terminated_string,
    detect_tmd_version,
    process_tmd_file,
    write_tmd_file,
    create_sample_height_map,
    generate_synthetic_tmd
)


class TestUtils(unittest.TestCase):
    """Test class for TMD utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test data
        self.test_bytes = b'Hello, World! This is test data.'
        self.test_string_with_null = b'Test\x00String'
        
        # Create sample height map for testing
        self.test_height_map = create_sample_height_map(
            width=20, height=15, pattern='waves', noise_level=0.0
        )
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary directory and its contents
        shutil.rmtree(self.temp_dir)
    
    def test_hexdump(self):
        """Test hexdump function."""
        # Test with simple input
        dump = hexdump(self.test_bytes)
        
        # Check that the output is a string with expected content
        self.assertIsInstance(dump, str)
        self.assertIn('48 65 6c 6c 6f', dump)  # Hex for "Hello"
        self.assertIn('|Hello', dump)  # ASCII representation
        
        # Test with custom parameters
        dump_custom = hexdump(self.test_bytes, start=2, length=5, width=8, show_ascii=False)
        
        # Check that parameters are respected
        self.assertIsInstance(dump_custom, str)
        self.assertNotIn('|Hello', dump_custom)  # ASCII should be omitted
        self.assertIn('00000002', dump_custom)  # Start offset
    
    def test_read_null_terminated_string(self):
        """Test reading null-terminated strings."""
        # Create a mock file with a null-terminated string
        mock_file = MagicMock()
        mock_file.tell.return_value = 0
        mock_file.read.return_value = self.test_string_with_null
        
        # Read the string
        result = read_null_terminated_string(mock_file)
        
        # Check result
        self.assertEqual(result, 'Test')
        mock_file.seek.assert_called_once_with(5)  # Position after null byte
        
        # Test with string without null terminator
        mock_file.reset_mock()
        mock_file.tell.return_value = 0
        mock_file.read.return_value = b'NoNullHere'
        
        # Should return whole string
        result = read_null_terminated_string(mock_file)
        self.assertEqual(result, 'NoNullHere')
        mock_file.seek.assert_not_called()  # Should not seek
    
    def test_detect_tmd_version(self):
        """Test TMD version detection."""
        # Create a mock v2 TMD file
        v2_header = b'Binary TrueMap Data File v2.0\x00' + b'\x00' * 32
        v1_header = b'Binary TrueMap Data File\x00' + b'\x00' * 32
        
        # Test with different headers
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=v2_header)):
                self.assertEqual(detect_tmd_version('dummy.tmd'), 2)
            
            with patch('builtins.open', mock_open(read_data=v1_header)):
                self.assertEqual(detect_tmd_version('dummy.tmd'), 1)
        
        # Test file not found
        with self.assertRaises(FileNotFoundError):
            detect_tmd_version('nonexistent.tmd')
    
    def test_create_sample_height_map(self):
        """Test creating sample height maps."""
        # Test different patterns
        patterns = ['waves', 'peak', 'dome', 'ramp', 'combined']
        
        for pattern in patterns:
            height_map = create_sample_height_map(width=50, height=40, pattern=pattern)
            
            # Check shape and type
            self.assertEqual(height_map.shape, (40, 50))
            self.assertEqual(height_map.dtype, np.float32)
            
            # Check value range (should be normalized to [0,1])
            self.assertGreaterEqual(np.min(height_map), 0.0)
            self.assertLessEqual(np.max(height_map), 1.0)
        
        # Test adding noise
        noisy_map = create_sample_height_map(width=20, height=20, pattern='peak', noise_level=0.2)
        quiet_map = create_sample_height_map(width=20, height=20, pattern='peak', noise_level=0.0)
        
        # Maps with noise should be different
        self.assertFalse(np.array_equal(noisy_map, quiet_map))
    
    def test_write_and_process_tmd(self):
        """Test writing and reading a TMD file."""
        # Create a simple test height map
        test_map = np.ones((5, 5), dtype=np.float32) * 0.5
        
        # Add a pattern to make it easier to verify orientation
        test_map[0, 0] = 0.1  # Top-left
        test_map[0, 4] = 0.2  # Top-right
        test_map[4, 0] = 0.3  # Bottom-left
        test_map[4, 4] = 0.4  # Bottom-right
        
        # Path for the test file
        test_file = os.path.join(self.temp_dir, 'test.tmd')
        
        # Write the test file
        write_tmd_file(
            test_map,
            test_file,
            comment="Test file",
            x_length=10.0,
            y_length=10.0,
            version=2
        )
        
        # Check file exists
        self.assertTrue(os.path.exists(test_file))
        
        # Read the file back
        metadata, height_map = process_tmd_file(test_file)
        
        # Verify metadata
        self.assertEqual(metadata['width'], 5)
        self.assertEqual(metadata['height'], 5)
        self.assertEqual(metadata['x_length'], 10.0)
        self.assertEqual(metadata['y_length'], 10.0)
        
        # Verify corner values to check orientation
        self.assertAlmostEqual(height_map[0, 0], test_map[0, 0])  # Top-left
        self.assertAlmostEqual(height_map[0, 4], test_map[0, 4])  # Top-right
        self.assertAlmostEqual(height_map[4, 0], test_map[4, 0])  # Bottom-left
        self.assertAlmostEqual(height_map[4, 4], test_map[4, 4])  # Bottom-right
    
    def test_generate_synthetic_tmd(self):
        """Test generation of synthetic TMD files."""
        # Generate a synthetic file
        output_path = os.path.join(self.temp_dir, "synthetic.tmd")
        result_path = generate_synthetic_tmd(
            output_path=output_path,
            width=30,
            height=25,
            pattern="peak",
            comment="Synthetic test"
        )
        
        # Verify file was created with correct path
        self.assertEqual(result_path, output_path)
        self.assertTrue(os.path.exists(output_path))
        
        # Read file and verify dimensions
        metadata, height_map = process_tmd_file(output_path)
        self.assertEqual(metadata['width'], 30)
        self.assertEqual(metadata['height'], 25)
    
    def test_process_tmd_with_offsets(self):
        """Test processing TMD files with spatial offsets."""
        # Create a test height map
        test_map = create_sample_height_map(width=10, height=10, pattern='peak')
        
        # Path for the test file
        test_file = os.path.join(self.temp_dir, 'offset_test.tmd')
        
        # Write file with offsets
        x_offset, y_offset = 2.5, 1.5
        write_tmd_file(
            test_map,
            test_file,
            x_offset=x_offset,
            y_offset=y_offset
        )
        
        # Process with original offsets
        metadata, _ = process_tmd_file(test_file)
        self.assertEqual(metadata['x_offset'], x_offset)
        self.assertEqual(metadata['y_offset'], y_offset)
        
        # Process with force_offset
        new_offsets = (3.5, 4.5)
        metadata_forced, _ = process_tmd_file(test_file, force_offset=new_offsets)
        self.assertEqual(metadata_forced['x_offset'], new_offsets[0])
        self.assertEqual(metadata_forced['y_offset'], new_offsets[1])


if __name__ == '__main__':
    unittest.main()
