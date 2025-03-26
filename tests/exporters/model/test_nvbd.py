"""Unit tests for TMD NVBD module."""

import unittest
import numpy as np
import os
import sys
import struct
import tempfile
from unittest.mock import patch, MagicMock

# Add the project root to the path to import tmd modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from tmd.exporters.model.nvbd import convert_heightmap_to_nvbd
from tmd.exporters.model.mesh_utils import calculate_heightmap_normals


class TestNvbd(unittest.TestCase):
    """Test class for NVBD export functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test heightmaps
        self.heightmap_flat = np.zeros((10, 10), dtype=np.float32)
        
        self.heightmap_slope = np.zeros((10, 10), dtype=np.float32)
        for i in range(10):
            for j in range(10):
                self.heightmap_slope[i, j] = i / 10.0
        
        self.heightmap_peak = np.zeros((10, 10), dtype=np.float32)
        for i in range(10):
            for j in range(10):
                self.heightmap_peak[i, j] = 1.0 - ((i-5)**2 + (j-5)**2) / 50.0
                if self.heightmap_peak[i, j] < 0:
                    self.heightmap_peak[i, j] = 0
    
    def test_input_validation(self):
        """Test input validation for NVBD export."""
        # Test with None heightmap
        result = convert_heightmap_to_nvbd(None, "test.nvbd")
        self.assertIsNone(result)
        
        # Test with empty heightmap
        result = convert_heightmap_to_nvbd(np.array([]), "test.nvbd")
        self.assertIsNone(result)
        
        # Test with invalid chunk size
        result = convert_heightmap_to_nvbd(self.heightmap_flat, "test.nvbd", chunk_size=0)
        self.assertIsNone(result)
    
    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    @patch('tmd.exporters.model.mesh_utils.ensure_directory_exists')
    def test_basic_nvbd_export(self, mock_ensure_dir, mock_open):
        """Test basic NVBD export functionality."""
        # Setup mock
        mock_ensure_dir.return_value = True
        mock_file = mock_open.return_value.__enter__.return_value
        
        # Test export
        result = convert_heightmap_to_nvbd(
            self.heightmap_flat,
            filename="test.nvbd",
            scale=1.0,
            chunk_size=8,
            include_normals=False
        )
        
        # Check that directory was created
        mock_ensure_dir.assert_called_once_with("test.nvbd")
        
        # Check that file was opened in binary write mode
        mock_open.assert_called_once_with("test.nvbd", 'wb')
        
        # Check that the result is the expected filename
        self.assertEqual(result, "test.nvbd")
        
        # Check that magic header was written
        write_calls = mock_file.write.call_args_list
        self.assertEqual(write_calls[0][0][0], b'NVBD')  # Magic header
    
    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    @patch('tmd.exporters.model.mesh_utils.ensure_directory_exists')
    @patch('tmd.exporters.model.nvbd.calculate_heightmap_normals')
    def test_nvbd_with_normals(self, mock_calc_normals, mock_ensure_dir, mock_open):
        """Test NVBD export with normals."""
        # Setup mocks
        mock_ensure_dir.return_value = True
        mock_file = mock_open.return_value.__enter__.return_value
        
        # Create mock normals that will be easy to verify
        mock_normals = np.zeros((10, 10, 3), dtype=np.float32)
        mock_normals[:, :, 2] = 1.0  # All normals point straight up
        mock_calc_normals.return_value = mock_normals
        
        # Test with normals included
        result = convert_heightmap_to_nvbd(
            self.heightmap_peak,
            filename="test.nvbd",
            scale=2.0,
            include_normals=True
        )
        
        # Check that normal calculation function was called
        mock_calc_normals.assert_called_once_with(self.heightmap_peak)
        
        # Verify normal count was written
        normal_count_bytes = None
        normal_data_written = False
        
        # Search for normal count (should be 10*10=100)
        for call in mock_file.write.call_args_list:
            data = call[0][0]
            if len(data) == 4:  # 4-byte integer
                try:
                    value = struct.unpack('<I', data)[0]
                    if value == 100:  # 10x10 = 100 normals
                        normal_count_bytes = data
                        break
                except:
                    pass
        
        self.assertIsNotNone(normal_count_bytes, "Normal count should be written")
        
        # Check for normal data (triplets of floats, all with z=1)
        for call in mock_file.write.call_args_list:
            data = call[0][0]
            if len(data) == 12:  # 3 floats = 12 bytes
                try:
                    nx, ny, nz = struct.unpack('<fff', data)
                    if nz == 1.0:  # Our mock normals have z=1
                        normal_data_written = True
                        break
                except:
                    pass
        
        self.assertTrue(normal_data_written, "Normal data should be written")
    
    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    @patch('tmd.exporters.model.mesh_utils.ensure_directory_exists')
    def test_watertight_flag(self, mock_ensure_dir, mock_open):
        """Test watertight flag setting."""
        # Setup mock
        mock_ensure_dir.return_value = True
        mock_file = mock_open.return_value.__enter__.return_value
        
        # Test with watertight=True
        convert_heightmap_to_nvbd(
            self.heightmap_flat,
            filename="test.nvbd",
            watertight=True
        )
        
        # Track if chunk flags were written with value 1
        flag_value_written = False
        
        # Search for flags with value 1
        for call in mock_file.write.call_args_list:
            data = call[0][0]
            if len(data) == 4:  # 4-byte integer
                try:
                    value = struct.unpack('<I', data)[0]
                    if value == 1:  # Flag is 1 for watertight
                        flag_value_written = True
                        break
                except struct.error:
                    pass
                    
        self.assertTrue(flag_value_written, "Watertight flag should be written")
        
        # Test with watertight=False
        mock_file.reset_mock()
        mock_open.reset_mock()
        mock_ensure_dir.reset_mock()
        mock_ensure_dir.return_value = True
        
        convert_heightmap_to_nvbd(
            self.heightmap_flat,
            filename="test2.nvbd",
            watertight=False
        )
        
        # Verify that a zero flag was written
        watertight_flags_found = []
        
        # Count watertight flags (0=non-watertight, 1=watertight)
        for call in mock_file.write.call_args_list:
            data = call[0][0]
            if len(data) == 4:  # 4-byte integer
                try:
                    value = struct.unpack('<I', data)[0]
                    if value in [0, 1]:  # Possible flag values
                        watertight_flags_found.append(value)
                except:
                    pass
        
        # Should find at least one flag with value 0
        self.assertIn(0, watertight_flags_found, "Non-watertight flag (0) should be written")
    
    @patch('tmd.exporters.model.mesh_utils.ensure_directory_exists')
    def test_directory_creation_failure(self, mock_ensure_dir):
        """Test handling of directory creation failure."""
        mock_ensure_dir.return_value = False
        
        result = convert_heightmap_to_nvbd(
            self.heightmap_flat,
            filename="test.nvbd"
        )
        
        # Should return None if directory creation fails
        self.assertIsNone(result)
        mock_ensure_dir.assert_called_once_with("test.nvbd")
    
    @patch('builtins.open')
    @patch('tmd.exporters.model.mesh_utils.ensure_directory_exists')
    def test_error_handling(self, mock_ensure_dir, mock_open):
        """Test error handling in NVBD export."""
        mock_ensure_dir.return_value = True
        
        # Make open raise an exception
        mock_open.side_effect = IOError("Test error")
        
        # Test export with IO error
        result = convert_heightmap_to_nvbd(
            self.heightmap_flat,
            filename="test.nvbd"
        )
        
        # Should return None on error
        self.assertIsNone(result)
    
    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    @patch('tmd.exporters.model.mesh_utils.ensure_directory_exists')
    def test_extension_handling(self, mock_ensure_dir, mock_open):
        """Test file extension handling."""
        mock_ensure_dir.return_value = True
        
        # Test with missing extension
        result = convert_heightmap_to_nvbd(
            self.heightmap_flat,
            filename="test",  # No extension
            scale=1.0
        )
        
        # Check that extension was added
        self.assertEqual(result, "test.nvbd")
        self.assertTrue(mock_open.call_args[0][0].endswith('.nvbd'))
    
    @unittest.skipIf(not os.getenv('RUN_INTEGRATION_TESTS'), "Integration tests disabled")
    def test_integration_real_file(self):
        """Test actual file creation (only runs if RUN_INTEGRATION_TESTS is set)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_file = os.path.join(tmp_dir, "test_real_output.nvbd")
            
            # Test NVBD output
            result = convert_heightmap_to_nvbd(
                self.heightmap_peak,
                filename=output_file,
                scale=1.0,
                include_normals=True
            )
            
            # Check that file exists
            self.assertTrue(os.path.isfile(output_file))
            
            # Basic check of NVBD format
            with open(output_file, 'rb') as f:
                header = f.read(4)
                self.assertEqual(header, b'NVBD')  # Check magic signature


if __name__ == '__main__':
    unittest.main()
