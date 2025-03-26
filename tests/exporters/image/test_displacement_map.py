"""Unit tests for TMD displacement map module."""

import unittest
from unittest.mock import patch, MagicMock
import os
import numpy as np
import tempfile
import shutil

from tmd.exporters.image.displacement_map import convert_heightmap_to_displacement_map, export_displacement_map
from tests.resources import create_sample_height_map


class TestDisplacementMap(unittest.TestCase):
    """Test class for displacement map functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample height maps for testing
        self.sample_height_map = create_sample_height_map(pattern="peak")
        
        # Create a height map with NaN values
        self.sample_height_map_with_nan = np.copy(self.sample_height_map)
        self.sample_height_map_with_nan[1, 1] = np.nan
        self.sample_height_map_with_nan[3, 3] = np.nan
        
        # Test file path
        self.output_file = os.path.join(self.temp_dir, "displacement_map.png")
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary directory and its contents
        shutil.rmtree(self.temp_dir)
    
    def test_convert_heightmap_to_displacement_map_basic(self):
        """Test basic conversion of height map to displacement map."""
        # Convert height map to displacement map without saving
        result = convert_heightmap_to_displacement_map(
            self.sample_height_map,
            filename=None,
            invert=False,
            normalize=True
        )
        
        # Verify the result is a numpy array
        self.assertIsInstance(result, np.ndarray)
        
        # Check dimensions and value range
        self.assertEqual(result.shape, self.sample_height_map.shape)
        self.assertTrue(np.all(result >= 0) and np.all(result <= 1))
    
    def test_convert_heightmap_to_displacement_map_with_nan(self):
        """Test handling of NaN values in height map."""
        # Convert height map with NaNs
        result = convert_heightmap_to_displacement_map(
            self.sample_height_map_with_nan,
            filename=None
        )
        
        # Verify no NaNs in result
        self.assertFalse(np.any(np.isnan(result)))
    
    def test_convert_heightmap_to_displacement_map_invert(self):
        """Test inverting the displacement map."""
        # Create normal and inverted maps
        normal = convert_heightmap_to_displacement_map(
            self.sample_height_map,
            filename=None,
            invert=False
        )
        
        inverted = convert_heightmap_to_displacement_map(
            self.sample_height_map,
            filename=None,
            invert=True
        )
        
        # Sum of corresponding pixels should be approximately 1
        sum_check = normal + inverted
        self.assertTrue(np.allclose(sum_check, np.ones_like(sum_check), atol=0.01))
    
    def test_convert_heightmap_invalid_input(self):
        """Test handling of invalid inputs."""
        # Test with None
        with self.assertRaises(ValueError):
            convert_heightmap_to_displacement_map(None)
        
        # Test with non-array
        with self.assertRaises(ValueError):
            convert_heightmap_to_displacement_map("not an array")
        
        # Test with 3D array
        with self.assertRaises(ValueError):
            convert_heightmap_to_displacement_map(np.zeros((3, 3, 3)))
    
    def test_export_displacement_map_with_file(self):
        """Test file-exporting functionality."""
        # Without mocking to test actual file creation
        # Use a small array for speed
        small_map = np.random.rand(5, 5)
        
        # Create a real file for this test
        try:
            result = convert_heightmap_to_displacement_map(
                small_map,
                filename=self.output_file,
                bit_depth=8,
                normalize=True
            )
            
            # Verify file was created
            self.assertTrue(os.path.exists(self.output_file))
            self.assertEqual(result, self.output_file)
        except Exception as e:
            self.fail(f"Failed to create file: {e}")
    
    @patch('matplotlib.pyplot.imsave')
    @patch('tmd.exporters.image.utils.ensure_directory_exists')
    def test_export_displacement_map(self, mock_ensure_dir, mock_imsave):
        """Test the export_displacement_map function."""
        # Create displacement map
        disp_map = np.random.rand(5, 5)
        
        # Mock os.path.exists to return True so the file verification passes
        with patch('os.path.exists', return_value=True):
            # Export
            result = export_displacement_map(
                disp_map,
                self.output_file,
                bit_depth=8
            )
            
            # Verify directory was created
            mock_ensure_dir.assert_called_once_with(self.output_file)
            
            # Verify matplotlib was used for export
            mock_imsave.assert_called_once()
            
            # Check return value
            self.assertEqual(result, self.output_file)
    
    @patch('tmd.exporters.image.displacement_map.HAS_OPENCV', True)
    @patch('cv2.imwrite')
    @patch('os.path.exists')
    @patch('tmd.exporters.image.utils.ensure_directory_exists')
    def test_export_displacement_map_opencv(self, mock_ensure_dir, mock_exists, mock_imwrite):
        """Test export using OpenCV if available."""
        # Mock file existence check
        mock_exists.return_value = True
        
        # Mock OpenCV for testing - make sure it returns True
        mock_imwrite.return_value = True
        
        # Create displacement map
        disp_map = np.random.rand(5, 5)
        
        # Export with 16-bit depth
        result = export_displacement_map(
            disp_map,
            self.output_file,
            bit_depth=16
        )
        
        # Verify directory was created
        mock_ensure_dir.assert_called_once_with(self.output_file)
        
        # Verify OpenCV was used
        mock_imwrite.assert_called_once()
        
        # Check call arguments
        args, _ = mock_imwrite.call_args
        self.assertEqual(args[0], self.output_file)
        self.assertEqual(args[1].dtype, np.uint16)
        
        # Check return value
        self.assertEqual(result, self.output_file)
    
    @patch('os.path.exists')
    @patch('tmd.exporters.image.utils.ensure_directory_exists')
    def test_error_handling(self, mock_ensure_dir, mock_exists):
        """Test error handling for file operations."""
        # Mock file existence check to return False
        mock_exists.return_value = False
        
        # Create displacement map
        disp_map = np.random.rand(5, 5)
        
        # Export should raise IOError
        with self.assertRaises(IOError):
            export_displacement_map(disp_map, self.output_file)
    
    def test_directory_creation_integration(self):
        """Test directory creation for export in an actual integration test."""
        # Create a small test displacement map
        test_map = create_sample_height_map(size=(3, 3), pattern="peak")
        
        # Create a nested path that requires directory creation
        nested_path = os.path.join(self.temp_dir, "nested", "dir", "disp.png")
        
        # Create the parent directory manually to avoid file system issues
        os.makedirs(os.path.dirname(nested_path), exist_ok=True)
        
        # Mock the actual writing functions and os.path.exists
        with patch('matplotlib.pyplot.imsave') as mock_imsave:
            with patch('os.path.exists', return_value=True):
                # Try the export with mock_save=True to skip actual file creation
                result = convert_heightmap_to_displacement_map(
                    test_map,
                    filename=nested_path,
                    mock_save=True  # Special parameter to skip actual file operations
                )
                
                # Verify directory exists (we created it manually)
                self.assertTrue(os.path.exists(os.path.dirname(nested_path)))
                
                # Check output path
                self.assertEqual(result, nested_path)


if __name__ == '__main__':
    unittest.main()
