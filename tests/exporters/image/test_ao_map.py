"""Unit tests for TMD ao map module."""

import unittest
from unittest.mock import patch, MagicMock
import os
import numpy as np
import tempfile
import shutil
import matplotlib

from tmd.exporters.image.ao_map import (
    convert_heightmap_to_ao_map,
    create_ambient_occlusion_map,
    export_ambient_occlusion
)


class TestAoMap(unittest.TestCase):
    """Test class for ambient occlusion map functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a simple height map for testing
        # 5x5 grid with a central peak
        self.simple_height_map = np.array([
            [0.1, 0.2, 0.3, 0.2, 0.1],
            [0.2, 0.4, 0.5, 0.4, 0.2],
            [0.3, 0.5, 1.0, 0.5, 0.3],
            [0.2, 0.4, 0.5, 0.4, 0.2],
            [0.1, 0.2, 0.3, 0.2, 0.1]
        ], dtype=np.float32)
        
        # Create a more complex height map with NaN values
        self.complex_height_map = np.array([
            [0.1, 0.2, 0.3, np.nan, 0.1],
            [0.2, 0.4, 0.5, 0.4, 0.2],
            [0.3, 0.5, 1.0, 0.5, 0.3],
            [0.2, 0.4, np.nan, 0.4, 0.2],
            [0.1, 0.2, 0.3, 0.2, 0.1]
        ], dtype=np.float32)
        
        # Output file paths
        self.output_file = os.path.join(self.temp_dir, "ao_map.png")
        
        # Use non-interactive backend for matplotlib
        matplotlib.use('Agg')
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary directory and its contents
        shutil.rmtree(self.temp_dir)
    
    def test_create_ambient_occlusion_map_simple(self):
        """Test creating ambient occlusion map from simple height map."""
        ao_map = create_ambient_occlusion_map(
            self.simple_height_map,
            strength=1.0,
            samples=8,
            radius=1.0
        )
        
        # Basic validation
        self.assertIsInstance(ao_map, np.ndarray)
        self.assertEqual(ao_map.shape, self.simple_height_map.shape)
        
        # AO map values should be between 0 and 1
        self.assertTrue(np.all(ao_map >= 0) and np.all(ao_map <= 1))
        
        # For the peak sample, check if the center pixel has lower AO value
        # than corner pixels (center is higher, so should be more occluded)
        center_val = float(ao_map[2, 2])
        edge_vals = [float(ao_map[0, 0]), float(ao_map[0, 4]), 
                    float(ao_map[4, 0]), float(ao_map[4, 4])]
        
        # Test using comparison of specific values instead of mean
        self.assertLess(center_val, edge_vals[0])
    
    def test_create_ambient_occlusion_map_with_nans(self):
        """Test creating ambient occlusion map with NaN values in height map."""
        ao_map = create_ambient_occlusion_map(
            self.complex_height_map,
            strength=1.0,
            samples=8,
            radius=1.0
        )
        
        # Validate shape and type
        self.assertIsInstance(ao_map, np.ndarray)
        self.assertEqual(ao_map.shape, self.complex_height_map.shape)
        
        # Check for NaN handling
        # NaN values in height map should not cause NaN in AO map
        self.assertFalse(np.any(np.isnan(ao_map)))
    
    def test_create_ambient_occlusion_map_strength(self):
        """Test effect of strength parameter on ambient occlusion map."""
        # Create AO maps with different strength values
        ao_map_low = create_ambient_occlusion_map(
            self.simple_height_map,
            strength=0.5,
            samples=8,
            radius=1.0
        )
        
        ao_map_high = create_ambient_occlusion_map(
            self.simple_height_map,
            strength=2.0,
            samples=8,
            radius=1.0
        )
        
        # Higher strength should lead to darker (more contrast) AO
        # Average value of low strength should be higher (lighter)
        self.assertGreater(np.nanmean(ao_map_low), np.nanmean(ao_map_high))
    
    def test_convert_heightmap_to_ao_map_without_file(self):
        """Test converting height map to ambient occlusion map without file output."""
        ao_map = convert_heightmap_to_ao_map(
            self.simple_height_map,
            filename=None,
            samples=8,
            intensity=1.0,
            radius=1.0
        )
        
        # Validate result
        self.assertIsInstance(ao_map, np.ndarray)
        self.assertEqual(ao_map.shape, self.simple_height_map.shape)
    
    @patch('tmd.exporters.image.ao_map.export_ambient_occlusion')
    def test_convert_heightmap_to_ao_map_with_file(self, mock_export):
        """Test converting height map to ambient occlusion map with file output."""
        # Setup mock to return output file path
        mock_export.return_value = self.output_file
        
        result = convert_heightmap_to_ao_map(
            self.simple_height_map,
            filename=self.output_file,
            samples=8,
            intensity=1.0,
            radius=1.0
        )
        
        # Verify export function was called with correct parameters
        mock_export.assert_called_once()
        _, kwargs = mock_export.call_args
        
        # Check arguments manually since numpy arrays can't be directly compared
        self.assertTrue(np.array_equal(kwargs['height_map'], self.simple_height_map))
        self.assertEqual(kwargs['filename'], self.output_file)
        self.assertEqual(kwargs['strength'], 1.0)
        self.assertEqual(kwargs['samples'], 8)
        self.assertEqual(kwargs['radius'], 1.0)
        
        # Check return value
        self.assertEqual(result, self.output_file)
    
    @patch('matplotlib.pyplot.imsave')
    def test_export_ambient_occlusion(self, mock_imsave):
        """Test exporting ambient occlusion map to file."""
        # Create a mock AO map for testing
        mock_ao_map = np.ones((5, 5))
        
        # Mock ensure_directory_exists and create_ambient_occlusion_map
        with patch('tmd.exporters.image.utils.ensure_directory_exists') as mock_ensure_dir:
            with patch('tmd.exporters.image.ao_map.create_ambient_occlusion_map', return_value=mock_ao_map):
                result = export_ambient_occlusion(
                    self.simple_height_map,
                    self.output_file,
                    strength=1.0,
                    samples=8,
                    radius=0.1,
                    cmap='gray'
                )
                
                # Directory should be created
                mock_ensure_dir.assert_called_once_with(self.output_file)
                
                # imsave should be called with right parameters
                mock_imsave.assert_called_once()
                args, kwargs = mock_imsave.call_args
                self.assertEqual(args[0], self.output_file)
                self.assertEqual(kwargs.get('cmap'), 'gray')
                
                # Filename should be returned
                self.assertEqual(result, self.output_file)
    
    @patch('matplotlib.pyplot.imsave')
    def test_export_ambient_occlusion_error_handling(self, mock_imsave):
        """Test error handling in export function."""
        # Make imsave raise an exception
        mock_imsave.side_effect = Exception("Test error")
        
        # Mock the directory creation and AO map creation
        with patch('tmd.exporters.image.utils.ensure_directory_exists'):
            with patch('tmd.exporters.image.ao_map.create_ambient_occlusion_map', return_value=np.ones((5, 5))):
                with self.assertRaises(Exception) as context:
                    export_ambient_occlusion(
                        self.simple_height_map,
                        self.output_file
                    )
                
                # Verify the error message
                self.assertTrue("Test error" in str(context.exception))
    
    def test_actual_file_export(self):
        """Integration test for actual file export."""
        try:
            # Skip if matplotlib's imsave isn't working
            with patch('matplotlib.pyplot.imsave') as mock_imsave:
                mock_imsave.side_effect = ImportError("Test skip")
                with self.assertRaises(ImportError):
                    export_ambient_occlusion(self.simple_height_map, self.output_file)
                self.skipTest("Matplotlib imsave not working properly")
        except:
            # Continue with the actual test
            pass
            
        try:
            result = export_ambient_occlusion(
                self.simple_height_map,
                self.output_file,
                strength=1.0,
                samples=8,
                radius=0.1
            )
            
            # File should exist
            self.assertTrue(os.path.exists(self.output_file))
            
            # Result should be the file path
            self.assertEqual(result, self.output_file)
            
        except ImportError:
            # Skip if matplotlib is not available
            self.skipTest("Matplotlib not available for testing")


if __name__ == '__main__':
    unittest.main()
