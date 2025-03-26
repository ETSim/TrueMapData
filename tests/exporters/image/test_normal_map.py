"""Unit tests for TMD normal map module."""

import unittest
from unittest.mock import patch, MagicMock
import os
import numpy as np
import tempfile
import shutil

from tmd.exporters.image.normal_map import (
    create_normal_map,
    export_normal_map,
    normal_map_to_rgb
)
from tests.resources import create_sample_height_map


class TestNormalMap(unittest.TestCase):
    """Test class for normal map functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample height maps
        self.sample_height_map = create_sample_height_map(pattern="peak")
        self.sample_height_map_with_nan = create_sample_height_map(pattern="with_nan")
        self.sample_ramp = create_sample_height_map(pattern="slope")
        
        # Test output file
        self.output_file = os.path.join(self.temp_dir, "normal_map.png")
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary directory and its contents
        shutil.rmtree(self.temp_dir)
    
    def test_create_normal_map_basic(self):
        """Test creating a normal map from a height map."""
        normal_map = create_normal_map(self.sample_height_map)
        
        # Check shape and type
        self.assertEqual(normal_map.shape, self.sample_height_map.shape + (3,))
        self.assertEqual(normal_map.dtype, np.float32)
        
        # Check value ranges (should be in [-1, 1])
        self.assertTrue(np.all(normal_map >= -1.0) and np.all(normal_map <= 1.0))
        
        # Normal vectors should have unit length
        norms = np.sqrt(np.sum(normal_map**2, axis=2))
        self.assertTrue(np.allclose(norms, 1.0, rtol=1e-5, atol=1e-5))
        
        # Top of the peak should have normal pointing up
        # For a peak, center should have small x,y components and large z component
        center_normal = normal_map[2, 2]
        self.assertAlmostEqual(center_normal[2], 1.0, delta=0.1)
    
    def test_create_normal_map_with_nan(self):
        """Test creating a normal map from a height map with NaN values."""
        normal_map = create_normal_map(self.sample_height_map_with_nan)
        
        # Check no NaNs in output
        self.assertFalse(np.any(np.isnan(normal_map)))
        
        # Normal vectors should have unit length
        norms = np.sqrt(np.sum(normal_map**2, axis=2))
        self.assertTrue(np.allclose(norms, 1.0, rtol=1e-5, atol=1e-5))
    
    def test_create_normal_map_z_scale(self):
        """Test the effect of z_scale parameter on normal maps."""
        # Create normal maps with different z_scale values
        normal_low_z = create_normal_map(self.sample_ramp, z_scale=0.1)
        normal_high_z = create_normal_map(self.sample_ramp, z_scale=10.0)
        
        # Higher z_scale should make normals more varied
        # Calculate variance in x and y components
        var_low_z = np.var(normal_low_z[:, :, :2])
        var_high_z = np.var(normal_high_z[:, :, :2])
        
        self.assertGreater(var_high_z, var_low_z)
    
    def test_create_normal_map_output_formats(self):
        """Test different output formats."""
        # Create normal maps with different output formats
        normal_rgb = create_normal_map(self.sample_height_map, output_format="rgb")
        normal_xyz = create_normal_map(self.sample_height_map, output_format="xyz")
        
        # Both should produce valid normal maps
        self.assertEqual(normal_rgb.shape, self.sample_height_map.shape + (3,))
        self.assertEqual(normal_xyz.shape, self.sample_height_map.shape + (3,))
    
    def test_invalid_input(self):
        """Test handling of invalid inputs."""
        # Test with invalid dimensions
        with self.assertRaises(ValueError):
            create_normal_map(np.zeros((3, 3, 3)))
    
    @patch('matplotlib.pyplot.imsave')
    @patch('tmd.exporters.image.utils.ensure_directory_exists')
    def test_export_normal_map(self, mock_ensure_dir, mock_imsave):
        """Test exporting normal map."""
        result = export_normal_map(
            self.sample_height_map,
            self.output_file,
            z_scale=1.0
        )
        
        # Check that directory was created
        mock_ensure_dir.assert_called_once_with(self.output_file)
        
        # Check that imsave was called
        mock_imsave.assert_called_once()
        
        # Check return value
        self.assertEqual(result, self.output_file)
    
    @patch('matplotlib.pyplot.imsave')
    @patch('tmd.exporters.image.utils.ensure_directory_exists')
    def test_export_normal_map_with_options(self, mock_ensure_dir, mock_imsave):
        """Test exporting normal map with various options."""
        # Test with additional kwargs
        export_normal_map(
            self.sample_height_map,
            self.output_file,
            z_scale=2.0,
            output_format="xyz",
            normalize=False,
            cmap='viridis'  # Extra parameter for plt.imsave
        )
        
        # Check that imsave was called with right parameters
        args, kwargs = mock_imsave.call_args
        self.assertEqual(args[0], self.output_file)
        self.assertEqual(kwargs.get('cmap'), 'viridis')
    
    @patch('PIL.Image.fromarray')
    @patch('matplotlib.pyplot.imsave', side_effect=ImportError("No matplotlib"))
    def test_export_normal_map_fallback(self, mock_imsave, mock_pil):
        """Test fallback to PIL if matplotlib is not available."""
        # Set up PIL mock
        mock_image = MagicMock()
        mock_pil.return_value = mock_image
        
        # Call export
        export_normal_map(self.sample_height_map, self.output_file)
        
        # Check that PIL was used
        mock_pil.assert_called_once()
        mock_image.save.assert_called_once_with(self.output_file)
    
    def test_normal_map_to_rgb(self):
        """Test converting normal map to RGB."""
        # Create a test normal map
        normal_map = create_normal_map(self.sample_height_map)
        
        # Convert to RGB
        rgb_map = normal_map_to_rgb(normal_map)
        
        # Check type and range
        self.assertEqual(rgb_map.dtype, np.uint8)
        self.assertTrue(np.all(rgb_map >= 0) and np.all(rgb_map <= 255))


if __name__ == '__main__':
    unittest.main()
