"""Unit tests for TMD image core module."""

import unittest
from unittest.mock import patch, MagicMock
import os
import numpy as np
import tempfile
import shutil
import matplotlib.pyplot as plt

from tmd.exporters.image.core import (
    export_heightmap_image,
    export_normal_map,
    export_displacement_map,
    export_ambient_occlusion,
    batch_export_maps,
    _calculate_normal_map_numpy
)
from tests.resources import create_sample_height_map


class TestImageCore(unittest.TestCase):
    """Test class for image core functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Sample height maps for testing
        self.sample_height_map = create_sample_height_map(pattern="peak")
        self.sample_height_map_with_nan = create_sample_height_map(pattern="with_nan")
        
        # Output file paths
        self.basic_output = os.path.join(self.temp_dir, "heightmap.png")
        self.normal_output = os.path.join(self.temp_dir, "normal_map.png")
        self.displacement_output = os.path.join(self.temp_dir, "displacement.png")
        self.ao_output = os.path.join(self.temp_dir, "ao.png")
        
        # Set matplotlib to non-interactive mode for testing
        plt.switch_backend('Agg')
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary directory and its contents
        shutil.rmtree(self.temp_dir)
    
    @patch('matplotlib.pyplot.imsave')
    @patch('os.makedirs')
    def test_export_heightmap_image(self, mock_makedirs, mock_imsave):
        """Test exporting height map as an image file."""
        # Set up mock return value
        mock_imsave.return_value = None
        
        # Call the function
        result = export_heightmap_image(
            self.sample_height_map,
            self.basic_output,
            colormap='viridis',
            normalize=True,
            vmin=0.2,
            vmax=0.8
        )
        
        # Check that the correct functions were called
        mock_makedirs.assert_called_once()
        mock_imsave.assert_called_once()
        
        # Verify function arguments
        args, kwargs = mock_imsave.call_args
        self.assertEqual(args[0], self.basic_output)  # First arg is filename
        self.assertEqual(kwargs['cmap'], 'viridis')  # Check colormap passed
        self.assertEqual(result, self.basic_output)  # Check return value
    
    def test_export_heightmap_image_with_nans(self):
        """Test exporting height map with NaN values."""
        result = export_heightmap_image(
            self.sample_height_map_with_nan,
            self.basic_output,
            colormap='gray'
        )
        
        # Verify file was created
        self.assertTrue(os.path.exists(self.basic_output))
        self.assertEqual(result, self.basic_output)
    
    @patch('tmd.exporters.image.core.HAS_OPENCV', False)
    @patch('tmd.exporters.image.core._calculate_normal_map_numpy')
    def test_export_normal_map_without_opencv(self, mock_numpy_impl):
        """Test exporting normal map without OpenCV."""
        # Set up mock to return a simple normal map
        mock_numpy_impl.return_value = np.zeros((5, 5, 3), dtype=np.float32)
        
        with patch('matplotlib.pyplot.imsave') as mock_imsave:
            result = export_normal_map(
                self.sample_height_map,
                self.normal_output,
                strength=1.5,
                resolution=0.5
            )
            
            # Verify the correct implementation was used
            mock_numpy_impl.assert_called_once()
            
            # Check if matplotlib was used to save
            mock_imsave.assert_called_once()
            
            # Verify return value
            self.assertEqual(result, self.normal_output)
    
    def test_calculate_normal_map_numpy(self):
        """Test normal map calculation without OpenCV."""
        # Create a simple ramp for testing
        ramp = create_sample_height_map(pattern="slope")
        
        # Calculate normal map
        normal_map = _calculate_normal_map_numpy(ramp, strength=1.0)
        
        # Check shape and type
        self.assertEqual(normal_map.shape, ramp.shape + (3,))  # Should be (h, w, 3)
        self.assertEqual(normal_map.dtype, np.float32)
        
        # Check normal vector ranges (should be in [-1, 1])
        self.assertTrue(np.all(normal_map >= -1))
        self.assertTrue(np.all(normal_map <= 1))
        
        # For a slope, check gradient direction (just basic validation)
        # Z component should always be positive
        self.assertTrue(np.all(normal_map[..., 2] > 0))
    
    @patch('tmd.exporters.image.core.HAS_OPENCV', True)
    @patch('tmd.exporters.image.core._calculate_normal_map_cv2')
    def test_export_normal_map_with_opencv(self, mock_cv2_impl):
        """Test exporting normal map with OpenCV."""
        # Set up mock to return a simple normal map
        mock_cv2_impl.return_value = np.zeros((5, 5, 3), dtype=np.float32)
        
        with patch('matplotlib.pyplot.imsave'):
            export_normal_map(
                self.sample_height_map,
                self.normal_output
            )
            
            # Verify OpenCV implementation was used
            mock_cv2_impl.assert_called_once()
    
    @patch('tmd.exporters.image.core.HAS_OPENCV', False)
    def test_export_displacement_map_no_opencv(self):
        """Test exporting displacement map without OpenCV."""
        with patch('matplotlib.pyplot.imsave') as mock_imsave:
            result = export_displacement_map(
                self.sample_height_map,
                self.displacement_output,
                invert=True,
                bit_depth=8
            )
            
            # Check matplotlib was used
            mock_imsave.assert_called_once()
            
            # Check arguments
            args, kwargs = mock_imsave.call_args
            self.assertEqual(args[0], self.displacement_output)
            self.assertEqual(kwargs['cmap'], 'gray')
            
            # Verify return value
            self.assertEqual(result, self.displacement_output)
    
    @patch('tmd.exporters.image.core.HAS_OPENCV', True)
    @patch('cv2.imwrite')
    def test_export_displacement_map_with_opencv(self, mock_imwrite):
        """Test exporting displacement map with OpenCV."""
        mock_imwrite.return_value = True
        
        # Test 8-bit export
        export_displacement_map(
            self.sample_height_map,
            self.displacement_output,
            bit_depth=8
        )
        mock_imwrite.assert_called_once()
        
        # Reset mock and test 16-bit export
        mock_imwrite.reset_mock()
        export_displacement_map(
            self.sample_height_map,
            os.path.join(self.temp_dir, "16bit.png"),
            bit_depth=16
        )
        mock_imwrite.assert_called_once()
        
        # Check different bit depths were used (different array values)
        args1, _ = mock_imwrite.call_args
        self.assertEqual(args1[1].dtype, np.uint16)
    
    def test_export_ambient_occlusion(self):
        """Test exporting ambient occlusion map."""
        with patch('matplotlib.pyplot.imsave') as mock_imsave:
            with patch('tmd.exporters.image.core._calculate_ambient_occlusion',
                      return_value=np.ones((5, 5))) as mock_calc:
                
                result = export_ambient_occlusion(
                    self.sample_height_map,
                    self.ao_output,
                    strength=1.5,
                    samples=8
                )
                
                # Verify calculation function was called with correct params
                mock_calc.assert_called_once()
                args, kwargs = mock_calc.call_args
                self.assertEqual(args[1], 8)  # samples
                self.assertEqual(args[3], 1.5)  # strength
                
                # Verify image was saved
                mock_imsave.assert_called_once()
                
                # Check return value
                self.assertEqual(result, self.ao_output)
    
    def test_calculate_ambient_occlusion(self):
        """Test ambient occlusion calculation."""
        from tmd.exporters.image.core import _calculate_ambient_occlusion
        
        # Calculate AO for peak height map (peak should be darker/more occluded)
        ao_map = _calculate_ambient_occlusion(
            self.sample_height_map,
            samples=8,
            radius=0.2,
            strength=1.0
        )
        
        # Check shape and data range
        self.assertEqual(ao_map.shape, self.sample_height_map.shape)
        self.assertTrue(np.all(ao_map >= 0))
        self.assertTrue(np.all(ao_map <= 1))
    
    def test_batch_export_maps(self):
        """Test batch exporting of multiple map formats."""
        with patch('tmd.exporters.image.core.export_heightmap_image',
                  return_value="heightmap.png") as mock_height:
            with patch('tmd.exporters.image.core.export_normal_map',
                      return_value="normal.png") as mock_normal:
                with patch('tmd.exporters.image.core.export_displacement_map',
                          return_value="displacement.png") as mock_disp:
                    
                    # Test with default formats
                    result = batch_export_maps(
                        self.sample_height_map,
                        self.temp_dir,
                        base_name="test"
                    )
                    
                    # Check that all exporters were called
                    mock_height.assert_called()
                    mock_normal.assert_called()
                    mock_disp.assert_called()
                    
                    # Check result contains all formats
                    self.assertIn("heightmap", result)
                    self.assertIn("normal_map", result)
                    self.assertIn("displacement_map", result)
                    self.assertIn("colored_heightmap", result)
    
    def test_batch_export_selective(self):
        """Test batch export with selective formats."""
        with patch('tmd.exporters.image.core.export_heightmap_image',
                  return_value="heightmap.png") as mock_height:
            with patch('tmd.exporters.image.core.export_normal_map',
                      return_value="normal.png") as mock_normal:
                
                # Only export heightmap and normal map
                formats = {
                    "heightmap": True,
                    "normal_map": True,
                    "displacement_map": False,
                    "ambient_occlusion": False,
                    "colored_heightmap": False
                }
                
                result = batch_export_maps(
                    self.sample_height_map,
                    self.temp_dir,
                    base_name="test",
                    formats=formats
                )
                
                # Check only the selected exporters were called
                mock_height.assert_called_once()
                mock_normal.assert_called_once()
                
                # Check result contains only selected formats
                self.assertIn("heightmap", result)
                self.assertIn("normal_map", result)
                self.assertNotIn("displacement_map", result)
                self.assertNotIn("ambient_occlusion", result)
                self.assertNotIn("colored_heightmap", result)
    
    def test_integration_actual_batch_export(self):
        """Integration test with actual file creation."""
        # Use a small height map to keep the test fast
        small_map = create_sample_height_map(size=(10, 10), pattern="peak")
        
        # Selective formats for speed
        formats = {
            "heightmap": True,
            "normal_map": False,  # Skip to speed up test
            "displacement_map": True,
            "ambient_occlusion": False,  # Skip as it's slow
            "colored_heightmap": True
        }
        
        try:
            # Perform actual export
            result = batch_export_maps(
                small_map,
                self.temp_dir,
                base_name="integration_test",
                formats=formats
            )
            
            # Verify files were created
            self.assertTrue(os.path.exists(result["heightmap"]))
            self.assertTrue(os.path.exists(result["displacement_map"]))
            self.assertTrue(os.path.exists(result["colored_heightmap"]))
            
        except ImportError:
            self.skipTest("Required libraries not available")


if __name__ == '__main__':
    unittest.main()
