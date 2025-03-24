"""
Tests for the image exporters module.
"""

import os
import tempfile
import unittest
import numpy as np
from unittest.mock import patch, MagicMock

from tmd.sequence.exporters.image import ImageExporter

class TestImageExporters(unittest.TestCase):
    """Test the image exporters."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.exporter = ImageExporter()
        
        # Create sample height maps
        self.height_map1 = np.zeros((64, 64), dtype=np.float32)
        self.height_map2 = np.zeros((64, 64), dtype=np.float32)
        
        # Add some features to the height maps
        x = np.linspace(-3, 3, 64)
        y = np.linspace(-3, 3, 64)
        X, Y = np.meshgrid(x, y)
        
        self.height_map1 = np.exp(-(X**2 + Y**2)/2)
        self.height_map2 = np.exp(-((X-1)**2 + (Y-1)**2)/2)
        
        # Create a list of frames
        self.frames = [self.height_map1, self.height_map2]
        self.timestamps = ["Frame 1", "Frame 2"]
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()
    
    @patch('matplotlib.pyplot.savefig')
    def test_export_images(self, mock_savefig):
        """Test exporting height maps as images."""
        # Set up the mock
        mock_savefig.return_value = None
        
        # Export images
        results = self.exporter.export_images(
            frames_data=self.frames,
            output_dir=self.temp_dir.name,
            timestamps=self.timestamps,
            format="png"
        )
        
        # Verify the export
        self.assertEqual(len(results), 2)
        self.assertTrue(mock_savefig.called)
    
    @patch('matplotlib.pyplot.savefig')
    def test_export_sequence_differences(self, mock_savefig):
        """Test exporting sequence differences."""
        # Set up the mock
        mock_savefig.return_value = None
        
        # Create difference frames
        diff1 = self.height_map2 - self.height_map1
        
        # Export differences
        results = self.exporter.export_sequence_differences(
            frames_data=[diff1],
            output_dir=self.temp_dir.name,
            timestamps=["Difference 1-2"],
            format="png",
            normalize=True,
            colormap="RdBu"
        )
        
        # Verify the export
        self.assertEqual(len(results), 1)
        self.assertTrue(mock_savefig.called)
    
    def test_normal_map(self):
        """Test generating a normal map from a heightmap."""
        # Generate a normal map
        normal_map = self.exporter._generate_normal_map(self.height_map1, z_scale=5.0)
        
        # Verify the normal map has the correct shape and properties
        self.assertEqual(normal_map.shape, (*self.height_map1.shape, 3))
        self.assertTrue(np.all(normal_map >= -1.0) and np.all(normal_map <= 1.0))
        
        # Check that Z component is mostly positive (normal vectors should point upward)
        z_component = normal_map[:, :, 2]
        self.assertTrue(np.mean(z_component) > 0)
    
    @patch('PIL.Image.fromarray')
    def test_export_normal_maps(self, mock_fromarray):
        """Test exporting normal maps."""
        # Mock PIL Image object and the save method
        mock_image = MagicMock()
        mock_fromarray.return_value = mock_image
        
        # Setup the mock for _generate_normal_map
        mock_normal_map = np.zeros((64, 64, 3), dtype=np.float32)
        mock_normal_map[:, :, 2] = 1.0  # Set Z component to 1
        
        with patch.object(self.exporter, '_generate_normal_map', return_value=mock_normal_map):
            # Export the normal maps
            result = self.exporter.export_normal_maps(
                frames_data=self.frames,
                output_dir=self.temp_dir.name,
                timestamps=self.timestamps
            )
            
            # Verify the export
            self.assertEqual(len(result), 2)
            self.assertTrue(mock_fromarray.called)
            self.assertTrue(mock_image.save.called)

if __name__ == "__main__":
    unittest.main()
