""".

Unit tests for the image exporter functionality in the sequence module.

Tests the ImageExporter class and related functionality.
"""

import os
import tempfile
import unittest
import numpy as np
from unittest.mock import patch, MagicMock

from tmd.sequence.exporters.image import ImageExporter
from tmd.utils.utils import create_sample_height_map

class TestImageExporter(unittest.TestCase):
    """Test cases for the ImageExporter class.."""
    
    def setUp(self):
        """Set up test fixtures.."""
        self.temp_dir = tempfile.mkdtemp()
        self.exporter = ImageExporter()
        
        # Create sample heightmaps
        self.height_map1 = create_sample_height_map(64, 64, 'sine')
        self.height_map2 = create_sample_height_map(64, 64, 'cosine')
        self.frames = [self.height_map1, self.height_map2]
        self.timestamps = ["Frame 1", "Frame 2"]
    
    def tearDown(self):
        """Clean up test fixtures.."""
        # Remove temp directory and all files
        for root, dirs, files in os.walk(self.temp_dir, topdown=False):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))
        os.rmdir(self.temp_dir)
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_export_sequence(self, mock_close, mock_savefig):
        """Test exporting a sequence of frames as images.."""
        # Export the sequence
        result = self.exporter.export_sequence(
            frames_data=self.frames,
            output_dir=self.temp_dir,
            timestamps=self.timestamps,
            format="png",
            colormap="viridis",
            dpi=100
        )
        
        # Verify the export was called correctly
        self.assertEqual(len(result), 2)
        self.assertTrue(mock_savefig.called)
        self.assertEqual(mock_savefig.call_count, 2)
        self.assertTrue(mock_close.called)
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_export_sequence_differences(self, mock_close, mock_savefig):
        """Test exporting differences between frames as images.."""
        # Create some difference data
        diff_data = [self.height_map2 - self.height_map1]
        diff_timestamps = ["Frame 1 → Frame 2"]
        
        # Export the differences
        result = self.exporter.export_sequence_differences(
            frames_data=diff_data,
            output_dir=self.temp_dir,
            timestamps=diff_timestamps,
            format="png",
            normalize=True,
            colormap="RdBu",
            dpi=100
        )
        
        # Verify the export was called correctly
        self.assertEqual(len(result), 1)
        self.assertTrue(mock_savefig.called)
        self.assertTrue(mock_close.called)
    
    @patch('PIL.Image.fromarray')
    def test_export_normal_maps(self, mock_fromarray):
        """Test exporting normal maps from heightmaps.."""
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
                output_dir=self.temp_dir,
                timestamps=self.timestamps,
                format="png",
                z_scale=5.0
            )
            
            # Verify the export was called correctly
            self.assertEqual(len(result), 2)
            self.assertTrue(mock_fromarray.called)
            self.assertEqual(mock_fromarray.call_count, 2)
            # Verify the save method was called on the mock image
            self.assertEqual(mock_image.save.call_count, 2)
    
    def test_generate_normal_map(self):
        """Test generating a normal map from a heightmap.."""
        # Generate a normal map
        normal_map = self.exporter._generate_normal_map(self.height_map1, z_scale=5.0)
        
        # Verify the normal map has the correct shape and properties
        self.assertEqual(normal_map.shape, (*self.height_map1.shape, 3))
        self.assertTrue(np.all(normal_map >= -1.0) and np.all(normal_map <= 1.0))
        
        # Check that Z component is mostly positive (normal vectors should point upward)
        z_component = normal_map[:, :, 2]
        self.assertTrue(np.mean(z_component) > 0)

if __name__ == "__main__":
    unittest.main()
