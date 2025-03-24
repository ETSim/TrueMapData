""".

Unit tests for the Matplotlib plotter implementation.

Tests functionality of the MatplotlibPlotter class.
"""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import os
import tempfile

from tmd.sequence.plotters.matplotlib import MatplotlibPlotter

# Custom exception for matplotlib import failures
class MockImportError(ImportError):
    pass

class TestMatplotlibPlotter(unittest.TestCase):
    """Test cases for MatplotlibPlotter class.."""
    
    def setUp(self):
        """Set up test fixtures.."""
        self.plotter = MatplotlibPlotter()
        self.test_height_maps = [
            np.zeros((10, 10)),
            np.ones((10, 10)),
            np.random.random((10, 10))
        ]
        self.test_timestamps = ["Frame 1", "Frame 2", "Frame 3"]
        self.test_stats = [
            {"min": 0.0, "max": 1.0, "mean": 0.5, "std": 0.2, "range": 1.0},
            {"min": 0.1, "max": 0.9, "mean": 0.4, "std": 0.15, "range": 0.8},
            {"min": 0.2, "max": 0.8, "mean": 0.5, "std": 0.1, "range": 0.6}
        ]
        
        # Create temp directory for output files
        self.temp_dir = tempfile.mkdtemp(prefix="tmd_test_matplotlib_")
    
    def tearDown(self):
        """Clean up test fixtures.."""
        # Clean up temp files
        for root, dirs, files in os.walk(self.temp_dir, topdown=False):
            for file in files:
                os.unlink(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))
        os.rmdir(self.temp_dir)
    
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_visualize_sequence_2d(self, mock_close, mock_savefig, mock_figure):
        """Test 2D visualization of sequence.."""
        # Setup mock returns
        mock_figure.return_value = MagicMock()
        
        # Call the method with 2D view
        result = self.plotter.visualize_sequence(
            height_maps=self.test_height_maps,
            timestamps=self.test_timestamps,
            view_type='2d',
            colorscale='viridis',
            show=False
        )
        
        # Verify matplotlib functions were called
        mock_figure.assert_called()
        self.assertIsNotNone(result)
    
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_visualize_sequence_3d(self, mock_close, mock_savefig, mock_figure):
        """Test 3D visualization of sequence.."""
        # Setup mock returns
        mock_figure.return_value = MagicMock()
        
        # Call the method with 3D view
        result = self.plotter.visualize_sequence(
            height_maps=self.test_height_maps,
            timestamps=self.test_timestamps,
            view_type='3d',
            colorscale='viridis',
            show=False
        )
        
        # Verify matplotlib functions were called
        mock_figure.assert_called()
        self.assertIsNotNone(result)
    
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_visualize_statistics(self, mock_close, mock_savefig, mock_figure):
        """Test statistics visualization.."""
        # Setup mock returns
        mock_figure.return_value = MagicMock()
        
        # Call the method
        result = self.plotter.visualize_statistics(
            stats=self.test_stats,
            timestamps=self.test_timestamps,
            show=False
        )
        
        # Verify matplotlib functions were called
        mock_figure.assert_called()
        self.assertIsNotNone(result)
    
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.animation.FuncAnimation')
    def test_create_animation(self, mock_animation, mock_close, mock_savefig, mock_figure):
        """Test animation creation.."""
        # Setup mock returns
        mock_figure.return_value = MagicMock()
        mock_animation.return_value = MagicMock()
        
        # Create a filename
        filename = os.path.join(self.temp_dir, "animation.gif")
        
        # Call the method
        result = self.plotter.create_animation(
            frames_data=self.test_height_maps,
            timestamps=self.test_timestamps,
            filename=filename,
            fps=2,
            show=False
        )
        
        # Verify matplotlib functions were called
        mock_figure.assert_called()
        mock_animation.assert_called()
        self.assertIsNotNone(result)
    
    @patch('matplotlib.pyplot')
    def test_save_figure(self, mock_plt):
        """Test saving figure.."""
        # Setup mock
        mock_figure = MagicMock()
        
        # Create a filename
        filename = os.path.join(self.temp_dir, "test_figure.png")
        
        # Call the method
        self.plotter._save_figure(mock_figure, filename)
        
        # Verify savefig was called
        mock_plt.savefig.assert_called_once()
        mock_plt.close.assert_called_once()
    
    @patch('tmd.sequence.plotters.matplotlib.plt', None)
    def test_check_dependencies_failure(self):
        """Test dependency check when matplotlib is not available.."""
        # This test simulates the scenario where matplotlib is not installed
        plotter = MatplotlibPlotter()
        result = plotter._check_dependencies()
        self.assertFalse(result)
    
    @patch('tmd.sequence.plotters.matplotlib.plt')
    @patch('tmd.sequence.plotters.matplotlib.mpl_toolkits')
    def test_check_dependencies_success(self, mock_mpl_toolkits, mock_plt):
        """Test dependency check when all dependencies are available.."""
        plotter = MatplotlibPlotter()
        result = plotter._check_dependencies()
        self.assertTrue(result)

if __name__ == "__main__":
    unittest.main()
