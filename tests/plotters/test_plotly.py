""".

Unit tests for the Plotly plotter functionality.

Tests the plotly-based visualization functions.
"""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import os
import tempfile

from tmd.plotters import plotly as plotly_module
from tmd.plotters.plotly import (
    create_surface_plot,
    create_heatmap,
    create_contour_plot,
    visualize_height_map,
    visualize_height_map_3d
)

class TestPlotlyPlotter(unittest.TestCase):
    """Test cases for Plotly plotter functionality.."""
    
    def setUp(self):
        """Set up test fixtures.."""
        # Create sample height map
        self.height_map = np.random.random((10, 10))
        
        # Create temp directory for output files
        self.temp_dir = tempfile.mkdtemp(prefix="tmd_test_plotly_")
    
    def tearDown(self):
        """Clean up test fixtures.."""
        # Clean up temp files
        for root, dirs, files in os.walk(self.temp_dir, topdown=False):
            for file in files:
                os.unlink(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))
        os.rmdir(self.temp_dir)
    
    @patch('plotly.graph_objects.Figure')
    @patch('plotly.graph_objects.Surface')
    def test_create_surface_plot(self, mock_surface, mock_figure):
        """Test creating a surface plot.."""
        # Setup mocks
        mock_figure.return_value = MagicMock()
        mock_surface.return_value = MagicMock()
        
        # Call the function
        fig = create_surface_plot(
            height_map=self.height_map,
            title="Test Surface Plot",
            colorscale="Viridis"
        )
        
        # Verify mocks were called
        mock_surface.assert_called_once()
        mock_figure.assert_called_once()
        self.assertIsNotNone(fig)
    
    @patch('plotly.graph_objects.Figure')
    @patch('plotly.graph_objects.Heatmap')
    def test_create_heatmap(self, mock_heatmap, mock_figure):
        """Test creating a heatmap.."""
        # Setup mocks
        mock_figure.return_value = MagicMock()
        mock_heatmap.return_value = MagicMock()
        
        # Call the function
        fig = create_heatmap(
            height_map=self.height_map,
            title="Test Heatmap",
            colorscale="Viridis"
        )
        
        # Verify mocks were called
        mock_heatmap.assert_called_once()
        mock_figure.assert_called_once()
        self.assertIsNotNone(fig)
    
    @patch('plotly.graph_objects.Figure')
    @patch('plotly.graph_objects.Contour')
    def test_create_contour_plot(self, mock_contour, mock_figure):
        """Test creating a contour plot.."""
        # Setup mocks
        mock_figure.return_value = MagicMock()
        mock_contour.return_value = MagicMock()
        
        # Call the function
        fig = create_contour_plot(
            height_map=self.height_map,
            title="Test Contour Plot",
            colorscale="Viridis"
        )
        
        # Verify mocks were called
        mock_contour.assert_called_once()
        mock_figure.assert_called_once()
        self.assertIsNotNone(fig)
    
    @patch('tmd.plotters.plotly.create_heatmap')
    @patch('plotly.graph_objects.Figure.write_html')
    def test_visualize_height_map_2d(self, mock_write_html, mock_create_heatmap):
        """Test visualizing a height map in 2D.."""
        # Setup mocks
        mock_fig = MagicMock()
        mock_create_heatmap.return_value = mock_fig
        
        # Create a filename
        filename = os.path.join(self.temp_dir, "test_heatmap.html")
        
        # Call the function
        result = visualize_height_map(
            height_map=self.height_map,
            plot_type="heatmap",
            title="Test Height Map 2D",
            colorscale="Viridis",
            filename=filename,
            show=False
        )
        
        # Verify mocks were called
        mock_create_heatmap.assert_called_once()
        mock_write_html.assert_called_once()
        self.assertEqual(result, mock_fig)
    
    @patch('tmd.plotters.plotly.create_surface_plot')
    @patch('plotly.graph_objects.Figure.write_html')
    def test_visualize_height_map_3d(self, mock_write_html, mock_create_surface):
        """Test visualizing a height map in 3D.."""
        # Setup mocks
        mock_fig = MagicMock()
        mock_create_surface.return_value = mock_fig
        
        # Create a filename
        filename = os.path.join(self.temp_dir, "test_surface.html")
        
        # Call the function
        result = visualize_height_map_3d(
            height_map=self.height_map,
            title="Test Height Map 3D",
            colorscale="Viridis",
            filename=filename,
            show=False
        )
        
        # Verify mocks were called
        mock_create_surface.assert_called_once()
        mock_write_html.assert_called_once()
        self.assertEqual(result, mock_fig)
    
    @patch('plotly.io.write_image')
    @patch('tmd.plotters.plotly.create_heatmap')
    def test_export_as_image(self, mock_create_heatmap, mock_write_image):
        """Test exporting a plot as an image.."""
        # Setup mocks
        mock_fig = MagicMock()
        mock_create_heatmap.return_value = mock_fig
        
        # Create a filename
        filename = os.path.join(self.temp_dir, "test_heatmap.png")
        
        # Call the function
        result = visualize_height_map(
            height_map=self.height_map,
            plot_type="heatmap",
            title="Test Height Map Export",
            colorscale="Viridis",
            filename=filename,
            image_export=True,
            show=False
        )
        
        # Verify mocks were called
        mock_create_heatmap.assert_called_once()
        mock_write_image.assert_called_once()
        self.assertEqual(result, mock_fig)
    
    @patch('tmd.plotters.plotly.go', None)
    def test_imports_missing(self):
        """Test behavior when Plotly is not installed.."""
        # This should raise an ImportError
        with self.assertRaises(ImportError):
            fig = create_surface_plot(
                height_map=self.height_map,
                title="Test Surface Plot",
                colorscale="Viridis"
            )

if __name__ == "__main__":
    unittest.main()
