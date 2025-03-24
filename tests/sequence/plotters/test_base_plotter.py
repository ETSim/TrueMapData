""".

Unit tests for the base plotter implementation.

Tests functionality of the BasePlotter class.
"""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from tmd.sequence.plotters.base import BasePlotter

class TestBasePlotter(unittest.TestCase):
    """Test cases for BasePlotter class.."""
    
    def setUp(self):
        """Set up test fixtures.."""
        self.plotter = BasePlotter()
        self.test_height_maps = [
            np.zeros((10, 10)),
            np.ones((10, 10)),
            np.random.random((10, 10))
        ]
        self.test_timestamps = ["Frame 1", "Frame 2", "Frame 3"]
    
    def test_init(self):
        """Test initialization.."""
        self.assertIsInstance(self.plotter, BasePlotter)
    
    def test_check_dependencies(self):
        """Test dependency checking.."""
        # The base implementation should always return True
        result = self.plotter._check_dependencies()
        self.assertTrue(result)
    
    def test_visualize_sequence_not_implemented(self):
        """Test visualize_sequence raises NotImplementedError.."""
        with self.assertRaises(NotImplementedError):
            self.plotter.visualize_sequence(
                height_maps=self.test_height_maps,
                timestamps=self.test_timestamps
            )
    
    def test_visualize_statistics_not_implemented(self):
        """Test visualize_statistics raises NotImplementedError.."""
        with self.assertRaises(NotImplementedError):
            self.plotter.visualize_statistics(
                stats=[{"min": 0, "max": 1} for _ in range(3)],
                timestamps=self.test_timestamps
            )
    
    def test_create_animation_not_implemented(self):
        """Test create_animation raises NotImplementedError.."""
        with self.assertRaises(NotImplementedError):
            self.plotter.create_animation(
                frames_data=self.test_height_maps,
                timestamps=self.test_timestamps
            )
    
    @patch('tmd.sequence.plotters.base.BasePlotter._check_dependencies')
    def test_dependency_check_failure(self, mock_check):
        """Test behavior when dependencies check fails.."""
        # Mock _check_dependencies to return False
        mock_check.return_value = False
        
        # Create a concrete implementation to test
        class ConcretePlotter(BasePlotter):
            def visualize_sequence(self, *args, **kwargs):
                return "Visualization"
        
        plotter = ConcretePlotter()
        
        # Calling visualize_sequence should return None due to failed dependency check
        result = plotter.visualize_sequence(
            height_maps=self.test_height_maps,
            timestamps=self.test_timestamps
        )
        self.assertIsNone(result)
        mock_check.assert_called_once()

if __name__ == "__main__":
    unittest.main()
