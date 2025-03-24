""".

Unit tests for the Seaborn plotting module.

These tests verify the functionality of the Seaborn-based visualization functions.
"""
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import matplotlib
import numpy as np

matplotlib.use("Agg")  # Use non-interactive backend for testing

# Patch the seaborn import to handle cases where seaborn is not installed
try:
    import seaborn as sns

    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# Mock Seaborn if it's not available
if not HAS_SEABORN:
    import sys

    sns = MagicMock()
    sys.modules["seaborn"] = sns

from tmd.plotters.seaborn import (
    plot_height_distribution,
    plot_heightmap_heatmap,
    plot_joint_distribution,
    plot_profile_comparison,
)


@unittest.skipIf(not HAS_SEABORN, "Seaborn not installed")
class TestSeabornPlotters(unittest.TestCase):
    """Test case for Seaborn-based visualization functions.."""

    def setUp(self):
        """Create test data and temporary directory for output files.."""
        # Create a test height map: small to keep tests fast
        self.height_map = np.zeros((50, 50))
        # Add a simple pattern for visualization
        x = np.linspace(-3, 3, 50)
        y = np.linspace(-3, 3, 50)
        X, Y = np.meshgrid(x, y)
        self.height_map = np.sin(X) * np.cos(Y)

        # Create test profiles
        x = np.linspace(0, 2 * np.pi, 100)
        self.profiles = [np.sin(x), np.cos(x), np.sin(2 * x)]
        self.labels = ["Sine", "Cosine", "Double Sine"]

        # Create a temp directory for test outputs
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        """Clean up temporary directory.."""
        self.temp_dir.cleanup()

    def test_plot_height_distribution(self):
        """Test creating a height distribution plot.."""
        # Test with basic parameters
        fig, ax = plot_height_distribution(self.height_map)
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)

        # Test with a filename
        filename = os.path.join(self.temp_dir.name, "height_dist.png")
        fig, ax = plot_height_distribution(
            self.height_map, title="Test Distribution", filename=filename, bins=30, kde=True
        )
        self.assertTrue(os.path.exists(filename))
        self.assertGreater(os.path.getsize(filename), 0)

    def test_plot_heightmap_heatmap(self):
        """Test creating a heatmap.."""
        # Test basic functionality
        fig, ax = plot_heightmap_heatmap(self.height_map)
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)

        # Test with a filename
        filename = os.path.join(self.temp_dir.name, "heatmap.png")
        fig, ax = plot_heightmap_heatmap(
            self.height_map, title="Test Heatmap", filename=filename, cmap="plasma", robust=True
        )
        self.assertTrue(os.path.exists(filename))
        self.assertGreater(os.path.getsize(filename), 0)

        # Test with annotations
        fig, ax = plot_heightmap_heatmap(
            self.height_map[:10, :10], annot=True  # Use smaller map for annotations
        )
        self.assertIsNotNone(fig)

    def test_plot_profile_comparison(self):
        """Test creating a profile comparison plot.."""
        # Test basic functionality
        fig, ax = plot_profile_comparison(self.profiles, self.labels)
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)

        # Test with a filename
        filename = os.path.join(self.temp_dir.name, "profile_comparison.png")
        fig, ax = plot_profile_comparison(
            self.profiles,
            self.labels,
            title="Test Profile Comparison",
            filename=filename,
            palette="Set2",
            fill=True,
        )
        self.assertTrue(os.path.exists(filename))
        self.assertGreater(os.path.getsize(filename), 0)

        # Test error handling for mismatched labels
        with self.assertRaises(ValueError):
            plot_profile_comparison(self.profiles, self.labels[:1])

    def test_plot_joint_distribution(self):
        """Test creating a joint distribution plot.."""
        # Test basic functionality
        joint_grid = plot_joint_distribution(self.height_map)
        self.assertIsNotNone(joint_grid)

        # Test with a filename
        filename = os.path.join(self.temp_dir.name, "joint_dist.png")
        joint_grid = plot_joint_distribution(
            self.height_map, title="Test Joint Distribution", filename=filename, kind="kde"
        )
        self.assertTrue(os.path.exists(filename))
        self.assertGreater(os.path.getsize(filename), 0)


if __name__ == "__main__":
    unittest.main()
