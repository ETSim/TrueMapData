#!/usr/bin/env python3
"""
Tests for TMD Base Plotter Abstract Classes and Factory

This module contains unit tests for the base plotter abstract classes and factory
classes defined in the TMD plotting module.
"""

import unittest
import pytest
import numpy as np
from unittest import mock

# Import the classes to test
from tmd.plotters.base import (
    BasePlotter,
    BaseSequencePlotter,
    BasePlotterFactory,
    TMDPlotterFactory,
    TMDSequencePlotterFactory
)


# Test implementation classes
class MockPlotter(BasePlotter):
    """Mock implementation of BasePlotter for testing."""
    
    def plot(self, height_map: np.ndarray, **kwargs):
        """Mock implementation."""
        return {"type": "plot", "height_map": height_map, "options": kwargs}
    
    def save(self, plot_obj, filename: str, **kwargs):
        """Mock implementation."""
        return filename


class MockSequencePlotter(BaseSequencePlotter):
    """Mock implementation of BaseSequencePlotter for testing."""
    
    def visualize_sequence(self, frames, **kwargs):
        """Mock implementation."""
        return {"type": "sequence", "frames": frames, "options": kwargs}
    
    def create_animation(self, frames, **kwargs):
        """Mock implementation."""
        return {"type": "animation", "frames": frames, "options": kwargs}
    
    def visualize_statistics(self, stats_data, **kwargs):
        """Mock implementation."""
        return {"type": "stats", "data": stats_data, "options": kwargs}
    
    def save_figure(self, fig, filename: str, **kwargs):
        """Mock implementation."""
        return filename


class FailingMockPlotter(BasePlotter):
    """Mock plotter that raises ImportError on instantiation."""
    
    def __init__(self):
        """Raise ImportError to simulate missing dependencies."""
        super().__init__()
        raise ImportError("Missing mock dependency")
    
    def plot(self, height_map: np.ndarray, **kwargs):
        """Will not be called due to ImportError in __init__."""
        pass
    
    def save(self, plot_obj, filename: str, **kwargs):
        """Will not be called due to ImportError in __init__."""
        pass


class TestBasePlotter:
    """Tests for BasePlotter abstract class."""
    
    def test_abstract_class(self):
        """Test that BasePlotter cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BasePlotter()
    
    def test_implementation(self):
        """Test that a concrete implementation can be instantiated."""
        plotter = MockPlotter()
        assert isinstance(plotter, BasePlotter)
        
        # Test plot method
        height_map = np.zeros((10, 10))
        result = plotter.plot(height_map, option1="value1")
        assert result["type"] == "plot"
        assert result["height_map"] is height_map
        assert result["options"]["option1"] == "value1"
        
        # Test save method
        filename = "test.png"
        saved_filename = plotter.save(result, filename)
        assert saved_filename == filename


class TestBaseSequencePlotter:
    """Tests for BaseSequencePlotter abstract class."""
    
    def test_abstract_class(self):
        """Test that BaseSequencePlotter cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseSequencePlotter()
    
    def test_implementation(self):
        """Test that a concrete implementation can be instantiated."""
        plotter = MockSequencePlotter()
        assert isinstance(plotter, BaseSequencePlotter)
        
        # Test visualize_sequence method
        frames = [np.zeros((10, 10)), np.ones((10, 10))]
        result = plotter.visualize_sequence(frames, option1="value1")
        assert result["type"] == "sequence"
        assert result["frames"] is frames
        assert result["options"]["option1"] == "value1"
        
        # Test create_animation method
        anim_result = plotter.create_animation(frames, fps=30)
        assert anim_result["type"] == "animation"
        assert anim_result["frames"] is frames
        assert anim_result["options"]["fps"] == 30
        
        # Test visualize_statistics method
        stats_data = {"metric1": [1.0, 2.0, 3.0], "metric2": [4.0, 5.0, 6.0]}
        stats_result = plotter.visualize_statistics(stats_data, title="Stats")
        assert stats_result["type"] == "stats"
        assert stats_result["data"] is stats_data
        assert stats_result["options"]["title"] == "Stats"
        
        # Test save_figure method
        filename = "test.png"
        saved_filename = plotter.save_figure(stats_result, filename)
        assert saved_filename == filename


class TestBasePlotterFactory:
    """Tests for BasePlotterFactory."""
    
    def setup_method(self):
        """Set up test environment."""
        # Since we can't directly modify the class _registry attribute normally,
        # for testing we'll create a temporary subclass
        class TestFactory(BasePlotterFactory):
            _registry = {}
        
        self.factory_class = TestFactory
    
    def test_register_and_get_registered(self):
        """Test registering plotters and getting the registered list."""
        # Register a plotter
        self.factory_class.register("test", MockPlotter)
        
        # Check registration
        assert "test" in self.factory_class.get_registered_plotters()
        
        # Register another
        self.factory_class.register("another", FailingMockPlotter)
        
        # Check both are registered
        registered = self.factory_class.get_registered_plotters()
        assert "test" in registered
        assert "another" in registered
        assert len(registered) == 2


class TestTMDPlotterFactory:
    """Tests for TMDPlotterFactory."""
    
    def setup_method(self):
        """Set up test environment."""
        # Clear registry before each test
        self._original_registry = TMDPlotterFactory._registry.copy()
        TMDPlotterFactory._registry = {}
    
    def teardown_method(self):
        """Restore registry after each test."""
        TMDPlotterFactory._registry = self._original_registry
    
    def test_register_plotter(self):
        """Test registering a plotter."""
        TMDPlotterFactory.register("mock", MockPlotter)
        assert "mock" in TMDPlotterFactory.get_registered_plotters()
        
        # Register the same plotter with a different name
        TMDPlotterFactory.register("another_mock", MockPlotter)
        registered = TMDPlotterFactory.get_registered_plotters()
        assert "mock" in registered
        assert "another_mock" in registered
        
        # Test case insensitivity
        TMDPlotterFactory.register("CaseSensitive", MockPlotter)
        assert "casesensitive" in TMDPlotterFactory.get_registered_plotters()
    
    def test_create_plotter(self):
        """Test creating a plotter instance."""
        TMDPlotterFactory.register("mock", MockPlotter)
        plotter = TMDPlotterFactory.create_plotter("mock")
        assert isinstance(plotter, MockPlotter)
        
        # Test case insensitivity
        plotter = TMDPlotterFactory.create_plotter("MOCK")
        assert isinstance(plotter, MockPlotter)
        
        # Test creating unregistered plotter
        with pytest.raises(ValueError) as excinfo:
            TMDPlotterFactory.create_plotter("nonexistent")
        assert "Unknown plotter" in str(excinfo.value)
        assert "mock" in str(excinfo.value)  # Should show available plotters
    
    def test_get_available_plotters(self):
        """Test getting available plotters."""
        TMDPlotterFactory.register("mock", MockPlotter)
        TMDPlotterFactory.register("failing", FailingMockPlotter)
        
        available = TMDPlotterFactory.get_available_plotters()
        assert "mock" in available
        assert "failing" not in available


class TestTMDSequencePlotterFactory:
    """Tests for TMDSequencePlotterFactory."""
    
    def setup_method(self):
        """Set up test environment."""
        # Clear registry before each test
        self._original_registry = TMDSequencePlotterFactory._registry.copy()
        TMDSequencePlotterFactory._registry = {}
    
    def teardown_method(self):
        """Restore registry after each test."""
        TMDSequencePlotterFactory._registry = self._original_registry
    
    def test_register_plotter(self):
        """Test registering a sequence plotter."""
        TMDSequencePlotterFactory.register("mock_seq", MockSequencePlotter)
        assert "mock_seq" in TMDSequencePlotterFactory.get_registered_plotters()
        
        # Register with a different name
        TMDSequencePlotterFactory.register("another_seq", MockSequencePlotter)
        registered = TMDSequencePlotterFactory.get_registered_plotters()
        assert "mock_seq" in registered
        assert "another_seq" in registered
    
    def test_create_plotter(self):
        """Test creating a sequence plotter instance."""
        TMDSequencePlotterFactory.register("mock_seq", MockSequencePlotter)
        plotter = TMDSequencePlotterFactory.create_plotter("mock_seq")
        assert isinstance(plotter, MockSequencePlotter)
        
        # Test creating unregistered plotter
        with pytest.raises(ValueError) as excinfo:
            TMDSequencePlotterFactory.create_plotter("nonexistent")
        assert "Unknown sequence plotter" in str(excinfo.value)
        assert "mock_seq" in str(excinfo.value)  # Should show available plotters
    
    def test_registry_separation(self):
        """Test that the registries of the two factories are separate."""
        # Register plotters in both factories
        TMDPlotterFactory.register("mock", MockPlotter)
        TMDSequencePlotterFactory.register("mock_seq", MockSequencePlotter)
        
        # Check that registries are separate
        assert "mock" in TMDPlotterFactory.get_registered_plotters()
        assert "mock" not in TMDSequencePlotterFactory.get_registered_plotters()
        assert "mock_seq" in TMDSequencePlotterFactory.get_registered_plotters()
        assert "mock_seq" not in TMDPlotterFactory.get_registered_plotters()


if __name__ == "__main__":
    pytest.main(["-v", __file__])