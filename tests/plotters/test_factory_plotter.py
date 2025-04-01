#!/usr/bin/env python3
"""
Tests for TMD Plotter Factories

This module contains unit tests for the TMD Plotter Factory classes that create
appropriate plotting instances based on strategy and available dependencies.
"""

import pytest
import logging
from unittest import mock
from typing import Dict, List, Any, Type

# Import the modules to test
from tmd.plotters.factory import (
    PlotterFactoryBase,
    TMDPlotterFactory,
    TMDSequencePlotterFactory,
    _register_all_plotters
)
from tmd.plotters.base import BasePlotter, BaseSequencePlotter
from tmd.utils.files import TMDFileUtilities


# Mock plotter classes
class MockMatplotlibPlotter(BasePlotter):
    """Mock Matplotlib plotter for testing."""
    def plot(self, height_map, **kwargs):
        return {"type": "matplotlib", "data": height_map}
        
    def save(self, plot_obj, filename, **kwargs):
        return filename


class MockPlotlyPlotter(BasePlotter):
    """Mock Plotly plotter for testing."""
    def plot(self, height_map, **kwargs):
        return {"type": "plotly", "data": height_map}
        
    def save(self, plot_obj, filename, **kwargs):
        return filename


class MockSeabornPlotter(BasePlotter):
    """Mock Seaborn plotter for testing."""
    def plot(self, height_map, **kwargs):
        return {"type": "seaborn", "data": height_map}
        
    def save(self, plot_obj, filename, **kwargs):
        return filename


class MockPolyscopePlotter(BasePlotter):
    """Mock Polyscope plotter for testing."""
    def __init__(self, is_sequence=False):
        self.is_sequence = is_sequence
        
    def plot(self, height_map, **kwargs):
        return {"type": "polyscope", "data": height_map, "is_sequence": self.is_sequence}
        
    def save(self, plot_obj, filename, **kwargs):
        return filename


class MockMatplotlibSequencePlotter(BaseSequencePlotter):
    """Mock Matplotlib sequence plotter for testing."""
    def visualize_sequence(self, frames, **kwargs):
        return {"type": "matplotlib_sequence", "frames": frames}
        
    def create_animation(self, frames, **kwargs):
        return {"type": "matplotlib_animation", "frames": frames}
        
    def visualize_statistics(self, stats_data, **kwargs):
        return {"type": "matplotlib_stats", "data": stats_data}
        
    def save_figure(self, fig, filename, **kwargs):
        return filename


class MockPlotlySequencePlotter(BaseSequencePlotter):
    """Mock Plotly sequence plotter for testing."""
    def visualize_sequence(self, frames, **kwargs):
        return {"type": "plotly_sequence", "frames": frames}
        
    def create_animation(self, frames, **kwargs):
        return {"type": "plotly_animation", "frames": frames}
        
    def visualize_statistics(self, stats_data, **kwargs):
        return {"type": "plotly_stats", "data": stats_data}
        
    def save_figure(self, fig, filename, **kwargs):
        return filename


# Tests for the PlotterFactoryBase class
class TestPlotterFactoryBase:
    """Test cases for the PlotterFactoryBase class."""
    
    def setup_method(self):
        """Set up a test subclass of PlotterFactoryBase."""
        class TestFactory(PlotterFactoryBase):
            STRATEGY_DEPENDENCIES = {
                "test_strat1": ["numpy", "scipy"],
                "test_strat2": ["nonexistent_module"],
                "test_strat3": []
            }
            STRATEGY_CLASSES = {
                "test_strat1": "module.path.Class1",
                "test_strat2": "module.path.Class2",
                "test_strat3": "module.path.Class3"
            }
            DEFAULT_STRATEGY = "test_strat1"
        
        self.factory_class = TestFactory
    
    def test_check_strategy_availability(self):
        """Test checking if a strategy is available."""
        # Mock TMDFileUtilities.import_optional_dependency to control what's "available"
        def mock_import(module_name):
            available_modules = ["numpy", "scipy"]
            return mock.MagicMock() if module_name in available_modules else None
        
        with mock.patch.object(TMDFileUtilities, 'import_optional_dependency', side_effect=mock_import):
            # Strategy with all dependencies available
            assert self.factory_class.check_strategy_availability("test_strat1") is True
            
            # Strategy with missing dependencies
            assert self.factory_class.check_strategy_availability("test_strat2") is False
            
            # Strategy with no dependencies
            assert self.factory_class.check_strategy_availability("test_strat3") is True
            
            # Non-existent strategy
            with pytest.raises(ValueError):
                self.factory_class.check_strategy_availability("nonexistent_strategy")
    
    def test_get_missing_dependencies(self):
        """Test getting missing dependencies for a strategy."""
        # Mock TMDFileUtilities.import_optional_dependency to control what's "available"
        def mock_import(module_name):
            available_modules = ["numpy"]
            return mock.MagicMock() if module_name in available_modules else None
        
        with mock.patch.object(TMDFileUtilities, 'import_optional_dependency', side_effect=mock_import):
            # Strategy with some missing dependencies
            missing = self.factory_class.get_missing_dependencies("test_strat1")
            assert "scipy" in missing
            assert "numpy" not in missing
            
            # Strategy with all missing dependencies
            missing = self.factory_class.get_missing_dependencies("test_strat2")
            assert "nonexistent_module" in missing
            
            # Strategy with no dependencies
            missing = self.factory_class.get_missing_dependencies("test_strat3")
            assert len(missing) == 0
            
            # Non-existent strategy
            with pytest.raises(ValueError):
                self.factory_class.get_missing_dependencies("nonexistent_strategy")
    
    def test_list_available_strategies(self):
        """Test listing available strategies."""
        # Mock TMDFileUtilities.import_optional_dependency to control what's "available"
        def mock_import(module_name):
            available_modules = ["numpy"]
            return mock.MagicMock() if module_name in available_modules else None
        
        with mock.patch.object(TMDFileUtilities, 'import_optional_dependency', side_effect=mock_import):
            strategies = self.factory_class.list_available_strategies()
            
            # Check the expected results
            assert "test_strat1" in strategies
            assert "test_strat2" in strategies
            assert "test_strat3" in strategies
            
            # Strategies with all dependencies available are True
            assert strategies["test_strat3"] is True
            
            # Strategies with some/all missing dependencies are False
            assert strategies["test_strat1"] is False  # Missing scipy
            assert strategies["test_strat2"] is False  # Missing nonexistent_module
    
    def test_import_class(self):
        """Test importing a class from a dotted path."""
        # Mock importlib.import_module to control class imports
        module_mock = mock.MagicMock()
        module_mock.TestClass = "mock_class_instance"
        
        with mock.patch('importlib.import_module', return_value=module_mock):
            result = self.factory_class._import_class("module.path.TestClass")
            assert result == "mock_class_instance"
        
        # Test import failure
        with mock.patch('importlib.import_module', side_effect=ImportError("Module not found")):
            with pytest.raises(ImportError):
                self.factory_class._import_class("nonexistent.path.Class")


# Tests for the TMDPlotterFactory class
class TestTMDPlotterFactory:
    """Test cases for the TMDPlotterFactory class."""
    
    def setup_method(self):
        """Set up test environment for TMDPlotterFactory."""
        # Save original registry and restore after test
        self._original_registry = TMDPlotterFactory._registry.copy()
        TMDPlotterFactory._registry = {}
    
    def teardown_method(self):
        """Clean up after each test."""
        TMDPlotterFactory._registry = self._original_registry
    
    def test_create_plotter_registered(self):
        """Test creating a plotter that's already registered."""
        # Register a mock plotter
        TMDPlotterFactory.register("test_plotter", MockMatplotlibPlotter)
        
        # Create the plotter
        plotter = TMDPlotterFactory.create_plotter("test_plotter")
        
        # Check that we got the right type
        assert isinstance(plotter, MockMatplotlibPlotter)
        
        # Test case insensitivity
        plotter = TMDPlotterFactory.create_plotter("TEST_PLOTTER")
        assert isinstance(plotter, MockMatplotlibPlotter)
    
    def test_create_plotter_matplotlib_special_case(self):
        """Test the special case handling for matplotlib."""
        # Mock import for matplotlib
        matplotlib_mock = mock.MagicMock()
        
        with mock.patch.dict('sys.modules', {'matplotlib.pyplot': matplotlib_mock}):
            with mock.patch('tmd.plotters.factory.TMDPlotterFactory._import_class',
                          return_value=MockMatplotlibPlotter):
                
                # Create matplotlib plotter
                plotter = TMDPlotterFactory.create_plotter("matplotlib")
                
                # Check that we got a matplotlib plotter
                assert isinstance(plotter, MockMatplotlibPlotter)
                
                # Check that it was registered
                assert "matplotlib" in TMDPlotterFactory._registry
    
    def test_create_plotter_fallback(self):
        """Test fallback to available strategy when requested one is not available."""
        # Set up mock dependencies
        def mock_import(name):
            # Only matplotlib is available
            if name == 'matplotlib.pyplot':
                return mock.MagicMock()
            raise ImportError(f"No module named '{name}'")
        
        # Mock the imports
        with mock.patch('builtins.__import__', side_effect=mock_import):
            with mock.patch('tmd.plotters.factory.TMDPlotterFactory._import_class') as mock_import_class:
                mock_import_class.return_value = MockMatplotlibPlotter
                
                # Request plotly but it's not available
                with mock.patch('tmd.plotters.factory.logger') as mock_logger:
                    plotter = TMDPlotterFactory.create_plotter("plotly")
                    
                    # Should fall back to matplotlib
                    assert isinstance(plotter, MockMatplotlibPlotter)
                    
                    # Should log a warning
                    mock_logger.warning.assert_called_once()
    
    def test_create_plotter_no_backend(self):
        """Test error when no backends are available."""
        # Mock to make all imports fail
        with mock.patch.dict('sys.modules', {}, clear=True):
            with mock.patch('tmd.plotters.factory.TMDPlotterFactory._import_class',
                          side_effect=ImportError):
                
                # Trying to create any plotter should fail
                with pytest.raises(ValueError) as excinfo:
                    TMDPlotterFactory.create_plotter()
                
                # Error should mention missing backends
                assert "No plotting backends available" in str(excinfo.value)
    
    def test_create_plotter_dynamic_import(self):
        """Test dynamically importing a plotter class."""
        # Register no plotters initially
        TMDPlotterFactory._registry = {}
        
        # Mock to make the import for plotly work
        plotly_mock = mock.MagicMock()
        with mock.patch.dict('sys.modules', {'plotly': plotly_mock, 'plotly.graph_objects': plotly_mock}):
            with mock.patch('tmd.plotters.factory.TMDPlotterFactory._import_class',
                          return_value=MockPlotlyPlotter):
                
                # Create plotly plotter
                plotter = TMDPlotterFactory.create_plotter("plotly")
                
                # Check that we got a plotly plotter
                assert isinstance(plotter, MockPlotlyPlotter)
                
                # Check that it was registered
                assert "plotly" in TMDPlotterFactory._registry


# Tests for the TMDSequencePlotterFactory class
class TestTMDSequencePlotterFactory:
    """Test cases for the TMDSequencePlotterFactory class."""
    
    def setup_method(self):
        """Set up test environment for TMDSequencePlotterFactory."""
        # Save original registry and restore after test
        self._original_registry = TMDSequencePlotterFactory._registry.copy()
        TMDSequencePlotterFactory._registry = {}
    
    def teardown_method(self):
        """Clean up after each test."""
        TMDSequencePlotterFactory._registry = self._original_registry
    
    def test_create_plotter_registered(self):
        """Test creating a plotter that's already registered."""
        # Register a mock plotter
        TMDSequencePlotterFactory.register("test_seq_plotter", MockMatplotlibSequencePlotter)
        
        # Create the plotter
        plotter = TMDSequencePlotterFactory.create_plotter("test_seq_plotter")
        
        # Check that we got the right type
        assert isinstance(plotter, MockMatplotlibSequencePlotter)
    
    def test_create_plotter_polyscope_special_case(self):
        """Test the special case handling for polyscope."""
        # Register polyscope plotter
        TMDSequencePlotterFactory.register("polyscope", MockPolyscopePlotter)
        
        # Create polyscope plotter
        plotter = TMDSequencePlotterFactory.create_plotter("polyscope")
        
        # Check that we got a polyscope plotter
        assert isinstance(plotter, MockPolyscopePlotter)
        
        # Check that is_sequence flag was set
        assert plotter.is_sequence is True
    
    def test_create_plotter_missing_dependencies(self):
        """Test error when dependencies are missing."""
        # Mock to check dependencies and report missing ones
        with mock.patch.object(TMDSequencePlotterFactory, 'get_missing_dependencies',
                            return_value=['missing_dep1', 'missing_dep2']):
            
            # Trying to create a plotter should raise ImportError
            with pytest.raises(ImportError) as excinfo:
                TMDSequencePlotterFactory.create_plotter("plotly")
            
            # Error should mention missing dependencies
            assert "Missing dependencies" in str(excinfo.value)
            assert "missing_dep1" in str(excinfo.value)
            assert "missing_dep2" in str(excinfo.value)
    
    def test_create_plotter_invalid_strategy(self):
        """Test error with invalid strategy."""
        # Try to create a plotter with an invalid strategy
        with pytest.raises(ValueError) as excinfo:
            TMDSequencePlotterFactory.create_plotter("invalid_strategy")
        
        # Error should mention the invalid strategy
        assert "Unsupported sequence plotter strategy" in str(excinfo.value)
        assert "invalid_strategy" in str(excinfo.value)


# Tests for the register_all_plotters function
class TestRegisterAllPlotters:
    """Test cases for the _register_all_plotters function."""
    
    def setup_method(self):
        """Save original registries and restore after test."""
        self._original_tmd_registry = TMDPlotterFactory._registry.copy()
        self._original_seq_registry = TMDSequencePlotterFactory._registry.copy()
        
        # Clear registries
        TMDPlotterFactory._registry = {}
        TMDSequencePlotterFactory._registry = {}
    
    def teardown_method(self):
        """Restore original registries."""
        TMDPlotterFactory._registry = self._original_tmd_registry
        TMDSequencePlotterFactory._registry = self._original_seq_registry
    
    def test_register_all_plotters(self):
        """Test registering all available plotters."""
        # Create mocks for all possible plotters
        matplotlib_mods = {
            'tmd.plotters.matplotlib': mock.MagicMock(
                MatplotlibHeightMapPlotter=MockMatplotlibPlotter,
                MatplotlibSequencePlotter=MockMatplotlibSequencePlotter
            )
        }
        
        plotly_mods = {
            'tmd.plotters.plotly': mock.MagicMock(
                PlotlyHeightMapVisualizer=MockPlotlyPlotter,
                PlotlySequenceVisualizer=MockPlotlySequencePlotter
            )
        }
        
        seaborn_mods = {
            'tmd.plotters.seaborn': mock.MagicMock(
                SeabornHeightMapPlotter=MockSeabornPlotter,
                SeabornSequencePlotter=mock.MagicMock()
            )
        }
        
        polyscope_mods = {
            'tmd.plotters.polyscope': mock.MagicMock(
                PolyscopePlotter=MockPolyscopePlotter
            )
        }
        
        # Test with all plotters available
        with mock.patch.dict('sys.modules', {**matplotlib_mods, **plotly_mods, **seaborn_mods, **polyscope_mods}):
            with mock.patch('tmd.plotters.factory.logger') as mock_logger:
                _register_all_plotters()
                
                # Should log success messages
                assert mock_logger.debug.call_count >= 4
                
                # Check that plotters were registered
                assert "matplotlib" in TMDPlotterFactory._registry
                assert "plotly" in TMDPlotterFactory._registry
                assert "seaborn" in TMDPlotterFactory._registry
                assert "polyscope" in TMDPlotterFactory._registry
                
                # Check that sequence plotters were registered
                assert "matplotlib" in TMDSequencePlotterFactory._registry
                assert "plotly" in TMDSequencePlotterFactory._registry
                assert "seaborn" in TMDSequencePlotterFactory._registry
                assert "polyscope" in TMDSequencePlotterFactory._registry
        
        # Reset registries for next test
        TMDPlotterFactory._registry = {}
        TMDSequencePlotterFactory._registry = {}
        
        # Test with only matplotlib available
        with mock.patch.dict('sys.modules', {**matplotlib_mods}):
            with mock.patch('tmd.plotters.factory.logger') as mock_logger:
                # Make imports for other modules fail
                def mock_import_error(name):
                    if "matplotlib" in name:
                        return matplotlib_mods[name]
                    raise ImportError(f"No module named '{name}'")
                
                with mock.patch('builtins.__import__', side_effect=mock_import_error):
                    _register_all_plotters()
                    
                    # Should log success for matplotlib and debug for others
                    assert mock_logger.debug.call_count >= 4
                    
                    # Only matplotlib should be registered
                    assert "matplotlib" in TMDPlotterFactory._registry
                    assert "plotly" not in TMDPlotterFactory._registry
                    assert "seaborn" not in TMDPlotterFactory._registry
                    assert "polyscope" not in TMDPlotterFactory._registry
                    
                    # Same for sequence plotters
                    assert "matplotlib" in TMDSequencePlotterFactory._registry
                    assert "plotly" not in TMDSequencePlotterFactory._registry
                    assert "seaborn" not in TMDSequencePlotterFactory._registry
                    assert "polyscope" not in TMDSequencePlotterFactory._registry


if __name__ == "__main__":
    pytest.main(["-v", __file__])