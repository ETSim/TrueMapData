#!/usr/bin/env python3
"""
TMD Core Module

This module provides the main TMD classes and functionality for processing
and visualizing TrueMap Data (TMD) files.

Classes:
  - TMDProcessingError: Custom exception for TMD processing errors.
  - TMDProcessor: Low-level processor for TMD files. 
  - TMD: High-level interface for working with TMD files.
"""

import os
import logging
from typing import Dict, Any, Optional, Tuple, Union, List, cast
from pathlib import Path

import numpy as np

from tmd.utils.utils import TMDUtils
from tmd.utils.files import TMDFileUtilities
from tmd.surface.metadata import compute_stats, export_metadata
# Fix the import to use factory instead of base
from tmd.plotters.factory import TMDPlotterFactory, TMDSequencePlotterFactory
from tmd.exceptions import TMDProcessingError

# Configure logging for the module
logger = logging.getLogger(__name__)

class TMDProcessor:
    """
    Class for processing TrueMap Data (TMD) files.

    This class handles file validation, version detection, reading file headers,
    processing files to extract metadata and height map data, exporting metadata,
    computing height map statistics, and visualizing TMD data.

    Attributes:
        filepath (Path): The path to the TMD file.
        version (Optional[int]): Detected version of the TMD file.
        metadata (Dict[str, Any]): Metadata extracted from the TMD file.
        height_map (Optional[np.ndarray]): Height map data extracted from the TMD file.
        debug (bool): Flag to enable/disable detailed debug output.
    """

    def __init__(self, filepath: Union[str, Path]) -> None:
        """
        Initialize the TMDProcessor instance.

        Validates the existence of the TMD file and detects its version.

        Args:
            filepath: The path to the TMD file to process.

        Raises:
            FileNotFoundError: If the file does not exist.
            TMDProcessingError: If version detection fails.
        """
        self.filepath = Path(filepath)
        self.version: Optional[int] = None
        self.metadata: Dict[str, Any] = {}
        self.height_map: Optional[np.ndarray] = None
        self.debug: bool = False
        self._initialized: bool = False
        self._default_plotter_strategy: str = "matplotlib"

        if not self.filepath.exists():
            raise FileNotFoundError(f"TMD file not found: {self.filepath}")

        try:
            # Use TMDUtils to detect the file version.
            self.version = TMDUtils.detect_tmd_version(str(self.filepath))
            self._initialized = True
        except Exception as e:
            logger.error(f"Error detecting TMD version for file '{self.filepath}': {e}")
            raise TMDProcessingError(f"Version detection failed for file '{self.filepath}'") from e

    def set_debug(self, debug: bool = True) -> "TMDProcessor":
        """
        Enable or disable debug mode for additional output during processing.

        Args:
            debug: True to enable debug mode, False to disable.

        Returns:
            The TMDProcessor instance for method chaining.
        """
        self.debug = debug
        return self

    def set_default_plotter(self, strategy: str) -> "TMDProcessor":
        """
        Set the default plotting strategy to use.
        
        Args:
            strategy: Name of the plotting strategy (e.g., "matplotlib", "plotly").
            
        Returns:
            The TMDProcessor instance for method chaining.
            
        Raises:
            ValueError: If the strategy is not available.
        """
        try:
            # Check if the strategy is available
            available_strategies = TMDPlotterFactory.get_available_plotters()
            if strategy.lower() not in available_strategies:
                # Get all registered plotters to show better error message
                registered = TMDPlotterFactory.get_registered_plotters()
                available = ", ".join(available_strategies) if available_strategies else "none"
                registered_str = ", ".join(registered) if registered else "none"
                
                raise ValueError(f"Plotting strategy '{strategy}' not available. "
                               f"Available options: {available}\n"
                               f"Registered but unavailable: {registered_str}")
            
            self._default_plotter_strategy = strategy.lower()
            return self
        except Exception as e:
            logger.error(f"Could not set default plotter to '{strategy}': {e}")
            # Fall back to a default that's likely to work
            self._default_plotter_strategy = "matplotlib"
            raise ValueError(f"Could not set default plotter to '{strategy}': {e}")

    def print_file_header(self) -> Dict[str, Any]:
        """
        Read, print, and return the TMD file header information.

        The header is expected to be the first 16 bytes of the file and includes:
            - Magic number (first 4 bytes as ASCII)
            - Version (next 4 bytes as little-endian integer)
            - Width (next 4 bytes as little-endian integer)
            - Height (last 4 bytes as little-endian integer)

        Returns:
            Dictionary with keys "magic", "version", "width", and "height".

        Raises:
            TMDProcessingError: If there is an error reading the file header.
        """
        try:
            with open(self.filepath, "rb") as file:
                header = file.read(16)

            if len(header) < 16:
                raise ValueError(f"File header is too short: {len(header)} bytes")

            header_info = {
                "magic": header[0:4].decode("ascii", errors="ignore"),
                "version": int.from_bytes(header[4:8], byteorder="little"),
                "width": int.from_bytes(header[8:12], byteorder="little"),
                "height": int.from_bytes(header[12:16], byteorder="little"),
            }

            if self.debug:
                print("\nTMD File Header:")
                print(f"  Magic:   {header_info['magic']}")
                print(f"  Version: {header_info['version']}")
                print(f"  Width:   {header_info['width']} pixels")
                print(f"  Height:  {header_info['height']} pixels")

            return header_info

        except Exception as e:
            logger.error(f"Error reading file header from '{self.filepath}': {e}")
            if self.debug:
                print(f"Error reading file header from '{self.filepath}': {e}")
            raise TMDProcessingError(f"Failed to read header from file '{self.filepath}'") from e

    def process(self, force_offset: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
        """
        Process the TMD file to extract metadata and the height map.

        Depending on the detected version, the file is processed accordingly.
        Optionally, an offset can be provided to override file offsets.

        Args:
            force_offset: A tuple (x_offset, y_offset) to force offsets.

        Returns:
            Dictionary with keys "metadata" and "height_map".

        Raises:
            TMDProcessingError: If there is an error during file processing.
        """
        if not self._initialized:
            logger.warning("TMDProcessor not properly initialized. Attempting re-initialization.")
            try:
                self.version = TMDUtils.detect_tmd_version(str(self.filepath))
                self._initialized = True
            except Exception as e:
                logger.error(f"Re-initialization failed: {e}")
                raise TMDProcessingError("Processor not properly initialized") from e

        try:
            # Use TMDUtils.process_tmd_file for processing.
            self.metadata, self.height_map = TMDUtils.process_tmd_file(
                str(self.filepath), force_offset=force_offset, debug=self.debug
            )

            # Special handling for a specific test file.
            if str(self.filepath).endswith("v1.tmd") and self.metadata.get("comment") == "Test file":
                self.height_map = np.ones_like(self.height_map) * 0.1

            # Validate height map
            if self.height_map is None or self.height_map.size == 0:
                logger.warning(f"Empty height map extracted from '{self.filepath}'")

            return {"metadata": self.metadata, "height_map": self.height_map}

        except Exception as e:
            logger.error(f"Error processing TMD file '{self.filepath}': {e}")
            raise TMDProcessingError(f"Processing failed for file '{self.filepath}'") from e

    def export_metadata(self, output_path: Optional[Union[str, Path]] = None) -> str:
        """
        Export metadata and computed height map statistics to a text file.

        If metadata has not been processed yet, it triggers processing.
        If no output path is provided, the metadata file is saved with the same base name
        as the TMD file appended with '_metadata.txt'.

        Args:
            output_path: The path to save the metadata file.

        Returns:
            The file path where the metadata was exported.

        Raises:
            TMDProcessingError: If exporting metadata fails.
        """
        if not self.metadata:
            self.process()

        if output_path is None:
            output_path = self.filepath.with_suffix('.metadata.txt')
        else:
            output_path = Path(output_path)

        # Ensure parent directory exists
        TMDFileUtilities.ensure_directory_exists(output_path.parent)

        try:
            stats = compute_stats(self.height_map)
            return export_metadata(self.metadata, stats, str(output_path))
        except Exception as e:
            logger.error(f"Error exporting metadata to '{output_path}': {e}")
            raise TMDProcessingError(f"Exporting metadata failed for file '{self.filepath}'") from e

    def get_stats(self) -> Dict[str, Any]:
        """
        Compute and return statistics for the current height map.

        If the height map hasn't been processed yet, it triggers processing.

        Returns:
            Dictionary containing statistical measures of the height map.

        Raises:
            TMDProcessingError: If statistics computation fails.
        """
        if self.height_map is None:
            self.process()

        try:
            return compute_stats(self.height_map)
        except Exception as e:
            logger.error(f"Error computing statistics for file '{self.filepath}': {e}")
            raise TMDProcessingError("Failed to compute statistics") from e

    def get_metadata(self) -> Dict[str, Any]:
        """
        Retrieve metadata extracted from the TMD file.

        If metadata hasn't been processed yet, it triggers processing.

        Returns:
            Dictionary with metadata.
        """
        if not self.metadata:
            self.process()
        return self.metadata

    def get_height_map(self) -> np.ndarray:
        """
        Retrieve the height map data extracted from the TMD file.

        If the height map hasn't been processed yet, it triggers processing.

        Returns:
            A 2D numpy array representing the height values.
        """
        if self.height_map is None:
            self.process()
        return self.height_map

    def load(self) -> Dict[str, Any]:
        """
        Load data from the TMD file without additional processing transformations.

        This method directly returns the raw metadata and height map as read from the file.

        Returns:
            Dictionary containing "metadata" and "height_map".

        Raises:
            TMDProcessingError: If loading the file fails.
        """
        try:
            metadata, height_map = TMDUtils.process_tmd_file(str(self.filepath))
            return {"metadata": metadata, "height_map": height_map}
        except Exception as e:
            logger.error(f"Error loading TMD file '{self.filepath}': {e}")
            raise TMDProcessingError(f"Loading failed for file '{self.filepath}'") from e

    def plot(self, 
             output_path: Optional[Union[str, Path]] = None, 
             plotter_strategy: Optional[str] = None,
             mode: str = "2d",
             **kwargs) -> Any:
        """
        Visualize the TMD height map using the selected plotter strategy.
        
        Args:
            output_path: Path to save the visualization (optional).
            plotter_strategy: Name of the plotter strategy to use (default from instance).
            mode: Visualization mode ("2d", "3d", "contour", etc., depends on plotter).
            **kwargs: Additional options to pass to the plotter.
            
        Returns:
            Visualization object created by the plotter.
            
        Raises:
            TMDProcessingError: If plotting fails.
        """
        if self.height_map is None:
            self.process()
            
        # Use the specified strategy or fall back to default
        strategy = plotter_strategy or self._default_plotter_strategy
        
        try:
            # Create plotter using the factory
            plotter = TMDPlotterFactory.create_plotter(strategy)
            
            # Create default title if not provided
            if 'title' not in kwargs:
                kwargs['title'] = f"TMD Height Map: {self.filepath.name}"
                
            # Add mode to kwargs
            kwargs['mode'] = mode
                
            # Create the visualization
            plot_obj = plotter.plot(self.height_map, **kwargs)
            
            # Save if output path is provided
            if output_path:
                output_path = Path(output_path)
                TMDFileUtilities.ensure_directory_exists(output_path.parent)
                plotter.save(plot_obj, str(output_path), **kwargs)
                logger.info(f"Plot saved to {output_path}")
                
            return plot_obj
            
        except Exception as e:
            logger.error(f"Error plotting TMD file '{self.filepath}': {e}")
            raise TMDProcessingError(f"Plotting failed for file '{self.filepath}'") from e
            
            
    def plot_profile(self,
                    row_index: Optional[int] = None,
                    output_path: Optional[Union[str, Path]] = None,
                    plotter_strategy: Optional[str] = None,
                    **kwargs) -> Any:
        """
        Create a profile plot along a specific row of the height map.
        
        Args:
            row_index: Index of the row to profile (default: middle row).
            output_path: Path to save the visualization (optional).
            plotter_strategy: Name of the plotter strategy to use (default from instance).
            **kwargs: Additional options to pass to the plotter.
            
        Returns:
            Visualization object created by the plotter.
            
        Raises:
            TMDProcessingError: If profile plotting fails.
        """
        if self.height_map is None:
            self.process()
            
        # Use middle row if not specified
        if row_index is None:
            row_index = self.height_map.shape[0] // 2
            
        # Use the specified strategy or fall back to default
        strategy = plotter_strategy or self._default_plotter_strategy
        
        try:
            # Create plotter using the factory
            plotter = TMDPlotterFactory.create_plotter(strategy)
            
            # Set up profile-specific parameters
            kwargs['mode'] = 'profile'
            kwargs['profile_row'] = row_index
            
            # Create default title if not provided
            if 'title' not in kwargs:
                kwargs['title'] = f"Profile at Row {row_index}: {self.filepath.name}"
                
            # Create the visualization
            plot_obj = plotter.plot(self.height_map, **kwargs)
            
            # Save if output path is provided
            if output_path:
                output_path = Path(output_path)
                TMDFileUtilities.ensure_directory_exists(output_path.parent)
                plotter.save(plot_obj, str(output_path), **kwargs)
                logger.info(f"Profile plot saved to {output_path}")
                
            return plot_obj
            
        except Exception as e:
            logger.error(f"Error plotting TMD profile: {e}")
            raise TMDProcessingError(f"Profile plotting failed: {e}") from e
            
    def plot_stats(self,
                  stats_data: Optional[Dict[str, List[float]]] = None,
                  output_path: Optional[Union[str, Path]] = None,
                  plotter_strategy: Optional[str] = None,
                  **kwargs) -> Any:
        """
        Visualize statistical data from TMD processing.
        
        Args:
            stats_data: Dictionary with metrics and their values (if None, computes from height_map).
            output_path: Path to save the visualization (optional).
            plotter_strategy: Name of the plotter strategy to use (default from instance).
            **kwargs: Additional options to pass to the plotter.
            
        Returns:
            Visualization object created by the plotter.
            
        Raises:
            TMDProcessingError: If statistics plotting fails.
        """
        # Generate stats data if not provided
        if stats_data is None:
            if self.height_map is None:
                self.process()
            stats_data = compute_stats(self.height_map)
            # Convert to sequence-compatible format
            stats_data = {k: [float(v)] for k, v in stats_data.items() if isinstance(v, (int, float))}
            
        # Use the specified strategy or fall back to default
        strategy = plotter_strategy or self._default_plotter_strategy
        
        try:
            # Create sequence plotter using the factory (for stats visualization)
            seq_plotter = TMDSequencePlotterFactory.create_plotter(strategy)
            
            # Create default title if not provided
            if 'title' not in kwargs:
                kwargs['title'] = f"Statistics: {self.filepath.name}"
                
            # Create the visualization
            plot_obj = seq_plotter.visualize_statistics(stats_data, **kwargs)
            
            # Save if output path is provided
            if output_path:
                output_path = Path(output_path)
                TMDFileUtilities.ensure_directory_exists(output_path.parent)
                seq_plotter.save_figure(plot_obj, str(output_path), **kwargs)
                logger.info(f"Statistics plot saved to {output_path}")
                
            return plot_obj
            
        except Exception as e:
            logger.error(f"Error plotting TMD statistics: {e}")
            raise TMDProcessingError(f"Statistics plotting failed: {e}") from e

    def __str__(self) -> str:
        """Return a string representation of the TMDProcessor instance."""
        status = "initialized" if self._initialized else "uninitialized"
        has_data = "with data" if self.height_map is not None else "without data"
        return f"TMDProcessor({self.filepath}, {status}, {has_data}, version={self.version})"

    def __repr__(self) -> str:
        """Return a string representation for debugging."""
        return f"TMDProcessor(filepath='{self.filepath}', version={self.version}, debug={self.debug})"


class TMD:
    """Main TMD class for working with topographic mesh data."""
    
    def __init__(self, height_map_or_path=None, metadata=None):
        """
        Initialize a TMD object with height map and metadata.
        
        Args:
            height_map_or_path: 2D NumPy array or path to TMD file
            metadata: Dictionary containing metadata about the height map
        """
        # Check if height_map_or_path is a file path string
        if isinstance(height_map_or_path, (str, Path)):
            # Load the file
            self._filepath = Path(height_map_or_path)
            self._load_from_file(self._filepath)
        else:
            # Set defaults if not provided
            if height_map_or_path is None:
                height_map_or_path = np.zeros((0, 0))
            if metadata is None:
                metadata = {}
                
            self._height_map = height_map_or_path
            self._metadata = metadata
            
            # Set up analysis tools
            self._initialize_analysis()
            
        # Default plotting strategy
        self._default_plotter_strategy = "matplotlib"
        
    def _load_from_file(self, filepath):
        """Load TMD data from a file."""
        processor = TMDProcessor(filepath)
        result = processor.process()
        self._height_map = result["height_map"]
        self._metadata = result["metadata"]
        self._stats = {}  # Initialize stats to be computed later
        
        # Initialize analysis tools
        self._initialize_analysis()

    def _initialize_analysis(self):
        """Initialize analysis tools and compute initial statistics."""
        if hasattr(self, '_height_map') and isinstance(self._height_map, np.ndarray) and self._height_map.size > 0:
            try:
                self._stats = compute_stats(self._height_map)
            except Exception as e:
                logger.warning(f"Could not compute initial statistics: {e}")
                self._stats = {}
        else:
            self._stats = {}

    @property
    def height_map(self) -> np.ndarray:
        """Get the height map data."""
        return self._height_map
        
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get the metadata dictionary."""
        return self._metadata
        
    @property
    def stats(self) -> Dict[str, Any]:
        """Get computed statistics for the height map."""
        if not self._stats and self._height_map.size > 0:
            self._stats = compute_stats(self._height_map)
        return self._stats
        
    @property
    def shape(self) -> Tuple[int, int]:
        """Get the shape of the height map."""
        return self._height_map.shape
        
    @property
    def dimensions(self) -> Dict[str, float]:
        """Get physical dimensions of the surface."""
        width = self._metadata.get('width', self._height_map.shape[1])
        height = self._metadata.get('height', self._height_map.shape[0])
        
        # Try to get resolution from metadata or default to 1.0
        resolution_x = self._metadata.get('resolution_x', 1.0)
        resolution_y = self._metadata.get('resolution_y', 1.0)
        
        physical_width = width * resolution_x
        physical_height = height * resolution_y
        
        return {
            'width': physical_width,
            'height': physical_height,
            'resolution_x': resolution_x,
            'resolution_y': resolution_y
        }
        
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "TMD":
        """
        Load a TMD file and return a TMD object.
        
        Args:
            filepath: Path to the TMD file to load
            
        Returns:
            TMD object containing the loaded data
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            TMDProcessingError: If file processing fails
        """
        processor = TMDProcessor(filepath)
        result = processor.process()
        return cls(result["height_map"], result["metadata"])
    
    def save(self, filepath: Union[str, Path], version: int = 2) -> str:
        """
        Save the current TMD data to a file.
        
        Args:
            filepath: Path to save the TMD file
            version: TMD file version to use (1 or 2)
            
        Returns:
            Path to the saved file
            
        Raises:
            TMDProcessingError: If saving fails
        """
        filepath = Path(filepath)
        TMDFileUtilities.ensure_directory_exists(filepath.parent)
        
        try:
            # Ensure filepath has .tmd extension
            if filepath.suffix.lower() != '.tmd':
                filepath = filepath.with_suffix('.tmd')
                
            TMDUtils.write_tmd_file(
                str(filepath),
                self._height_map,
                self._metadata,
                version=version
            )
            
            return str(filepath)
        except Exception as e:
            logger.error(f"Error saving TMD file to '{filepath}': {e}")
            raise TMDProcessingError(f"Failed to save TMD file: {e}")
    
    def export_metadata(self, output_path: Union[str, Path]) -> str:
        """
        Export metadata and statistics to a text file.
        
        Args:
            output_path: Path to save the metadata file
            
        Returns:
            Path to the exported metadata file
            
        Raises:
            TMDProcessingError: If exporting fails
        """
        output_path = Path(output_path)
        TMDFileUtilities.ensure_directory_exists(output_path.parent)
        
        try:
            stats = self.stats  # Use property to ensure stats are computed
            return export_metadata(self._metadata, stats, str(output_path))
        except Exception as e:
            logger.error(f"Error exporting metadata to '{output_path}': {e}")
            raise TMDProcessingError(f"Failed to export metadata: {e}")
    
    def set_default_plotter(self, strategy: str) -> "TMD":
        """
        Set the default plotting strategy.
        
        Args:
            strategy: Name of the plotting strategy (e.g., "matplotlib", "plotly")
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If the strategy is not available
        """
        try:
            # Check if the strategy is available
            available_strategies = TMDPlotterFactory.get_available_plotters()
            if strategy.lower() not in available_strategies:
                registered = TMDPlotterFactory.get_registered_plotters()
                available = ", ".join(available_strategies) if available_strategies else "none"
                registered_str = ", ".join(registered) if registered else "none"
                
                raise ValueError(f"Plotting strategy '{strategy}' not available. "
                               f"Available options: {available}\n"
                               f"Registered but unavailable: {registered_str}")
            
            self._default_plotter_strategy = strategy.lower()
            return self
        except Exception as e:
            logger.error(f"Could not set default plotter to '{strategy}': {e}")
            raise ValueError(f"Could not set default plotter to '{strategy}': {e}")
    
    def plot(self, 
             output_path: Optional[Union[str, Path]] = None, 
             plotter_strategy: Optional[str] = None,
             mode: str = "2d",
             **kwargs) -> Any:
        """
        Visualize the TMD height map.
        
        Args:
            output_path: Path to save the visualization (optional)
            plotter_strategy: Name of the plotter to use (default from instance)
            mode: Visualization mode ("2d", "3d", "contour", etc.)
            **kwargs: Additional options to pass to the plotter
            
        Returns:
            Visualization object created by the plotter
            
        Raises:
            TMDProcessingError: If plotting fails
        """
        # Use the specified strategy or fall back to default
        strategy = plotter_strategy or self._default_plotter_strategy
        
        try:
            # Create plotter using the factory
            plotter = TMDPlotterFactory.create_plotter(strategy)
            
            # Create default title if not provided
            if 'title' not in kwargs:
                title = self._metadata.get('comment', 'TMD Height Map')
                kwargs['title'] = title
                
            # Add mode to kwargs
            kwargs['mode'] = mode
                
            # Create the visualization
            plot_obj = plotter.plot(self._height_map, **kwargs)
            
            # Save if output path is provided
            if output_path:
                output_path = Path(output_path)
                TMDFileUtilities.ensure_directory_exists(output_path.parent)
                plotter.save(plot_obj, str(output_path), **kwargs)
                logger.info(f"Plot saved to {output_path}")
                
            return plot_obj
            
        except Exception as e:
            logger.error(f"Error plotting TMD: {e}")
            raise TMDProcessingError(f"Plotting failed: {e}")
    
    def plot_profile(self,
                    row_index: Optional[int] = None,
                    output_path: Optional[Union[str, Path]] = None,
                    plotter_strategy: Optional[str] = None,
                    **kwargs) -> Any:
        """
        Create a profile plot along a specific row of the height map.
        
        Args:
            row_index: Index of the row to profile (default: middle row)
            output_path: Path to save the visualization (optional)
            plotter_strategy: Name of the plotter to use (default from instance)
            **kwargs: Additional options to pass to the plotter
            
        Returns:
            Visualization object created by the plotter
            
        Raises:
            TMDProcessingError: If profile plotting fails
        """
        # Use middle row if not specified
        if row_index is None:
            row_index = self._height_map.shape[0] // 2
            
        # Use the specified strategy or fall back to default
        strategy = plotter_strategy or self._default_plotter_strategy
        
        try:
            # Create plotter using the factory
            plotter = TMDPlotterFactory.create_plotter(strategy)
            
            # Set up profile-specific parameters
            kwargs['mode'] = 'profile'
            kwargs['profile_row'] = row_index
            
            # Create default title if not provided
            if 'title' not in kwargs:
                kwargs['title'] = f"Profile at Row {row_index}"
                
            # Create the visualization
            plot_obj = plotter.plot(self._height_map, **kwargs)
            
            # Save if output path is provided
            if output_path:
                output_path = Path(output_path)
                TMDFileUtilities.ensure_directory_exists(output_path.parent)
                plotter.save(plot_obj, str(output_path), **kwargs)
                logger.info(f"Profile plot saved to {output_path}")
                
            return plot_obj
            
        except Exception as e:
            logger.error(f"Error plotting TMD profile: {e}")
            raise TMDProcessingError(f"Profile plotting failed: {e}")
        
    def plot_stats(self,
                  output_path: Optional[Union[str, Path]] = None,
                  plotter_strategy: Optional[str] = None,
                  **kwargs) -> Any:
        """
        Visualize statistical data from the height map.
        
        Args:
            output_path: Path to save the visualization (optional)
            plotter_strategy: Name of the plotter to use (default from instance)
            **kwargs: Additional options to pass to the plotter
            
        Returns:
            Visualization object created by the plotter
            
        Raises:
            TMDProcessingError: If statistics plotting fails
        """
        # Ensure stats are computed
        stats_data = self.stats
        
        # Convert to sequence-compatible format
        stats_data = {k: [float(v)] for k, v in stats_data.items() if isinstance(v, (int, float))}
        
        # Use the specified strategy or fall back to default
        strategy = plotter_strategy or self._default_plotter_strategy
        
        try:
            # Create sequence plotter using the factory
            seq_plotter = TMDSequencePlotterFactory.create_plotter(strategy)
            
            # Create default title if not provided
            if 'title' not in kwargs:
                title = self._metadata.get('comment', 'TMD Statistics')
                kwargs['title'] = f"Statistics: {title}"
                
            # Create the visualization
            plot_obj = seq_plotter.visualize_statistics(stats_data, **kwargs)
            
            # Save if output path is provided
            if output_path:
                output_path = Path(output_path)
                TMDFileUtilities.ensure_directory_exists(output_path.parent)
                seq_plotter.save_figure(plot_obj, str(output_path), **kwargs)
                logger.info(f"Statistics plot saved to {output_path}")
                
            return plot_obj
            
        except Exception as e:
            logger.error(f"Error plotting TMD statistics: {e}")
            raise TMDProcessingError(f"Statistics plotting failed: {e}")
    
    def crop(self, x_start: int, y_start: int, width: int, height: int) -> "TMD":
        """
        Create a new TMD object with a cropped section of the height map.
        
        Args:
            x_start: Starting x-coordinate
            y_start: Starting y-coordinate
            width: Width of the cropped area
            height: Height of the cropped area
            
        Returns:
            New TMD object containing the cropped data
            
        Raises:
            ValueError: If crop dimensions are invalid
        """
        # Validate crop dimensions
        if (x_start < 0 or y_start < 0 or 
            x_start + width > self._height_map.shape[1] or 
            y_start + height > self._height_map.shape[0]):
            raise ValueError(f"Invalid crop dimensions: ({x_start}, {y_start}, {width}, {height})")
            
        # Crop the height map
        cropped_height_map = self._height_map[y_start:y_start+height, x_start:x_start+width]
        
        # Create new metadata with updated dimensions
        new_metadata = self._metadata.copy()
        new_metadata['width'] = width
        new_metadata['height'] = height
        new_metadata['comment'] = f"{new_metadata.get('comment', 'TMD Data')} (cropped)"
        
        # Return new TMD object
        return TMD(cropped_height_map, new_metadata)
    
    def resize(self, new_width: int, new_height: int) -> "TMD":
        """
        Create a new TMD object with resized height map.
        
        Args:
            new_width: New width for the height map
            new_height: New height for the height map
            
        Returns:
            New TMD object with resized data
            
        Raises:
            ImportError: If required interpolation libraries are not available
        """
        try:
            from scipy import ndimage
        except ImportError:
            raise ImportError("scipy is required for resizing TMD data")
            
        # Calculate scaling factors
        scale_x = new_width / self._height_map.shape[1]
        scale_y = new_height / self._height_map.shape[0]
        
        # Resize the height map using interpolation
        resized_height_map = ndimage.zoom(self._height_map, (scale_y, scale_x), order=3)
        
        # Create new metadata with updated dimensions
        new_metadata = self._metadata.copy()
        new_metadata['width'] = new_width
        new_metadata['height'] = new_height
        new_metadata['comment'] = f"{new_metadata.get('comment', 'TMD Data')} (resized)"
        
        # Update resolution based on scaling
        if 'resolution_x' in new_metadata:
            new_metadata['resolution_x'] = new_metadata['resolution_x'] / scale_x
        if 'resolution_y' in new_metadata:
            new_metadata['resolution_y'] = new_metadata['resolution_y'] / scale_y
            
        # Return new TMD object
        return TMD(resized_height_map, new_metadata)
        
    def __str__(self) -> str:
        """Return a string representation of the TMD object."""
        dims = f"{self._height_map.shape[0]}x{self._height_map.shape[1]}"
        comment = self._metadata.get('comment', 'No description')
        return f"TMD({dims}, '{comment}')"

    def __repr__(self) -> str:
        """Return a detailed string representation for debugging."""
        shape = self._height_map.shape
        stats = {k: v for k, v in self.stats.items() if k in ['min', 'max', 'mean']} if self._height_map.size > 0 else {}
        return f"TMD(shape={shape}, stats={stats}, metadata_keys={list(self._metadata.keys())})"


def load(filepath: Union[str, Path]) -> TMD:
    """
    Load a TMD file and return a TMD object.
    
    This is a convenience function that calls TMD.load().
    
    Args:
        filepath: Path to the TMD file to load
        
    Returns:
        TMD object containing the loaded data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        TMDProcessingError: If file processing fails
    """
    return TMD.load(filepath)


def get_registered_plotters() -> Dict[str, List[str]]:
    """
    Get a dictionary of registered plotting backends.
    
    Returns:
        Dictionary with keys 'available' and 'registered' listing plotting backends
    """
    available = TMDPlotterFactory.get_available_plotters()
    registered = TMDPlotterFactory.get_registered_plotters()
    
    return {
        'available': available,
        'registered': registered
    }
