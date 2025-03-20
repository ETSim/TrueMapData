"""
TMD file processor module.

This module provides the TMDProcessor class for loading and analyzing TMD files.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from tmd.utils.utils import hexdump, process_tmd_file

# Configure logger
logger = logging.getLogger(__name__)


class TMDProcessor:
    """
    Class for processing and analyzing TMD (TrueMap Data) files.
    """

    def __init__(self, file_path: str):
        """
        Initialize a TMD file processor.

        Args:
            file_path: Path to the TMD file
        """
        self.file_path = file_path
        self.basename = os.path.basename(file_path)
        self.data = None
        self.debug = False
        self._stats_cache = {}

    def set_debug(self, debug: bool = True):
        """
        Set debug mode.

        Args:
            debug: Whether to enable debug mode

        Returns:
            self for method chaining
        """
        self.debug = debug
        return self

    def print_file_header(self, num_bytes: int = 64):
        """
        Print the file header in hexdump format for inspection.

        Args:
            num_bytes: Number of bytes to print

        Returns:
            None
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")

        with open(self.file_path, "rb") as f:
            header_bytes = f.read(num_bytes)

        # Print hexdump of header
        dump = hexdump(header_bytes)
        logger.info(f"File header hex dump:\n{dump}")

        # Print ASCII representation
        ascii_text = "".join([chr(b) if 32 <= b < 127 else "." for b in header_bytes])
        logger.info(f"File header ASCII: {ascii_text}")

    def process(self, force_offset: Optional[Tuple[float, float]] = None):
        """
        Process the TMD file to extract metadata and height map.

        Args:
            force_offset: Optional tuple (x_offset, y_offset) to override values in file

        Returns:
            Dict with extracted data or None if processing failed
        """
        logger.info("=" * 80)
        logger.info(f"Processing file: {self.file_path}")

        try:
            # Check if file exists
            if not Path(self.file_path).exists():
                logger.error(f"Error: File not found: {self.file_path}")
                return None

            # Check file size
            file_size = Path(self.file_path).stat().st_size
            if file_size < 64:  # Basic header size check
                logger.error(f"Error: File is too small to be valid: {file_size} bytes")
                return None

            # Print file header for debugging
            try:
                self.print_file_header()
            except Exception as e:
                logger.error(f"Error inspecting header: {str(e)}")

            # Process the file
            try:
                metadata, height_map = process_tmd_file(
                    self.file_path, force_offset=force_offset, debug=self.debug
                )

                # Store results
                self.data = {
                    "file_path": self.file_path,
                    "version": metadata["version"],
                    "header": "",  # For now, we don't store the raw header
                    "comment": metadata["comment"],
                    "width": metadata["width"],
                    "height": metadata["height"],
                    "x_length": metadata["x_length"],
                    "y_length": metadata["y_length"],
                    "x_offset": metadata["x_offset"],
                    "y_offset": metadata["y_offset"],
                    "height_map": height_map,
                }

                # Clear stats cache
                self._stats_cache = {}

                # Log successful processing
                logger.info("Successfully processed TMD file")
                logger.info(f"Version: {metadata['version']}")
                logger.info(f"Comment: {metadata['comment']}")
                logger.info(f"Dimensions: {metadata['width']} x {metadata['height']}")
                logger.info(
                    f"X length: {metadata['x_length']:.4f}, Y length: {metadata['y_length']:.4f}"
                )
                logger.info(
                    f"X offset: {metadata['x_offset']:.4f}, Y offset: {metadata['y_offset']:.4f}"
                )

                return self.data

            except Exception as e:
                logger.error(f"Error processing file: {str(e)}")
                return None

        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return None

    def get_height_map(self):
        """
        Get the processed height map.

        Returns:
            NumPy array containing height map or None if not processed
        """
        if self.data is None:
            return None
        return self.data.get("height_map")

    def get_metadata(self):
        """
        Get metadata from the processed file.

        Returns:
            Dict containing metadata (without height map)

        Raises:
            ValueError: If file hasn't been processed yet
        """
        if self.data is None:
            raise ValueError("File has not been processed yet. Call process() first.")

        metadata = {k: v for k, v in self.data.items() if k != "height_map"}
        return metadata

    def get_stats(self):
        """
        Get statistics about the height map.

        Returns:
            Dict containing various statistical measures

        Raises:
            ValueError: If file hasn't been processed yet
        """
        if self.data is None or self.get_height_map() is None:
            raise ValueError("File has not been processed yet. Call process() first.")

        # Use cached stats if available
        if self._stats_cache:
            return self._stats_cache

        height_map = self.get_height_map()

        # Calculate basic statistics - convert NumPy types to Python types
        stats = {
            "min": float(np.nanmin(height_map)),  # Convert to Python float
            "max": float(np.nanmax(height_map)),  # Convert to Python float
            "mean": float(np.nanmean(height_map)),  # Convert to Python float
            "median": float(np.nanmedian(height_map)),  # Convert to Python float
            "std": float(np.nanstd(height_map)),  # Convert to Python float
            "shape": tuple(map(int, height_map.shape)),  # Convert to Python tuple of ints
            "non_nan": int(np.count_nonzero(~np.isnan(height_map))),  # Convert to Python int
            "nan_count": int(np.count_nonzero(np.isnan(height_map))),  # Convert to Python int
        }

        # Cache the results
        self._stats_cache = stats

        return stats

    def export_metadata(self, output_path: str = None):
        """
        Export metadata and statistics to a text file.

        Args:
            output_path: Path to save the metadata file (default: same dir as TMD file)

        Returns:
            Path to the created metadata file
        """
        if self.data is None:
            raise ValueError("File has not been processed yet. Call process() first.")

        if output_path is None:
            # Set default output path
            tmd_dir = os.path.dirname(self.file_path)
            base_name = os.path.splitext(self.basename)[0]
            output_path = os.path.join(tmd_dir, f"{base_name}_metadata.txt")

        # Get metadata and statistics
        metadata = self.get_metadata()
        stats = self.get_stats()

        with open(output_path, "w") as f:
            # Write basic information
            f.write(f"TMD File: {self.basename}\n")
            f.write("=" * 40 + "\n\n")

            # Write metadata
            f.write("Metadata:\n")
            for key, value in metadata.items():
                if key not in ("file_path", "header"):
                    f.write(f"  {key}: {value}\n")

            # Write statistics
            f.write("\nHeight Map Statistics:\n")
            for key, value in stats.items():
                f.write(f"  {key}: {value}\n")

        logger.info(f"Metadata exported to {output_path}")
        return output_path
