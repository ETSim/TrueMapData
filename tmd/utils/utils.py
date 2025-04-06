#!/usr/bin/env python3
"""
Core utility functions for TMD file processing and analysis.

This module provides utilities for working with True Map Data (TMD) files,
including version detection, reading, writing, and formatting binary data.
"""

import logging
import os
import struct
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, BinaryIO, List

# Define logger before it's used
logger = logging.getLogger(__name__)

# Required dependencies
import numpy as np

# Import exceptions from the dedicated exceptions module
from tmd.utils.exceptions import TMDFileError, TMDVersionError, TMDDataError

# Rich text formatting library for advanced console output
from rich import print as rprint
from rich.console import Console

# Initialize rich console
console = Console()


class TMDUtils:
    """
    Utility class for TMD file processing and analysis.

    Contains methods for creating hexdumps, reading null-terminated strings,
    detecting TMD file versions, processing TMD files, and writing TMD files.
    """

    @staticmethod
    def hexdump(
        bytes_data: bytes,
        start: int = 0,
        length: Optional[int] = None,
        width: int = 16,
        show_ascii: bool = True,
    ) -> str:
        """
        Create a formatted hexdump of the bytes data.

        Args:
            bytes_data: The bytes to format.
            start: Starting offset for the addresses.
            length: Number of bytes to dump (None = all).
            width: Number of bytes per row.
            show_ascii: Whether to include ASCII representation.

        Returns:
            Formatted hexdump string.

        Examples:
            >>> TMDUtils.hexdump(b'Hello, World!', width=8)
            '00000000:  48 65 6c 6c 6f 2c 20 57  |Hello, W|\\n00000008:  6f 72 6c 64 21           |orld!|'
        """
        if not bytes_data:
            return "(empty)"

        if length is None:
            length = len(bytes_data) - start

        # Make sure we only process the specified length and bounds
        if start < 0:
            start = 0
        if start >= len(bytes_data):
            return "(invalid start offset)"
        
        data_to_process = bytes_data[start:start + length]

        result = []
        for i in range(0, len(data_to_process), width):
            chunk = data_to_process[i:i + width]
            # Format each byte as a two-digit hex, join with spaces
            hex_part = " ".join(f"{b:02x}" for b in chunk)
            
            # Calculate proper padding: each byte takes 2 chars + 1 space
            # We need to ensure consistent width regardless of how many bytes in this row
            padding = width * 3 - len(hex_part) - (0 if len(chunk) == width else 1)
            line = f"{start + i:08x}:  {hex_part}{' ' * padding}"

            if show_ascii:
                ascii_part = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
                line += f"  |{ascii_part}|"

            result.append(line)

        return "\n".join(result)

    @staticmethod
    def read_null_terminated_string(file_handle: BinaryIO, chunk_size: int = 256) -> str:
        """
        Read a null-terminated ASCII string from a binary file.

        Args:
            file_handle: Open file handle.
            chunk_size: Maximum string length to read.

        Returns:
            Decoded string up to the null terminator.

        Raises:
            IOError: If there's an error reading from the file.
        """
        try:
            pos = file_handle.tell()
            chunk = file_handle.read(chunk_size)
            
            if not chunk:  # End of file
                return ""
                
            null_index = chunk.find(b"\0")
            if null_index == -1:
                return chunk.decode("ascii", errors="ignore")
            else:
                file_handle.seek(pos + null_index + 1)
                return chunk[:null_index].decode("ascii", errors="ignore")
        except IOError as e:
            logger.error(f"Error reading null-terminated string: {e}")
            raise

    @staticmethod
    def detect_tmd_version(file_path: Union[str, Path]) -> int:
        """
        Determine the TMD file version based on header content.

        Args:
            file_path: Path to the TMD file.

        Returns:
            Integer version (1 or 2).

        Raises:
            FileNotFoundError: If the file does not exist.
            TMDVersionError: If the file header is invalid or cannot be read.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            with open(file_path, "rb") as f:
                header_bytes = f.read(64)
                
                if len(header_bytes) < 16:
                    raise TMDVersionError(f"File too small to determine version: {file_path}")
                
                header_text = header_bytes.decode("ascii", errors="replace")
                logger.debug(f"Header text: {header_text}")

                # Check for explicit version indicators
                if "v2.0" in header_text:
                    return 2

                # Check for standard headers
                if "Binary TrueMap Data File" in header_text:
                    # Most likely v1 if it has the standard header
                    return 1
                    
                # Try to infer version from header structure
                magic_bytes = header_bytes[:4]
                if magic_bytes == b"TMD\0" or magic_bytes == b"TMD\x00":
                    # Check version field at position 4-8
                    try:
                        version = struct.unpack("<I", header_bytes[4:8])[0]
                        if 1 <= version <= 2:
                            return version
                    except struct.error:
                        pass
                
                # Fallback to v1 if no other indicators found
                return 1
                
        except (IOError, UnicodeDecodeError) as e:
            logger.error(f"Error reading TMD file header: {e}")
            raise TMDVersionError(f"Could not determine TMD file version: {e}") from e

    @staticmethod
    def process_tmd_file(
        file_path: Union[str, Path],
        force_offset: Optional[Tuple[float, float]] = None,
        debug: bool = False,
    ) -> Tuple[Dict[str, Any], np.ndarray]:
        """
        Process a TMD file and extract metadata and height map.
        Handles both v1 and v2 file formats, plus GelSight format.

        Args:
            file_path: Path to the TMD file.
            force_offset: Optional tuple (x_offset, y_offset) to override file values.
            debug: Whether to print debug information.

        Returns:
            Tuple of (metadata_dict, height_map_array).

        Raises:
            FileNotFoundError: If the file does not exist.
            TMDVersionError: If there's an issue with file version detection.
            TMDDataError: If there's an error processing the TMD data.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            # Detect version from file header
            version = TMDUtils.detect_tmd_version(file_path)
            if debug:
                print(f"Detected TMD file version: {version}")

            # Read and process file content
            with open(file_path, "rb") as f:
                metadata, height_map = TMDUtils._read_tmd_file(
                    f, version, force_offset, debug
                )
                
            # Add file path to metadata
            metadata["file_path"] = str(file_path)
            
            return metadata, height_map
            
        except TMDVersionError:
            # Re-raise version errors
            raise
        except Exception as e:
            logger.error(f"Error processing TMD file: {e}")
            raise TMDDataError(f"Failed to process TMD file: {e}") from e

    @staticmethod
    def _read_tmd_file(
        f: BinaryIO,
        version: int,
        force_offset: Optional[Tuple[float, float]] = None,
        debug: bool = False,
    ) -> Tuple[Dict[str, Any], np.ndarray]:
        """
        Read a TMD file from an open file handle.
        
        Args:
            f: Open binary file handle.
            version: TMD file version.
            force_offset: Optional tuple (x_offset, y_offset) to override file values.
            debug: Whether to print debug information.
            
        Returns:
            Tuple of (metadata_dict, height_map_array).
            
        Raises:
            TMDDataError: If there's an error processing the TMD data.
        """
        # Initialize metadata with defaults
        metadata = {
            "version": version,
            "width": 1,
            "height": 1,
            "x_length": 10.0,
            "y_length": 10.0,
            "x_offset": 0.0,
            "y_offset": 0.0,
            "mmpp": 1.0,
            "comment": None,
            "px_off_x": 0,
            "px_off_y": 0,
        }
        
        # Position file pointer based on version
        try:
            if version == 1:
                f.seek(28)  # Offset for v1 files
                if debug:
                    print("⚠️ Detected v1 file format. Reading metadata...")
            elif version == 2:
                f.seek(32)  # Default for v2 files
                if debug:
                    print("⚠️ Detected v2 file format. Reading metadata...")
                
                # Read comment section
                try:
                    comment_bytes = f.read(24)
                    metadata["comment"] =  comment_bytes.decode("ascii").strip()
                    if debug and metadata["comment"]:
                        print(f"Comment: {metadata['comment']}")
                except Exception as e:
                    if debug:
                        print(f"Error reading comment: {e}")
                    # Ensure we're in the right position for the next block
                    metadata["comment"] = None
                    try:
                        f.read(24)
                        f.seek(33)
                    except Exception:
                        metadata["comment"] = None
                        f.seek(33)
            else:
                logger.error(f"Unsupported TMD file version: {version}")
                raise TMDDataError(f"Unsupported TMD file version: {version}")
            
            # Read and extract dimensions
            metadata.update(TMDUtils._read_dimensions(f, debug))
            
            # Read and extract spatial parameters
            metadata.update(TMDUtils._read_spatial_params(f, version, debug))
            
            # Apply forced offsets if provided
            if force_offset:
                metadata["x_offset"], metadata["y_offset"] = force_offset
                if debug:
                    print(f"Using forced offsets: x_offset={metadata['x_offset']}, y_offset={metadata['y_offset']}")
                    
            # Calculate derived values
            width, height = metadata["width"], metadata["height"]
            x_offset, y_offset = metadata["x_offset"], metadata["y_offset"]
            
            # Calculate mm per pixel and pixel offsets
            metadata["mmpp"] = metadata["x_length"] / width if width > 0 else 1.0
            metadata["px_off_x"] = int(round(x_offset / metadata["mmpp"])) if x_offset != 0 else 0
            metadata["px_off_y"] = int(round(y_offset / metadata["mmpp"])) if y_offset != 0 else 0
            
            if debug and (metadata["px_off_x"] != 0 or metadata["px_off_y"] != 0):
                print(f"Pixel offsets: x={metadata['px_off_x']}, y={metadata['px_off_y']}")
                
            # Read height map data
            height_map = TMDUtils._read_height_data(f, metadata, version, debug)
            
            return metadata, height_map
            
        except Exception as e:
            logger.error(f"Error reading TMD file: {e}")
            # Return default empty height map on error
            logger.error((metadata["height"], metadata["width"]))
            return metadata, np.zeros((metadata["height"], metadata["width"]), dtype=np.float32)

    @staticmethod
    def _read_dimensions(f: BinaryIO, debug: bool = False) -> Dict[str, Any]:
        """
        Read width and height dimensions from TMD file.
        
        Args:
            f: Open binary file handle.
            debug: Whether to print debug information.
            
        Returns:
            Dictionary with width and height.
        """
        dimensions = {"width": 1, "height": 1}
        
        try:
            width_bytes = f.read(4)
            height_bytes = f.read(4)

            if len(width_bytes) < 4 or len(height_bytes) < 4:
                if debug:
                    print("Warning: File too small to read dimensions properly.")
                return dimensions
                
            try:
                dimensions["width"] = struct.unpack("<I", width_bytes)[0]
                dimensions["height"] = struct.unpack("<I", height_bytes)[0]
                if debug:
                    print(f"Dimensions: {dimensions['width']} x {dimensions['height']}")
            except struct.error:
                if debug:
                    print("Error unpacking width/height values")
        except Exception as e:
            if debug:
                print(f"Error parsing dimensions: {e}")
                
        return dimensions

    @staticmethod
    def _read_spatial_params(f: BinaryIO, version: int, debug: bool = False) -> Dict[str, float]:
        """
        Read spatial parameters from TMD file.
        
        Args:
            f: Open binary file handle.
            version: TMD file version.
            debug: Whether to print debug information.
            
        Returns:
            Dictionary with spatial parameters.
        """
        params = {
            "x_length": 10.0,  # Default values
            "y_length": 10.0,
            "x_offset": 0.0,
            "y_offset": 0.0
        }
        
        try:
            x_length_bytes = f.read(4)
            y_length_bytes = f.read(4)
            x_offset_bytes = b'\x00\x00\x00\x00'  # Default zeroes
            y_offset_bytes = b'\x00\x00\x00\x00'
            
            if version == 2:
                x_offset_bytes = f.read(4)
                y_offset_bytes = f.read(4)

            # Extract values where possible
            if len(x_length_bytes) == 4:
                params["x_length"] = struct.unpack("<f", x_length_bytes)[0]
            if len(y_length_bytes) == 4:
                params["y_length"] = struct.unpack("<f", y_length_bytes)[0]
            if len(x_offset_bytes) == 4 and version == 2:
                params["x_offset"] = struct.unpack("<f", x_offset_bytes)[0]
            if len(y_offset_bytes) == 4 and version == 2:
                params["y_offset"] = struct.unpack("<f", y_offset_bytes)[0]

            if debug:
                print(f"X Length: {params['x_length']}, Y Length: {params['y_length']}")
                print(f"X Offset: {params['x_offset']}, Y Offset: {params['y_offset']}")
        except Exception as e:
            if debug:
                print(f"Error reading spatial parameters: {e}. Using defaults.")
                
        return params

    @staticmethod
    def _read_height_data(
        f: BinaryIO, 
        metadata: Dict[str, Any], 
        version: int, 
        debug: bool = False
    ) -> np.ndarray:
        """
        Read height map data from TMD file.
        
        Args:
            f: Open binary file handle.
            metadata: Metadata dictionary with dimensions.
            version: TMD file version.
            debug: Whether to print debug information.
            
        Returns:
            2D numpy array with height map data.
        """
        width = metadata["width"]
        height = metadata["height"]
        px_off_x = metadata["px_off_x"]
        px_off_y = metadata["px_off_y"]
        
        try:
            # For v1 files, read all at once
            if version == 1:
                return TMDUtils._read_v1_height_data(f, width, height, px_off_x, px_off_y, debug)
            else:
                return TMDUtils._read_v2_height_data(f, width, height, px_off_x, px_off_y, debug)
        except Exception as e:
            if debug:
                print(f"Error parsing height map data: {e}. Creating empty height map.")
            return np.zeros((height, width), dtype=np.float32)

    @staticmethod
    def _read_v1_height_data(
        f: BinaryIO, width: int, height: int, px_off_x: int, px_off_y: int, debug: bool = False
    ) -> np.ndarray:
        """
        Read height map data from v1 TMD file.
        
        Args:
            f: Open binary file handle.
            width: Width of the height map.
            height: Height of the height map.
            px_off_x: X offset in pixels.
            px_off_y: Y offset in pixels.
            debug: Whether to print debug information.
            
        Returns:
            2D numpy array with height map data.
        """
        expected_data_size = width * height * 4  # 4 bytes per float
        height_data = f.read()

        if debug:
            print(f"Expected {expected_data_size} bytes of height data, read {len(height_data)} bytes")

        # Handle data size mismatches
        if len(height_data) < expected_data_size:
            if debug:
                print(f"Padding height data: expected {expected_data_size}, got {len(height_data)}")
            height_data = height_data.ljust(expected_data_size, b"\0")
        elif len(height_data) > expected_data_size:
            if debug:
                print(f"Trimming height data: expected {expected_data_size}, got {len(height_data)}")
            height_data = height_data[:expected_data_size]

        # Convert to numpy array
        height_map_data = np.frombuffer(height_data, dtype=np.float32)

        # Apply offsets if needed
        if px_off_x != 0 or px_off_y != 0:
            full_width = width + px_off_x
            full_height = height + px_off_y
            height_map = np.zeros((full_height, full_width), dtype=np.float32)

            # Reshape data and place in correct position
            data_reshaped = height_map_data.reshape((height, width))
            height_map[px_off_y:px_off_y + height, px_off_x:px_off_x + width] = data_reshaped
            return height_map
        else:
            # No offset, just reshape
            return height_map_data.reshape((height, width))

    @staticmethod
    def _read_v2_height_data(
        f: BinaryIO, width: int, height: int, px_off_x: int, px_off_y: int, debug: bool = False
    ) -> np.ndarray:
        """
        Read height map data from v2 TMD file.
        
        Args:
            f: Open binary file handle.
            width: Width of the height map.
            height: Height of the height map.
            px_off_x: X offset in pixels.
            px_off_y: Y offset in pixels.
            debug: Whether to print debug information.
            
        Returns:
            2D numpy array with height map data.
        """
        # With offsets, read row by row
        if px_off_x != 0 or px_off_y != 0:
            full_width = width + px_off_x
            full_height = height + px_off_y
            height_map = np.zeros((full_height, full_width), dtype=np.float32)

            # Read each row and position with offset
            for y in range(height):
                row_data = f.read(width * 4)
                if len(row_data) != width * 4:
                    if debug:
                        print(f"Warning: Row {y} - Expected {width * 4} bytes, got {len(row_data)}")
                    row_data = row_data.ljust(width * 4, b"\0")

                row_floats = np.frombuffer(row_data, dtype=np.float32)
                height_map[y + px_off_y, px_off_x:px_off_x + width] = row_floats
                
            return height_map
        else:
            # No offset, read as a block
            height_data = f.read(width * height * 4)

            if len(height_data) < width * height * 4:
                if debug:
                    print(f"Warning: Expected {width * height * 4} bytes, got {len(height_data)}")
                height_data = height_data.ljust(width * height * 4, b"\0")
            elif len(height_data) > width * height * 4:
                if debug:
                    print("Warning: Extra data detected. Trimming.")
                height_data = height_data[:width * height * 4]

            return np.frombuffer(height_data, dtype=np.float32).reshape((height, width))

    @staticmethod
    def write_tmd_file(
        height_map: np.ndarray,
        output_path: Union[str, Path],
        comment: str = "Created by TrueMap v6\n",
        x_length: float = 10.0,
        y_length: float = 10.0,
        x_offset: float = 0.0,
        y_offset: float = 0.0,
        version: int = 2,
        debug: bool = False,
    ) -> str:
        """
        Write a height map to a TMD file.

        Args:
            height_map: 2D numpy array of height values.
            output_path: Path where to save the TMD file.
            comment: Comment to include in the file.
            x_length: Physical length in X direction.
            y_length: Physical length in Y direction.
            x_offset: X-axis offset.
            y_offset: Y-axis offset.
            version: TMD version (1 or 2).
            debug: Whether to print debug information.

        Returns:
            Path to the created file.
            
        Raises:
            TMDFileError: If there's an error creating the TMD file.
            ValueError: If height_map is not a valid 2D numpy array.
        """
        # Validate input
        if not isinstance(height_map, np.ndarray) or height_map.ndim != 2:
            raise ValueError("Height map must be a 2D numpy array")
            
        if version not in (1, 2):
            raise ValueError(f"Unsupported TMD version: {version}. Must be 1 or 2.")
            
        output_path = Path(output_path)
        
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Get height map dimensions (rows, cols) = (height, width)
            height, width = height_map.shape

            with open(output_path, "wb") as f:
                if debug:
                    print(f"Writing TMD file v{version} to {output_path}")

                if version == 2:
                    TMDUtils._write_v2_header(f, comment, debug)
                else:
                    TMDUtils._write_v1_header(f, debug)

                # Write dimensions: width and height (4 bytes each, little-endian)
                f.write(struct.pack("<II", width, height))

                # Write spatial info: x_length, y_length, x_offset, y_offset (each as 4-byte float)
                f.write(struct.pack("<ffff", x_length, y_length, x_offset, y_offset))

                # Write the height map data (float32 values)
                height_map_flat = height_map.astype(np.float32).flatten()
                f.write(height_map_flat.tobytes())

            if debug:
                print(f"Dimensions: {width} x {height}, Spatial info: {x_length}, {y_length}, {x_offset}, {y_offset}")
                print(f"Successfully wrote TMD file: {output_path}")
                print(f"File size: {output_path.stat().st_size} bytes")

            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error writing TMD file: {e}")
            raise TMDFileError(f"Failed to write TMD file: {e}") from e

    @staticmethod
    def _write_v1_header(f: BinaryIO, debug: bool = False) -> None:
        """
        Write v1 TMD file header.
        
        Args:
            f: Open binary file handle.
            debug: Whether to print debug information.
        """
        header = "Binary TrueMap Data File\r\n"
        header_bytes = header.encode("ascii")
        remaining_header = 28 - len(header_bytes)  # v1 metadata starts at 28
        
        if remaining_header > 0:
            header_bytes += b"\0" * remaining_header
            
        f.write(header_bytes[:28])
        
        if debug:
            print(f"Wrote v1 header ({len(header_bytes[:28])} bytes)")

    @staticmethod
    def _write_v2_header(f: BinaryIO, comment: str, debug: bool = False) -> None:
        """
        Write v2 TMD file header with comment.
        
        Args:
            f: Open binary file handle.
            comment: Comment to include in header.
            debug: Whether to print debug information.
        """
        # Write the standard header
        header = "Binary TrueMap Data File v2.0\n\r"
        header_comment = comment if comment else "Created by TrueMap v6\n"

        # Ensure header_comment ends with newline
        if not header_comment.endswith("\n"):
            header_comment += "\n"

        # Write header and pad to 32 bytes with nulls if needed
        header_bytes = header.encode("ascii")
        remaining_header = 32 - len(header_bytes)
        
        if debug:
            print(f"Remaining header space: {remaining_header} bytes")
            
        if remaining_header > 0:
            header_bytes += b"\0" * remaining_header
            
        f.write(header_bytes[:32])  # Truncate if too long

        # Write comment and pad to 24 bytes
        comment_bytes = header_comment.encode("ascii")
        remaining_comment = 24 - len(comment_bytes)
        
        if debug:
            print(f"Remaining comment space: {remaining_comment} bytes")
            
        if remaining_comment > 0:
            comment_bytes += b"\0" * remaining_comment
            
        f.write(comment_bytes[:24])
        
        if debug:
            print(f"Wrote v2 header with comment ({len(header_bytes[:32]) + len(comment_bytes[:24])} bytes)")

    @staticmethod
    def print_message(message: str, message_type: str = "info", use_rich: bool = None) -> None:
        """
        Print a formatted message with optional rich formatting.

        Args:
            message: The message to print
            message_type: Type of message (info, warning, error, success)
            use_rich: Override automatic rich detection (None = auto-detect)
        """
        # Determine whether to use rich
        use_rich = True if use_rich is None else use_rich
        
        if use_rich:
            if message_type == "warning":
                rprint(f"[bold yellow]Warning:[/bold yellow] {message}")
            elif message_type == "error":
                rprint(f"[bold red]Error:[/bold red] {message}")
            elif message_type == "success":
                rprint(f"[bold green]Success:[/bold green] {message}")
            else:
                rprint(f"[bold blue]Info:[/bold blue] {message}")
        else:
            prefix = {
                "warning": "Warning",
                "error": "Error",
                "success": "Success",
                "info": "Info"
            }.get(message_type, "")
            
            if prefix:
                print(f"{prefix}: {message}")
            else:
                print(message)

    @staticmethod
    def get_scipy_or_fallback():
        """
        Try to import scipy for advanced processing functions.
        
        Returns:
            A tuple of (scipy_module, has_scipy)
        """
        try:
            import scipy
            import scipy.ndimage
            return scipy, True
        except ImportError:
            TMDUtils.print_message(
                "scipy not found, using simple numpy downsampling. "
                "For better results, consider installing scipy.",
                "warning"
            )
            return None, False
    
    @staticmethod
    def downsample_array(array: np.ndarray, new_width: int, new_height: int, method: str = "bilinear") -> np.ndarray:
        """
        Downsample a 2D array to new dimensions using the specified method.
        
        Args:
            array: Input 2D array
            new_width: Target width
            new_height: Target height
            method: Interpolation method (nearest, bilinear, bicubic)
            
        Returns:
            Downsampled array
        """
        scipy_module, has_scipy = TMDUtils.get_scipy_or_fallback()
        
        if has_scipy:
            # Map method name to scipy order parameter
            order = {
                "nearest": 0,
                "bilinear": 1,
                "bicubic": 3
            }.get(method.lower(), 1)
            
            return scipy_module.ndimage.zoom(
                array,
                (new_height / array.shape[0], new_width / array.shape[1]),
                order=order
            )
        else:
            # Fallback to simple numpy interpolation (nearest neighbor)
            y_indices = np.linspace(0, array.shape[0] - 1, new_height).astype(np.int32)
            x_indices = np.linspace(0, array.shape[1] - 1, new_width).astype(np.int32)
            return array[y_indices[:, np.newaxis], x_indices]
    
    @staticmethod
    def quantize_array(array: np.ndarray, levels: int = 256) -> np.ndarray:
        """
        Quantize an array to reduce precision using a specified number of levels.
        
        Args:
            array: Input array to quantize
            levels: Number of quantization levels
            
        Returns:
            Quantized array with the same shape as input
        """
        if levels < 2:
            levels = 2  # Ensure at least 2 levels
        
        # Get data range
        data_min = np.min(array)
        data_max = np.max(array)
        
        # Check if range is valid
        if data_max <= data_min:
            return array  # No change needed
        
        # Normalize to 0-1 range
        normalized = (array - data_min) / (data_max - data_min)
        
        # Quantize to specified levels
        quantized_normalized = np.round(normalized * (levels - 1)) / (levels - 1)
        
        # Convert back to original range
        quantized = quantized_normalized * (data_max - data_min) + data_min
        
        return quantized