"""
Core utility functions for TMD file processing and analysis.
"""

import logging
import os
import re
import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


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
        bytes_data: The bytes to format
        start: Starting offset for the addresses
        length: Number of bytes to dump (None = all)
        width: Number of bytes per row
        show_ascii: Whether to include ASCII representation

    Returns:
        Formatted hexdump string
    """
    if length is None:
        length = len(bytes_data) - start

    # Make sure we only process the specified length
    data_to_process = bytes_data[start : start + length]

    result = []
    for i in range(0, len(data_to_process), width):
        chunk = data_to_process[i : i + width]
        hex_part = " ".join(f"{b:02x}" for b in chunk)

        line = f"{start + i:08x}:  {hex_part:<{width * 3}}"

        if show_ascii:
            ascii_part = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
            line += f"  |{ascii_part}|"

        result.append(line)

    return "\n".join(result)


def read_null_terminated_string(file_handle, chunk_size=256):
    """
    Read a null-terminated ASCII string from a binary file.

    Args:
        file_handle: Open file handle
        chunk_size: Maximum string length to read

    Returns:
        Decoded string up to the null terminator
    """
    pos = file_handle.tell()
    chunk = file_handle.read(chunk_size)
    null_index = chunk.find(b"\0")
    if null_index == -1:
        return chunk.decode("ascii", errors="ignore")
    else:
        file_handle.seek(pos + null_index + 1)
        return chunk[:null_index].decode("ascii", errors="ignore")


def detect_tmd_version(file_path: str) -> int:
    """
    Determine the TMD file version based on header content.

    Args:
        file_path: Path to the TMD file

    Returns:
        Integer version (1 or 2)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "rb") as f:
        header_bytes = f.read(64)
        header_text = header_bytes.decode("ascii", errors="replace")
        logger.debug(f"Header text: {header_text}")

        # Check for version number in header
        if "v2.0" in header_text:
            return 2

        # If no explicit version, try to determine from structure
        if "Binary TrueMap Data File" in header_text:
            # Most likely v2 if it has the standard header
            return 1

    # Default to v1 if unable to determine
    return 1

def process_tmd_file(
    file_path: str,
    force_offset: Optional[Tuple[float, float]] = None,
    debug: bool = False,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Process a TMD file and extract metadata and height map.
    Handles both v1 and v2 file formats, plus GelSight format.

    Args:
        file_path: Path to the TMD file
        force_offset: Optional tuple (x_offset, y_offset) to override file values
        debug: Whether to print debug information

    Returns:
        Tuple of (metadata_dict, height_map_array)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Detect version from file header
    version = detect_tmd_version(file_path)
    if debug:
        print(f"Detected TMD file version: {version}")

    with open(file_path, "rb") as f:
        # Handle different versions
        if version == 1:
            f.seek(28)  # Offset for v1 files
            if debug:
                print("⚠️ Detected v1 file format. Reading metadata...")
            comment = None
        elif version == 2:
            f.seek(32)  # Default for v2 files
            if debug:
                print("⚠️ Detected v2 file format. Reading metadata...")
            try:
                comment = f.read(24).decode('ascii').strip()
                if debug and comment:
                    print(f"Comment: {comment}")
            except Exception:
                comment = None
                try:
                    f.read(24)
                    f.seek(33)
                except Exception:
                    comment = None
                    f.seek(33)
        
        # Read dimensions (width and height)
        try:
            width_bytes = f.read(4)
            height_bytes = f.read(4)
            
            if len(width_bytes) < 4 or len(height_bytes) < 4:
                if debug:
                    print("Warning: File too small to read dimensions properly.")
                width, height = 1, 1
            else:
                width = struct.unpack("<I", width_bytes)[0]
                height = struct.unpack("<I", height_bytes)[0]
                if debug:
                    print(f"Dimensions: {width} x {height}")
        except Exception as e:
            if debug:
                print(f"Error parsing dimensions: {e}")
            width, height = 1, 1
        
        # Read spatial parameters (physical dimensions)
        x_length = 10.0  # Default values
        y_length = 10.0
        x_offset = 0.0
        y_offset = 0.0
        
        try:
            x_length_bytes = f.read(4)
            y_length_bytes = f.read(4)
            if version == 2:
                x_offset_bytes = f.read(4)
                y_offset_bytes = f.read(4)
            
            if len(x_length_bytes) == 4:
                x_length = struct.unpack("<f", x_length_bytes)[0]
            if len(y_length_bytes) == 4:
                y_length = struct.unpack("<f", y_length_bytes)[0]
            if len(x_offset_bytes) == 4 and version == 2:
                x_offset = struct.unpack("<f", x_offset_bytes)[0]
            if len(y_offset_bytes) == 4 and version == 2:
                y_offset = struct.unpack("<f", y_offset_bytes)[0]
            
            if debug:
                print(f"X Length: {x_length}, Y Length: {y_length}")
                print(f"X Offset: {x_offset}, Y Offset: {y_offset}")
        except Exception as e:
            if debug:
                print(f"Error reading spatial parameters: {e}. Using defaults.")
        
        # Apply forced offsets if provided
        if force_offset:
            x_offset, y_offset = force_offset
            if debug:
                print(f"Using forced offsets: x_offset={x_offset}, y_offset={y_offset}")
        
        # Calculate mm per pixel and pixel offsets
        mmpp = x_length / width if width > 0 else 1.0
        px_off_x = int(round(x_offset / mmpp)) if x_offset != 0 else 0
        px_off_y = int(round(y_offset / mmpp)) if y_offset != 0 else 0
        
        if debug and (px_off_x != 0 or px_off_y != 0):
            print(f"Pixel offsets: x={px_off_x}, y={px_off_y}")
        
        # Read height map data based on file version
        try:
            # For v1 files or GelSight, read all at once
            if version == 1:
                # Calculate expected size and read all remaining data
                expected_data_size = width * height * 4  # 4 bytes per float
                height_data = f.read()
                
                if debug:
                    print(f"Expected {expected_data_size} bytes of height data, read {len(height_data)} bytes")
                
                # Handle data size mismatches
                if len(height_data) < expected_data_size:
                    if debug:
                        print(f"Padding height data: expected {expected_data_size}, got {len(height_data)}")
                    height_data = height_data.ljust(expected_data_size, b'\0')
                elif len(height_data) > expected_data_size:
                    if debug:
                        print(f"Trimming height data: expected {expected_data_size}, got {len(height_data)}")
                    height_data = height_data[:expected_data_size]
                
                # Convert to float array
                height_map_data = np.frombuffer(height_data, dtype=np.float32)
                
                # Apply offsets if needed
                if px_off_x != 0 or px_off_y != 0:
                    full_width = width + px_off_x
                    full_height = height + px_off_y
                    height_map = np.zeros((full_height, full_width), dtype=np.float32)
                    
                    # Reshape data and place in correct position
                    data_reshaped = height_map_data.reshape((height, width))
                    height_map[px_off_y:px_off_y+height, px_off_x:px_off_x+width] = data_reshaped
                else:
                    # No offset, just reshape
                    height_map = height_map_data.reshape((height, width))
            else:
                # For v2 files, use row-by-row approach or block read
                if px_off_x != 0 or px_off_y != 0:
                    # With offsets, read row by row
                    full_width = width + px_off_x
                    full_height = height + px_off_y
                    height_map = np.zeros((full_height, full_width), dtype=np.float32)
                    
                    # Read each row and position with offset
                    for y in range(height):
                        row_data = f.read(width * 4)
                        if len(row_data) != width * 4:
                            if debug:
                                print(f"Warning: Row {y} - Expected {width * 4} bytes, got {len(row_data)}")
                            row_data = row_data.ljust(width * 4, b'\0')
                        
                        row_floats = np.frombuffer(row_data, dtype=np.float32)
                        height_map[y + px_off_y, px_off_x:px_off_x + width] = row_floats
                else:
                    # No offset, read as a block
                    height_data = f.read(width * height * 4)
                    
                    if len(height_data) < width * height * 4:
                        if debug:
                            print(f"Warning: Expected {width * height * 4} bytes, got {len(height_data)}")
                        height_data = height_data.ljust(width * height * 4, b'\0')
                    elif len(height_data) > width * height * 4:
                        if debug:
                            print(f"Warning: Extra data detected. Trimming.")
                        height_data = height_data[:width * height * 4]
                    
                    height_map = np.frombuffer(height_data, dtype=np.float32).reshape((height, width))
        except Exception as e:
            if debug:
                print(f"Error parsing height map data: {e}. Creating empty height map.")
            height_map = np.zeros((height, width), dtype=np.float32)

    # Build metadata dictionary
    metadata = {
        "version": version,
        "width": width,
        "height": height,
        "x_length": x_length,
        "y_length": y_length,
        "x_offset": x_offset,
        "y_offset": y_offset,
        "mmpp": mmpp,
        "comment": comment,
        "px_off_x": px_off_x,
        "px_off_y": px_off_y
    }

    return metadata, height_map


def write_tmd_file(
    height_map: np.ndarray,
    output_path: str,
    comment: str = "Created by TrueMap v6",
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
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Get height map dimensions (rows, cols) = (height, width)
    height, width = height_map.shape

    with open(output_path, "wb") as f:
        if debug:
            print(f"Writing TMD file v{version} to {output_path}")
            
        if version == 2:
            # Write the exact header format shown in the example
            header = "Binary TrueMap Data File v2.0\n"
            header_comment = comment if comment else "Created by TrueMap v6\n"
            
            # Ensure header_comment ends with newline
            if not header_comment.endswith('\n'):
                header_comment += '\n'

            # Write header and pad to 32 bytes with nulls if needed
            header_bytes = header.encode('ascii')
            remaining_header = 32 - len(header_bytes)
            if debug:
                print(f"Remaining header: {remaining_header}")
            if remaining_header > 0:
                header_bytes += b'\0' * remaining_header
            f.write(header_bytes[:32])  # Truncate if too long
            
            # Write comment and pad to 24 bytes
            comment_bytes = header_comment.encode('ascii')
            remaining_comment = 24 - len(comment_bytes)
            if debug:
                print(f"Remaining comment: {remaining_comment}")
            if remaining_comment > 0:
                comment_bytes += b'\0' * remaining_comment
            f.write(comment_bytes[:24])
        else:
            # For v1 files, just write a basic header
            header = "Binary TrueMap Data File\r\n"
            header_bytes = header.encode('ascii')
            remaining_header = 28 - len(header_bytes)  # v1 metadata starts at 28
            if remaining_header > 0:
                header_bytes += b'\0' * remaining_header
            f.write(header_bytes[:28])
        
        # Write dimensions: width and height (4 bytes each, little-endian)
        f.write(struct.pack('<II', width, height))
        
        # Write spatial info: x_length, y_length, x_offset, y_offset (each as 4-byte float)
        f.write(struct.pack('<ffff', x_length, y_length, x_offset, y_offset))
        
        # Write the height map data (float32 values)
        height_map_flat = height_map.astype(np.float32).flatten()
        f.write(height_map_flat.tobytes())
    
    if debug:
        print(f"Dimensions: {width} x {height}, Spatial info: {x_length}, {y_length}, {x_offset}, {y_offset}")
        print(f"Successfully wrote TMD file: {output_path}")
        print(f"File size: {os.path.getsize(output_path)} bytes")
    
    return output_path


def create_sample_height_map(
    width: int = 100,
    height: int = 100,
    pattern: str = "waves",
    noise_level: float = 0.05,
) -> np.ndarray:
    """
    Create a sample height map for testing or demonstration purposes.

    Args:
        width: Width of the height map
        height: Height of the height map
        pattern: Type of pattern to generate ("waves", "peak", "dome", "ramp")
        noise_level: Level of random noise to add (0.0 - 1.0+)

    Returns:
        2D numpy array with the generated height map
    """
    # Create coordinate grid
    x = np.linspace(-5, 5, width)
    y = np.linspace(-5, 5, height)
    X, Y = np.meshgrid(x, y)

    # Generate pattern
    if pattern == "waves":
        Z = np.sin(X) * np.cos(Y)
    elif pattern == "peak":
        Z = np.exp(-(X**2 + Y**2) / 8) * 2
    elif pattern == "dome":
        Z = 1.0 - np.sqrt(X**2 + Y**2) / 5
        Z[Z < 0] = 0
    elif pattern == "ramp":
        Z = X + Y
    elif pattern == "combined":
        # Create a combination of patterns
        Z = (
            np.sin(X) * np.cos(Y)  # Wave pattern
            + np.exp(-(X**2 + Y**2) / 8) * 2  # Central peak
        )
    else:
        Z = np.zeros((height, width))

    # Calculate base amplitude to scale the noise appropriately
    base_amplitude = np.max(np.abs(Z)) if np.max(np.abs(Z)) > 0 else 1.0
    
    # Add random noise with consistent application
    if noise_level > 0:
        noise = np.random.normal(0, noise_level * base_amplitude, Z.shape)
        Z = Z + noise

    # Normalize to [0, 1] range
    Z_min = Z.min()
    Z_max = Z.max()
    if Z_max > Z_min:  # Avoid division by zero
        Z = (Z - Z_min) / (Z_max - Z_min)

    return Z.astype(np.float32)


def generate_synthetic_tmd(
    output_path: str = None,
    width: int = 100,
    height: int = 100,
    pattern: str = "combined",
    comment: str = "Created by TrueMap v6",
    version: int = 2,
) -> str:
    """
    Generate a synthetic TMD file for testing or demonstration.

    Args:
        output_path: Path where to save the TMD file (default: "output/synthetic.tmd")
        width: Width of the height map
        height: Height of the height map
        pattern: Type of pattern for the height map
        comment: Comment to include in the file
        version: TMD version to write (1 or 2)

    Returns:
        Path to the created TMD file
    """
    if output_path is None:
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "synthetic.tmd")

    # Create a sample height map with named parameters for test compatibility
    height_map = create_sample_height_map(width=width, height=height, pattern=pattern)

    # Write the height map to a TMD file
    tmd_path = write_tmd_file(
        height_map=height_map,
        output_path=output_path,
        comment=comment,
        x_length=10.0,
        y_length=10.0,
        x_offset=0.0,
        y_offset=0.0,
        version=version,
        debug=True,
    )

    return tmd_path
