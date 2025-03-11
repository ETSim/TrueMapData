"""
Core utility functions for TMD file processing and analysis.
"""

import os
import struct
import re
import numpy as np
from typing import Dict, Any, Tuple, Optional, List


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
        length = len(bytes_data)

    result = []
    for i in range(0, min(length, len(bytes_data) - start), width):
        chunk = bytes_data[start + i : start + i + width]
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

        # Check for version number in header
        if "v2.0" in header_text:
            return 2
        elif "v1.0" in header_text:
            return 1

        # If no explicit version, try to determine from structure
        if "Binary TrueMap Data File" in header_text:
            # Most likely v2 if it has the standard header
            return 2

    # Default to v1 if unable to determine
    return 1


def get_header_offset(version):
    """
    Get the header offset for the given TMD version.

    Args:
        version: TMD file version (1 or 2)

    Returns:
        Header offset in bytes
    """
    if version == 1:
        return 32  # Version 1 files have header at offset 32
    else:
        return 64  # Version 2 files have header at offset 64


def process_tmd_file(
    file_path: str,
    force_offset: Optional[Tuple[float, float]] = None,
    debug: bool = False,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Process a TMD file and extract metadata and height map.
    Handles both v1 and v2 file formats.

    Args:
        file_path: Path to the TMD file
        force_offset: Optional tuple (x_offset, y_offset) to override file values
        debug: Whether to print debug information

    Returns:
        Tuple of (metadata_dict, height_map_array)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    version = detect_tmd_version(file_path)

    with open(file_path, "rb") as f:
        # Handle different versions
        if version == 1:
            f.seek(28)  # Offset for v1 files
            if debug:
                print("⚠️ Detected v1 file format. Reading metadata...")
        elif version == 2:
            f.seek(32)  # Default for v2 files
            if debug:
                print("⚠️ Detected v2 file format. Reading metadata...")
            try:
                comment = read_null_terminated_string(f).strip()
                if debug:
                    print(f"Comment: {comment}")
            except Exception:
                comment = None
                f.seek(33)  # Reset to read the rest of the data

        # Read dimensions and spatial info
        width = struct.unpack("<I", f.read(4))[0]
        height = struct.unpack("<I", f.read(4))[0]
        x_length = struct.unpack("<f", f.read(4))[0]
        y_length = struct.unpack("<f", f.read(4))[0]
        x_offset = struct.unpack("<f", f.read(4))[0]
        y_offset = struct.unpack("<f", f.read(4))[0]

        if debug:
            print(f"Width: {width}, Height: {height}")
            print(f"X Length: {x_length}, Y Length: {y_length}")
            print(f"X Offset: {x_offset}, Y Offset: {y_offset}")

        # Apply forced offsets if provided
        if force_offset:
            x_offset, y_offset = force_offset
            if debug:
                print(
                    f"⚠️ Using forced offsets: x_offset={x_offset}, y_offset={y_offset}"
                )

        # Read height map data
        height_map_data = np.frombuffer(f.read(), dtype=np.float32)
        expected_size = width * height

        # Handle potential size mismatches
        if height_map_data.size < expected_size:
            if debug:
                print(
                    f"Warning: Incomplete height map. Expected {expected_size}, got {height_map_data.size}."
                )
            height_map = np.pad(
                height_map_data,
                (0, expected_size - height_map_data.size),
                mode="constant",
            )
        elif height_map_data.size > expected_size:
            if debug:
                print(
                    f"Warning: Extra data detected. Expected {expected_size}, got {height_map_data.size}. Trimming excess."
                )
            height_map = height_map_data[:expected_size]
        else:
            height_map = height_map_data

        # Reshape to 2D array
        height_map = height_map.reshape((height, width))

    # Build metadata dictionary
    metadata = {
        "version": version,
        "width": width,
        "height": height,
        "x_length": x_length,
        "y_length": y_length,
        "x_offset": x_offset,
        "y_offset": y_offset,
        "comment": comment if version == 2 else None,
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
    Write a height map to a TMD file with v2 format.

    File structure:
    - Fixed header (32 bytes): "Binary TrueMap Data File v2.0" or similar
    - Comment (null-terminated string, padded to 32 bytes)
    - Dimensions: width and height (4 bytes each, little-endian)
    - Spatial info: x_length, y_length, x_offset, y_offset (4 bytes each, float32)
    - Height map data (float32 values)

    Args:
        height_map: 2D numpy array of height values
        output_path: Path where to save the TMD file
        comment: Optional comment string
        x_length: Physical length in X direction (mm)
        y_length: Physical length in Y direction (mm)
        x_offset: Offset in X direction (mm)
        y_offset: Offset in Y direction (mm)
        version: TMD version to write (1 or 2)
        debug: Whether to print debug information

    Returns:
        Path to the created TMD file
    """
    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get height map dimensions
    height, width = height_map.shape

    if debug:
        print(f"Writing TMD file v{version} to {output_path}")
        print(
            f"Dimensions: {width}x{height}, Spatial info: {x_length},{y_length},{x_offset},{y_offset}"
        )

    with open(output_path, "wb") as f:
        # Write v2 header (32 bytes)
        header_str = "Binary TrueMap Data File v2.0"
        header_bytes = header_str.encode("ascii") + b"\0"
        f.write(header_bytes)

        # Write comment field (null-terminated, padded to reach offset 64)
        comment_str = "Created by TrueMap v6"
        comment_bytes = comment_str.encode("ascii") + b"\0"
        f.write(comment_bytes)

        # Write dimensions: width and height (4 bytes each, little-endian)
        f.write(struct.pack("<II", width, height))

        # Write spatial info (each as 4-byte float)
        f.write(struct.pack("<ffff", x_length, y_length, x_offset, y_offset))

        # Write height map data as float32 values
        f.write(height_map.astype(np.float32).tobytes())

    if debug:
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
        noise_level: Level of random noise to add (0.0 - 1.0)

    Returns:
        2D numpy array with the generated height map
    """
    # Create coordinate grid
    x = np.linspace(-5, 5, width)
    y = np.linspace(-5, 5, height)
    X, Y = np.meshgrid(x, y)

    # Generate base pattern
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

    # Add random noise
    if noise_level > 0:
        Z += np.random.normal(0, noise_level, Z.shape)

    # Normalize to [0, 1] range
    Z = (Z - Z.min()) / (Z.max() - Z.min())

    return Z.astype(np.float32)


def generate_synthetic_tmd(
    output_path: str = None,
    width: int = 100,
    height: int = 100,
    pattern: str = "combined",
    comment: str = "Synthetic TMD File",
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

    # Create a sample height map
    height_map = create_sample_height_map(width, height, pattern)

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
