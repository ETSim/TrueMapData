"""
Core utility functions for TMD file processing and analysis.
"""

import os
import struct
import re
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
import logging

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
    data_to_process = bytes_data[start:start + length]
    
    result = []
    for i in range(0, len(data_to_process), width):
        chunk = data_to_process[i:i + width]
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

        # Read dimensions
        width_bytes = f.read(4)
        height_bytes = f.read(4)
        
        if len(width_bytes) < 4 or len(height_bytes) < 4:
            if debug:
                print("Warning: File too small to read dimensions properly.")
            width = 1
            height = 1
        else:
            try:
                # In TMD format, dimensions are stored as (width, height)
                width = struct.unpack("<I", width_bytes)[0]
                height = struct.unpack("<I", height_bytes)[0]
                print(f"Width: {width}, Height: {height}")
            except Exception as e:
                if debug:
                    print(f"Error parsing dimensions: {str(e)}")
                width = 1
                height = 1
        
        # Validate dimensions to avoid memory issues
        if width <= 0 or height <= 0 or width > 10000 or height > 10000:
            raise ValueError(f"Invalid dimensions: {width} x {height}")
        
        # Read spatial parameters
        x_length_bytes = f.read(4)
        y_length_bytes = f.read(4)
        x_offset_bytes = f.read(4)
        y_offset_bytes = f.read(4)
        
        # Default values if reading fails
        x_length = 10.0
        y_length = 10.0
        x_offset = 0.0
        y_offset = 0.0
        
        # Try to parse values
        if len(x_length_bytes) == 4:
            try:
                x_length = struct.unpack("<f", x_length_bytes)[0]
            except Exception:
                pass
        
        if len(y_length_bytes) == 4:
            try:
                y_length = struct.unpack("<f", y_length_bytes)[0]
            except Exception:
                pass
        
        if len(x_offset_bytes) == 4:
            try:
                x_offset = struct.unpack("<f", x_offset_bytes)[0]
            except Exception:
                pass
        
        if len(y_offset_bytes) == 4:
            try:
                y_offset = struct.unpack("<f", y_offset_bytes)[0]
            except Exception:
                pass

        if debug:
            print(f"Width: {width}, Height: {height}")
            print(f"X Length: {x_length}, Y Length: {y_length}")
            print(f"X Offset: {x_offset}, Y Offset: {y_offset}")

        # Apply forced offsets if provided
        if force_offset:
            x_offset, y_offset = force_offset
            if debug:
                print(f"⚠️ Using forced offsets: x_offset={x_offset}, y_offset={y_offset}")

        # Read height map data with a safety check
        try:
            remaining_bytes = f.read()
            max_safe_size = width * height * 4 * 2  # Allow for some extra data
            
            if len(remaining_bytes) > max_safe_size:
                if debug:
                    print(f"Warning: File contains much more data than expected. Limiting read size.")
                remaining_bytes = remaining_bytes[:max_safe_size]
                
            height_map_data = np.frombuffer(remaining_bytes, dtype=np.float32)
            expected_size = width * height

            # Handle potential size mismatches
            if len(height_map_data) < expected_size:
                if debug:
                    print(
                        f"Warning: Incomplete height map. Expected {expected_size}, got {len(height_map_data)}. Padding with zeros."
                    )
                # Pad the array safely
                padding_size = expected_size - len(height_map_data)
                padding = np.zeros(padding_size, dtype=np.float32)
                height_map_data = np.concatenate([height_map_data, padding])
            elif len(height_map_data) > expected_size:
                if debug:
                    print(
                        f"Warning: Extra data detected. Expected {expected_size}, got {len(height_map_data)}. Trimming excess."
                    )
                height_map_data = height_map_data[:expected_size]
                
            # Reshape into 2D array - TMD stores data in row-major order but with width first
            # So need to reshape to (height, width) for NumPy
            height_map = height_map_data.reshape((height, width))
        except Exception as e:
            if debug:
                print(f"Error parsing height map data: {str(e)}. Creating empty map.")
            # Create minimal empty height map
            height_map = np.zeros((height, width), dtype=np.float32)

    # Build metadata dictionary
    metadata = {
        "version": version,
        "width": width,  # width is columns (x-axis)
        "height": height,  # height is rows (y-axis)
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
        print(f"Writing TMD file v{version} to {output_path}")
        if version == 2:
            # Write the exact header format shown in the example
            header = "Binary TrueMap Data File v2.0\n"
            header_comment = "Created by TrueMap v6\n"
            
            # Write header and pad to 32 bytes with nulls if needed
            header_bytes = header.encode('ascii')
            remaining_header = 32 - len(header_bytes)
            logger.info(f"Remaining header: {remaining_header}")
            print(f"Remaining header: {remaining_header}")
            if remaining_header > 0:
                header_bytes += b'\0' * remaining_header
            f.write(header_bytes[:32])  # Truncate if too long
            
            # Write comment and pad to 24 bytes
            comment_bytes = header_comment.encode('ascii')
            remaining_comment = 24 - len(comment_bytes)
            logger.info(f"Remaining comment: {remaining_comment}")
            print(f"Remaining comment: {remaining_comment}")
            if remaining_comment > 0:
                comment_bytes += b'\0' * remaining_comment
            f.write(comment_bytes[:24])
        
        # Write dimensions: width and height (4 bytes each, little-endian)
        f.write(struct.pack('<II', width, height))
        
        # Write spatial info: x_length, y_length, x_offset, y_offset (each as 4-byte float)
        f.write(struct.pack('<ffff', x_length, y_length, x_offset, y_offset))
        
        # Write the height map data (float32 values)
        height_map_flat = height_map.astype(np.float32).flatten()
        f.write(height_map_flat.tobytes())
    
    if debug:
        print(f"Writing TMD file v{version} to {output_path}")
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
