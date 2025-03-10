"""
Core utility functions for TMD file processing and analysis.
"""
import os
import struct
import re
import numpy as np
from typing import Dict, Any, Tuple, Optional, List


def hexdump(bytes_data: bytes, start: int = 0, length: Optional[int] = None, 
           width: int = 16, show_ascii: bool = True) -> str:
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
        chunk = bytes_data[start + i:start + i + width]
        hex_part = ' '.join(f'{b:02x}' for b in chunk)
        
        line = f"{start + i:08x}:  {hex_part:<{width * 3}}"
        
        if show_ascii:
            ascii_part = ''.join(chr(b) if 32 <= b < 127 else '.' for b in chunk)
            line += f"  |{ascii_part}|"
        
        result.append(line)
    
    return '\n'.join(result)


def find_null_terminated_strings(data: bytes, min_length: int = 3) -> List[Tuple[int, str]]:
    """
    Find all null-terminated strings in a byte array.
    
    Args:
        data: Bytes to search
        min_length: Minimum length of strings to include
        
    Returns:
        List of tuples with (offset, string)
    """
    result = []
    start = 0
    
    while start < len(data):
        # Find the next printable ASCII character
        while start < len(data) and not (32 <= data[start] <= 126):
            start += 1
            
        if start >= len(data):
            break
            
        # Find the end of this potential string
        end = start
        while end < len(data) and 32 <= data[end] <= 126:
            end += 1
            
        if end - start >= min_length:
            string_bytes = data[start:end]
            try:
                string_value = string_bytes.decode('ascii')
                result.append((start, string_value))
            except UnicodeDecodeError:
                pass
        
        start = end + 1
    
    return result


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
    null_index = chunk.find(b'\0')
    if null_index == -1:
        return chunk.decode('ascii', errors='ignore')
    else:
        file_handle.seek(pos + null_index + 1)
        return chunk[:null_index].decode('ascii', errors='ignore')


def analyze_tmd_file(file_path: str, detail_level: int = 1) -> Dict[str, Any]:
    """
    Analyze a TMD file and return information about its structure.
    
    Args:
        file_path: Path to the TMD file
        detail_level: Level of detail in the analysis (1-3)
        
    Returns:
        Dictionary with analysis results
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Read the file content
    with open(file_path, 'rb') as f:
        content = f.read()
    
    file_size = len(content)
    
    results = {
        "file_path": file_path,
        "file_size": file_size,
        "header_magic": None,
        "possible_formats": [],
    }
    
    # Analyze the start of the file
    if file_size > 32:
        header_bytes = content[:32]
        try:
            header_ascii = header_bytes.decode('ascii', errors='replace')
            results["header_ascii"] = header_ascii.replace('\x00', '\\0')
            
            # Check for known header patterns
            if "Binary TrueMap Data File" in header_ascii:
                results["possible_formats"].append("TrueMap")
            if "GelSight" in header_ascii:
                results["possible_formats"].append("GelSight")
        except UnicodeDecodeError:
            results["header_ascii"] = "Unable to decode as ASCII"
    
    # Show detailed hex dump
    if detail_level >= 2:
        results["hex_dump_header"] = hexdump(content, 0, min(128, file_size))
    
    # Try to find dimensions
    if file_size > 40:
        # Try different offsets where dimensions might be stored
        candidate_offsets = [32, 36, 40, 64]  # Common offsets
        
        for offset in candidate_offsets:
            if offset + 8 > file_size:
                continue
                
            # Try little-endian and big-endian
            for endian in ['<', '>']:
                try:
                    dim_bytes = content[offset:offset+8]
                    width, height = struct.unpack(f'{endian}II', dim_bytes)
                    
                    # Only consider reasonable dimensions
                    if 1 <= width <= 10000 and 1 <= height <= 10000:
                        if "dimension_candidates" not in results:
                            results["dimension_candidates"] = []
                            
                        results["dimension_candidates"].append({
                            "offset": offset,
                            "endian": "little" if endian == '<' else "big",
                            "width": width,
                            "height": height,
                            "product": width * height,
                            "hex": dim_bytes.hex()
                        })
                except Exception:
                    pass
    
    return results


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
        
    with open(file_path, 'rb') as f:
        header_bytes = f.read(64)
        header_text = header_bytes.decode('ascii', errors='replace')
        
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


def try_read_dimensions(f, endian='<', offset=None):
    """
    Try to read dimensions from the current file position or specified offset.
    
    Args:
        f: Open file handle
        endian: Endianness to use ('<' for little-endian, '>' for big-endian)
        offset: Byte offset to read from (None = current position)
        
    Returns:
        Tuple of (width, height) or None if dimensions couldn't be read
    """
    if offset is not None:
        original_pos = f.tell()
        f.seek(offset)
    
    try:
        width, height = struct.unpack(f'{endian}II', f.read(8))
        if width > 0 and height > 0 and width * height <= 1e8:  # Reasonable limit
            return width, height
    except Exception:
        pass
    
    if offset is not None:
        f.seek(original_pos)
    
    return None


def process_tmd_file(file_path: str, 
                   force_offset: Optional[Tuple[float, float]] = None,
                   debug: bool = False) -> Tuple[Dict[str, Any], np.ndarray]:
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
    
    # Initialize metadata
    metadata = {"source_file": file_path}
    
    file_size = os.path.getsize(file_path)
    if file_size < 60:  # Minimum size for valid TMD
        raise ValueError(f"File too small to be a valid TMD: {file_size} bytes")
    
    with open(file_path, 'rb') as f:
        # Read and analyze header
        header_bytes = f.read(32)
        try:
            header = header_bytes.decode('ascii', errors='replace')
            metadata["header"] = header.strip()
        except:
            metadata["header"] = "Unable to decode header"
        
        # Check for version indicators
        version_match = re.search(r'v(\d+\.\d+)', metadata["header"])
        version = float(version_match.group(1)) if version_match else 2.0
        metadata["version"] = version
        
        # Handle comment field for v2 files
        comment = ""
        if version >= 2.0:
            f.seek(32)  # Position after header
            comment = read_null_terminated_string(f)
            metadata["comment"] = comment
        
        # Try to find dimensions at different offsets
        dimensions_found = False
        found_offset = None
        endian_used = '<'  # Default to little-endian
        
        # List of offsets to try for dimensions
        offsets_to_try = [f.tell(), 32, 40, 52, 64]
        
        # Try each offset with both endians
        for offset in offsets_to_try:
            if offset + 24 >= file_size:  # Need space for dimensions + metadata
                continue
                
            for endian in ['<', '>']:
                result = try_read_dimensions(f, endian, offset)
                if result:
                    width, height = result
                    
                    # If dimensions seem valid, try to read spatial info
                    f.seek(offset + 8)  # Skip dimensions
                    try:
                        x_length, y_length, x_offset, y_offset = struct.unpack(f'{endian}ffff', f.read(16))
                        
                        if debug:
                            print(f"Found valid dimensions at offset {offset} with {endian}-endian")
                            print(f"Dimensions: {width}x{height}, Spatial: {x_length},{y_length},{x_offset},{y_offset}")
                        
                        # If values seem reasonable
                        dimensions_found = True
                        found_offset = offset
                        endian_used = endian
                        
                        # Update metadata
                        metadata.update({
                            "width": width,
                            "height": height,
                            "x_length": x_length,
                            "y_length": y_length,
                            "x_offset": x_offset if force_offset is None else force_offset[0],
                            "y_offset": y_offset if force_offset is None else force_offset[1],
                            "_parse_info": {
                                "data_offset": offset + 24,  # After dimensions and spatial info
                                "endian": endian
                            }
                        })
                        break
                    except Exception as e:
                        if debug:
                            print(f"Error reading spatial info at offset {offset}: {e}")
                        
            if dimensions_found:
                break
        
        if not dimensions_found:
            raise ValueError("Could not find valid dimensions in the TMD file")
        
        # Read height map data
        width = metadata["width"]
        height = metadata["height"]
        data_offset = metadata["_parse_info"]["data_offset"]
        
        f.seek(data_offset)
        num_points = width * height
        expected_bytes = num_points * 4  # 4 bytes per float32
        
        # Check if we have enough data
        remaining_bytes = file_size - f.tell()
        if remaining_bytes < expected_bytes:
            raise ValueError(f"Not enough data for height map. Expected {expected_bytes} bytes, but only {remaining_bytes} remain")
        
        # Read height map data
        height_bytes = f.read(expected_bytes)
        height_data = np.frombuffer(height_bytes, dtype=np.float32 if endian_used == '<' else '>f4')
        
        # Reshape into 2D array
        height_map = height_data.reshape((height, width))
    
    return metadata, height_map


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
