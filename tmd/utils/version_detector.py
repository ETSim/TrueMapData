"""
Utilities for detecting and handling different versions of TMD file formats.
"""
import os
import re
import struct
from typing import Tuple, Optional, List

from .utils import read_null_terminated_string

# TMD format constants
TMD_V1_OFFSET = 28
TMD_MAX_DIMENSION = 10000


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
        header_bytes = f.read(32)
        header_text = header_bytes.decode('ascii', errors='replace')
        
        # Check for explicit version number in header
        version_match = re.search(r'v(\d+)\.(\d+)', header_text)
        if version_match:
            major_version = int(version_match.group(1))
            return major_version
            
        # No explicit version - check for file structure indicators
        if "TrueMap Data File" in header_text:
            # Check for v1 specific structure
            if len(header_bytes) >= TMD_V1_OFFSET and header_bytes[TMD_V1_OFFSET-1:TMD_V1_OFFSET] == b'\0':
                return 1
    
    # Default to v2 if unable to determine
    return 2


def get_header_offset(file_path: str) -> int:
    """
    Determine the offset where the dimension data begins based on file version.
    
    Args:
        file_path: Path to the TMD file
        
    Returns:
        Byte offset where dimension data begins
    """
    version = detect_tmd_version(file_path)
    
    if version == 1:
        # V1 files have dimensions at fixed position
        return TMD_V1_OFFSET
    else:
        # V2 files have variable-length comment field
        with open(file_path, 'rb') as f:
            f.seek(32)  # Skip fixed header
            
            # Find the null terminator for the comment field
            pos = 32
            while True:
                b = f.read(1)
                pos += 1
                if not b or b == b'\0':
                    break
            
            return pos  # Position after the null-terminated comment


def extract_dimensions(file_path: str, offset: Optional[int] = None) -> Tuple[int, int]:
    """
    Extract width and height from a TMD file.
    
    Args:
        file_path: Path to the TMD file
        offset: Byte offset where dimensions data begins, or None to auto-detect
        
    Returns:
        Tuple of (width, height)
    """
    if offset is None:
        offset = get_header_offset(file_path)
    
    with open(file_path, 'rb') as f:
        f.seek(offset)
        
        # Try to read dimensions with appropriate endianness
        try:
            # Most likely little-endian
            width = struct.unpack('<I', f.read(4))[0]
            height = struct.unpack('<I', f.read(4))[0]
            
            # Quick sanity check - if values are extreme, try big-endian
            if width > TMD_MAX_DIMENSION or height > TMD_MAX_DIMENSION:
                f.seek(offset)  # Go back to start of dimensions
                width = struct.unpack('>I', f.read(4))[0]
                height = struct.unpack('>I', f.read(4))[0]
        except Exception as e:
            raise ValueError(f"Error reading dimensions: {e}")
        
    # Sanity check for reasonable dimensions (prevent errors from misreading)
    if width > TMD_MAX_DIMENSION or height > TMD_MAX_DIMENSION or width == 0 or height == 0:
        raise ValueError(f"Unreasonable dimensions detected: {width}x{height}. Possible file format mismatch.")
        
    return width, height


def find_dimension_candidates(file_content: bytes, max_offset: int = 256) -> List[Tuple[int, str, int, int]]:
    """
    Find potential dimension fields in the file by scanning for reasonable-sized values.
    
    Args:
        file_content: Binary content of the TMD file
        max_offset: Maximum byte offset to search
        
    Returns:
        List of tuples: (offset, endianness, width, height)
    """
    candidates = []
    search_limit = min(len(file_content) - 8, max_offset)
    
    for offset in range(0, search_limit, 4):
        # Try both endian formats
        for endian in ['<', '>']:
            try:
                width, height = struct.unpack(f'{endian}II', file_content[offset:offset+8])
                
                # Keep only if dimensions seem reasonable
                if 1 <= width <= 10000 and 1 <= height <= 10000:
                    candidates.append((offset, endian, width, height))
            except Exception:
                pass
    
    return candidates
