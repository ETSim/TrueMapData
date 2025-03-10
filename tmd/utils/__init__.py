"""
TMD Utilities module.
"""

# Import and re-export core utilities
from .utils import (
    hexdump,
    find_null_terminated_strings,
    read_null_terminated_string,
    analyze_tmd_file,
    detect_tmd_version,
    try_read_dimensions,
    process_tmd_file,
    get_header_offset  # This was missing but called in processor.py
)

__all__ = [
    'hexdump',
    'find_null_terminated_strings',
    'read_null_terminated_string',
    'analyze_tmd_file',
    'detect_tmd_version',
    'try_read_dimensions',
    'process_tmd_file',
    'get_header_offset'
]
