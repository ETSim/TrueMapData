""".

Compression exporters for TMD data.

This module provides functions to export TMD data in compressed formats.
"""

from .npy import export_to_npy, load_from_npy
from .npz import export_to_npz, load_from_npz

__all__ = [
    'export_to_npy',
    'load_from_npy',
    'export_to_npz',
    'load_from_npz'
]
