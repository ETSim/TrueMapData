#!/usr/bin/env python3
"""
ZIP Exporter/Importer for TMD Data.

This module provides concrete implementations for exporting and importing
TMD data in the ZIP format with optional compression levels.
"""

import os
import logging
import zipfile
from pathlib import Path
import json
import numpy as np
from typing import Any, Dict, Optional, List, Union, Tuple

from .base import TMDDataExporter, TMDDataImporter

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _export_zip(data: Dict[str, Any], output_path: str, compression_level: int = 9, 
               optimize_arrays: bool = True, chunk_threshold_mb: float = 100,
               metadata_format: str = "json") -> str:
    """
    Export data to a ZIP file with metadata and arrays as separate entries.
    
    Args:
        data: Dictionary containing data to export
        output_path: Path to save the ZIP file
        compression_level: Compression level (0-9, where 9 is highest compression)
        optimize_arrays: Whether to optimize large arrays for better compression
        chunk_threshold_mb: Size threshold (in MB) for array chunking
        metadata_format: Format for metadata file ("json" or "txt")
        
    Returns:
        Path to the created ZIP file
    """
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        # Set compression method based on level
        if compression_level == 0:
            compression = zipfile.ZIP_STORED
        else:
            compression = zipfile.ZIP_DEFLATED
            
        # Create a structured data dictionary: separate arrays from metadata
        meta_data = {}
        array_entries = []
        chunked_arrays = {}
        
        # Process data entries
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                # Process arrays separately
                array_entries.append(key)
                
                # Check if array should be chunked (large arrays)
                array_size_mb = value.nbytes / (1024 * 1024)
                if optimize_arrays and array_size_mb > chunk_threshold_mb:
                    # Store info about chunking in metadata
                    chunks = _calculate_optimal_chunks(value.shape)
                    chunked_arrays[key] = {
                        "original_shape": value.shape,
                        "chunks": chunks,
                        "dtype": str(value.dtype)
                    }
                    logger.debug(f"Array '{key}' ({array_size_mb:.1f} MB) will be chunked into {len(chunks)} parts")
                    
            else:
                # Add to metadata
                meta_data[key] = value
                
        # Add array info to metadata
        meta_data["_array_entries"] = array_entries
        meta_data["_chunked_arrays"] = chunked_arrays
        
        # Create the ZIP file
        with zipfile.ZipFile(output_path, 'w', compression=compression) as zf:
            # Save metadata based on format preference
            if metadata_format.lower() == "json":
                zf.writestr('metadata.json', json.dumps(meta_data, default=str, indent=2))
            else:
                # Create plain text representation of metadata
                txt_content = ["TMD Data Export Metadata", "=======================", ""]
                for key, value in meta_data.items():
                    if key.startswith('_'):  # Skip internal keys
                        continue
                    txt_content.append(f"{key}: {value}")
                
                txt_content.append("\nArray Information:")
                for key in array_entries:
                    if key in chunked_arrays:
                        chunks_info = chunked_arrays[key]
                        txt_content.append(f"  {key}: {chunks_info['original_shape']} ({chunks_info['dtype']}, chunked)")
                    elif key in data:
                        shape = data[key].shape
                        dtype = data[key].dtype
                        txt_content.append(f"  {key}: {shape} ({dtype})")
                
                zf.writestr('metadata.txt', '\n'.join(txt_content))
            
            # Save each array
            for key in array_entries:
                array_data = data[key]
                
                # Check if this array should be chunked
                if key in chunked_arrays:
                    chunks = chunked_arrays[key]["chunks"]
                    # Save array in chunks
                    for i, (start_indices, end_indices) in enumerate(chunks):
                        # Create slices for each dimension
                        slices = tuple(slice(start, end) for start, end in zip(start_indices, end_indices))
                        chunk_data = array_data[slices]
                        
                        # Save chunk
                        chunk_name = f"{key}_chunk_{i}.npy"
                        with zf.open(chunk_name, 'w') as f:
                            np.lib.format.write_array(f, chunk_data)
                else:
                    # Save as single array
                    with zf.open(f"{key}.npy", 'w') as f:
                        np.lib.format.write_array(f, array_data)
            
        logger.info(f"Data exported to ZIP file: {output_path} (compression level: {compression_level})")
        return output_path
    
    except Exception as e:
        logger.error(f"Error exporting ZIP: {e}")
        raise


def _calculate_optimal_chunks(shape: Tuple[int, ...]) -> List[Tuple[List[int], List[int]]]:
    """
    Calculate optimal chunk sizes for a large array.
    
    Args:
        shape: Shape of the array to chunk
        
    Returns:
        List of (start_indices, end_indices) pairs for each chunk
    """
    # For 2D arrays (like height maps), divide into grid of chunks
    if len(shape) == 2:
        height, width = shape
        # Target around 25-50MB per chunk
        rows_per_chunk = max(1, min(height, 1000))
        cols_per_chunk = max(1, min(width, 1000))
        
        chunks = []
        for row_start in range(0, height, rows_per_chunk):
            row_end = min(row_start + rows_per_chunk, height)
            for col_start in range(0, width, cols_per_chunk):
                col_end = min(col_start + cols_per_chunk, width)
                chunks.append(([row_start, col_start], [row_end, col_end]))
        return chunks
    
    # For other dimensionality, use simpler chunking along first dimension
    else:
        dim0 = shape[0]
        items_per_chunk = max(1, min(dim0, 1000))
        
        chunks = []
        for i in range(0, dim0, items_per_chunk):
            start_idx = [i] + [0] * (len(shape) - 1)
            end_idx = [min(i + items_per_chunk, dim0)] + list(shape[1:])
            chunks.append((start_idx, end_idx))
        return chunks


def _load_zip(file_path: str) -> Dict[str, Any]:
    """
    Load data from a ZIP file.
    
    Args:
        file_path: Path to the ZIP file
        
    Returns:
        Dictionary with loaded data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"ZIP file not found: {file_path}")
        
    try:
        data = {}
        
        with zipfile.ZipFile(file_path, 'r') as zf:
            # Load metadata first
            with zf.open('metadata.json') as f:
                metadata = json.loads(f.read().decode('utf-8'))
                
            # Extract array entries and chunking info
            array_entries = metadata.pop("_array_entries", [])
            chunked_arrays = metadata.pop("_chunked_arrays", {})
            
            # Add metadata to result
            data.update(metadata)
            
            # Load each array
            for key in array_entries:
                # Check if this array was chunked
                if key in chunked_arrays:
                    # Get original array info
                    original_shape = tuple(chunked_arrays[key]["original_shape"])
                    dtype_str = chunked_arrays[key]["dtype"]
                    dtype = np.dtype(dtype_str)
                    
                    # Create empty array to fill with chunks
                    array_data = np.zeros(original_shape, dtype=dtype)
                    
                    # Find all chunks for this array
                    chunk_files = [name for name in zf.namelist() 
                                 if name.startswith(f"{key}_chunk_") and name.endswith(".npy")]
                    
                    # Sort by chunk number
                    chunk_files.sort(key=lambda x: int(x.split("_chunk_")[1].split(".")[0]))
                    
                    # Load each chunk
                    for chunk_idx, chunk_file in enumerate(chunk_files):
                        with zf.open(chunk_file) as f:
                            chunk_data = np.lib.format.read_array(f)
                            
                        # Get chunk indices
                        start_indices, end_indices = chunked_arrays[key]["chunks"][chunk_idx]
                        
                        # Create slices for each dimension
                        slices = tuple(slice(start, end) for start, end in zip(start_indices, end_indices))
                        
                        # Place chunk in the appropriate location
                        array_data[slices] = chunk_data
                    
                    data[key] = array_data
                    
                else:
                    # Load as single array
                    with zf.open(f"{key}.npy") as f:
                        array_data = np.lib.format.read_array(f)
                        data[key] = array_data
                    
        logger.info(f"Data loaded from ZIP file: {file_path}")
        return data
        
    except Exception as e:
        logger.error(f"Error loading ZIP file: {e}")
        raise


class ZIPExporter(TMDDataExporter):
    """ZIP format exporter for TMD data with optimization options."""
    
    def __init__(self, compression_level: int = 9, optimize: bool = True, 
                chunk_threshold_mb: float = 50.0, metadata_format: str = "json"):
        """
        Initialize the ZIP exporter with specific options.
        
        Args:
            compression_level: Compression level (0-9, where 0 is no compression, 9 is highest)
            optimize: Whether to apply optimizations for large arrays
            chunk_threshold_mb: Size threshold (in MB) for array chunking
            metadata_format: Format for metadata file ("json" or "txt")
        """
        self.compression_level = max(0, min(9, compression_level))  # Clamp to valid range
        self.optimize = optimize
        self.chunk_threshold_mb = chunk_threshold_mb
        self.metadata_format = metadata_format
    
    def export(self, data: Dict[str, Any], output_path: str) -> str:
        """
        Export data to ZIP format.
        
        Args:
            data: Dictionary containing data to export
            output_path: Destination file path
            
        Returns:
            Path to the exported file
        """
        return _export_zip(
            data, 
            output_path, 
            compression_level=self.compression_level,
            optimize_arrays=self.optimize,
            chunk_threshold_mb=self.chunk_threshold_mb,
            metadata_format=self.metadata_format
        )


class ZIPImporter(TMDDataImporter):
    """ZIP format importer for TMD data."""
    
    def __init__(self):
        """
        Initialize the ZIP importer.
        """
        pass
    
    def load(self, file_path: str) -> Dict[str, Any]:
        """
        Load data from ZIP file.
        
        Args:
            file_path: Path to the ZIP file
            
        Returns:
            Dictionary containing the loaded data
            
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        return _load_zip(file_path)
