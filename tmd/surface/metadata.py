"""
Metadata utilities for TMD files.
"""

import os
import json
import numpy as np
from typing import Dict, Any, Optional, List, Tuple


def compute_stats(height_map: np.ndarray) -> Dict[str, Any]:
    """
    Calculate statistics for a given height map.
    
    Args:
        height_map: 2D numpy array of height values.
    
    Returns:
        Dictionary containing statistics (min, max, mean, median, std, shape, etc.).
    """
    # Handle NaN values properly
    valid_data = ~np.isnan(height_map)
    valid_values = height_map[valid_data]
    
    # If no valid values, return zeros
    if len(valid_values) == 0:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "shape": height_map.shape,
            "non_nan": 0,
            "nan_count": int(np.prod(height_map.shape)),
        }
    
    stats = {
        "min": float(np.nanmin(height_map)),
        "max": float(np.nanmax(height_map)),
        "mean": float(np.nanmean(height_map)),
        "median": float(np.nanmedian(height_map)),
        "std": float(np.nanstd(height_map)),
        "shape": height_map.shape,
        "non_nan": int(np.count_nonzero(valid_data)),
        "nan_count": int(np.count_nonzero(np.isnan(height_map))),
    }
    return stats


def export_metadata(
    metadata: Dict[str, Any], stats: Dict[str, Any], output_path: str
) -> str:
    """
    Export metadata and height map statistics to a text file.
    
    Args:
        metadata: Dictionary containing metadata (excluding the height map).
        stats: Dictionary containing computed statistics.
        output_path: File path to save the metadata.
    
    Returns:
        The output path to the saved metadata file.
    """
    with open(output_path, "w") as f:
        f.write(f"TMD File: {metadata.get('file_path', 'N/A')}\n")
        f.write("=" * 80 + "\n\n")
        for key, value in metadata.items():
            if key != "file_path":
                f.write(f"{key}: {value}\n")
        f.write("\nHeight Map Statistics\n")
        f.write("-" * 20 + "\n")
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    return output_path


def export_metadata_txt(data_dict, filename="tmd_metadata.txt"):
    """
    Exports TMD metadata to a human-readable text file.
    
    Args:
        data_dict: Dictionary containing TMD data
        filename: Name of the output text file
    
    Returns:
        Path to the saved file
    """
    with open(filename, "w") as f:
        f.write("TMD File Metadata\n")
        f.write("================\n\n")

        # Write metadata values
        for key, value in data_dict.items():
            if key != "height_map":  # Skip the height map
                f.write(f"{key}: {value}\n")

        # Write height map statistics
        if "height_map" in data_dict:
            height_map = data_dict["height_map"]
            f.write("\nHeight Map Statistics\n")
            f.write("====================\n")
            f.write(f"Shape: {height_map.shape}\n")
            f.write(f"Min: {height_map.min()}\n")
            f.write(f"Max: {height_map.max()}\n")
            f.write(f"Mean: {height_map.mean()}\n")
            f.write(f"Std Dev: {height_map.std()}\n")

    print(f"TMD metadata saved to text file: {filename}")
    return filename


def extract_metadata(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract metadata from a TMD data dictionary.
    
    Args:
        data: Dictionary containing TMD data
        
    Returns:
        Dictionary with metadata
    """
    metadata = {}
    
    # Copy basic metadata
    if 'metadata' in data:
        metadata = data['metadata'].copy()
    
    # Extract size info
    if 'height_map' in data:
        height_map = data['height_map']
        metadata['dimensions'] = {
            'height': height_map.shape[0],
            'width': height_map.shape[1],
            'size_bytes': height_map.nbytes
        }
        
        # Calculate height stats
        if np.size(height_map) > 0:
            metadata['height_stats'] = {
                'min': float(np.min(height_map)),
                'max': float(np.max(height_map)),
                'mean': float(np.mean(height_map)),
                'std': float(np.std(height_map))
            }
    
    # Extract file info if available
    if 'file_info' in data:
        metadata['file_info'] = data['file_info']
    
    return metadata


def extract_metadata_from_tmd_file(filepath: str) -> Optional[Dict[str, Any]]:
    """
    Extract metadata from a TMD file without loading the full height map.
    
    Args:
        filepath: Path to the TMD file
        
    Returns:
        Dictionary with metadata or None if file can't be processed
    """
    # Import here to avoid circular imports
    from tests.core.tmd import TMDProcessor
    
    # Create processor instance
    processor = TMDProcessor(filepath)
    
    # Extract file info and metadata
    metadata = processor.extract_metadata()
    
    return metadata


def save_metadata_to_json(metadata: Dict[str, Any], output_path: str) -> str:
    """
    Save metadata to a JSON file.
    
    Args:
        metadata: Metadata dictionary
        output_path: Path to save the JSON file
        
    Returns:
        Path to the saved file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Convert numpy types to native Python types
    def convert_numpy_types(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.number):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(i) for i in obj]
        else:
            return obj
    
    # Convert numpy types
    metadata_converted = convert_numpy_types(metadata)
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_converted, f, indent=2)
    
    return output_path


def analyze_surface_roughness(
    height_map: np.ndarray,
    region: Optional[Tuple[int, int, int, int]] = None
) -> Dict[str, float]:
    """
    Analyze surface roughness metrics from a height map.
    
    Calculates standard roughness parameters used in surface metrology:
    - Rq: Root Mean Square Roughness
    - Ra: Arithmetic Average Roughness
    - Rp: Maximum Peak Height (from mean plane)
    - Rv: Maximum Valley Depth (from mean plane)
    - Rt: Maximum Height (peak-to-valley height)
    
    Args:
        height_map: 2D numpy array of height values
        region: Optional region to analyze as (row_start, row_end, col_start, col_end)
        
    Returns:
        Dictionary of roughness metrics with float values
    """
    # Extract region if specified
    if region:
        row_start, row_end, col_start, col_end = region
        height_region = height_map[row_start:row_end, col_start:col_end]
    else:
        height_region = height_map
    
    # Handle NaN values
    if np.any(np.isnan(height_region)):
        # In metadata.py context, we need to handle NaNs directly
        # Replace NaNs with the mean of valid values
        valid_data = ~np.isnan(height_region)
        if np.any(valid_data):
            mean_value = np.mean(height_region[valid_data])
            height_region = np.where(valid_data, height_region, mean_value)
        else:
            # If all values are NaN, use zeros
            height_region = np.zeros_like(height_region)
    
    # Calculate roughness metrics
    # 1. Root Mean Square Roughness (Rq)
    rq = np.sqrt(np.mean(height_region**2))
    
    # 2. Arithmetic Average Roughness (Ra)
    ra = np.mean(np.abs(height_region - np.mean(height_region)))
    
    # 3. Maximum Peak Height (Rp)
    rp = np.max(height_region) - np.mean(height_region)
    
    # 4. Maximum Valley Depth (Rv)
    rv = np.mean(height_region) - np.min(height_region)
    
    # 5. Maximum Height (Rt)
    rt = np.max(height_region) - np.min(height_region)
    
    # Return metrics
    return {
        'rq': float(rq),
        'ra': float(ra),
        'rp': float(rp),
        'rv': float(rv),
        'rt': float(rt),
        'mean': float(np.mean(height_region)),
        'std_dev': float(np.std(height_region))
    }


def calculate_surface_metrics(
    height_map: np.ndarray, 
    include_roughness: bool = True,
    sample_regions: Optional[List[Tuple[int, int, int, int]]] = None
) -> Dict[str, Any]:
    """
    Calculate comprehensive surface metrics from a height map.
    
    This function combines basic statistics with roughness metrics
    to provide a complete analysis of the surface characteristics.
    
    Args:
        height_map: 2D numpy array of height values
        include_roughness: Whether to include roughness metrics
        sample_regions: Optional list of regions to analyze separately
            as (row_start, row_end, col_start, col_end) tuples
            
    Returns:
        Dictionary containing all calculated metrics
    """
    # Get basic statistics
    basic_stats = compute_stats(height_map)
    
    # Initialize result
    metrics = {
        'basic': basic_stats
    }
    
    # Add roughness metrics if requested
    if include_roughness:
        roughness = analyze_surface_roughness(height_map)
        metrics['roughness'] = roughness
    
    # Process individual sample regions if provided
    if sample_regions:
        region_metrics = {}
        for i, region in enumerate(sample_regions):
            region_name = f"region_{i+1}"
            region_stats = compute_stats(height_map[region[0]:region[1], region[2]:region[3]])
            
            if include_roughness:
                region_roughness = analyze_surface_roughness(height_map, region)
                region_stats.update({f"roughness_{k}": v for k, v in region_roughness.items()})
                
            region_metrics[region_name] = region_stats
            
        metrics['regions'] = region_metrics
    
    return metrics
