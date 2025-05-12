"""
Heightmap export module.

This module provides functions for exporting heightmaps to image formats.
"""

import os
import numpy as np
import logging
from typing import Any, Dict, Optional, Tuple, Union

from .image_io import save_heightmap as save_image_heightmap
from .utils import ensure_directory_exists, normalize_heightmap, handle_nan_values

# Set up logging
logger = logging.getLogger(__name__)


def convert_heightmap_to_heightmap(
    height_map: np.ndarray,
    output_file: str,
    bit_depth: int = 16,
    normalize: bool = True,
    **kwargs
) -> Optional[str]:
    """
    Convert a heightmap to an image file.
    
    This function essentially passes through a heightmap for saving as an image,
    applying any requested normalization and other processing.
    
    Args:
        height_map: 2D numpy array of height values
        output_file: Path to save the output image
        bit_depth: Bit depth of output image (8 or 16)
        normalize: Whether to normalize the height values
        **kwargs: Additional options
        
    Returns:
        Path to the saved image or None if failed
    """
    try:
        # Ensure output directory exists
        if not ensure_directory_exists(output_file):
            logger.error(f"Failed to create output directory for {output_file}")
            return None
        
        # Process the heightmap
        processed_map = height_map.copy()
        
        # Handle NaN values if present
        if np.any(np.isnan(processed_map)):
            processed_map = handle_nan_values(processed_map)
        
        # Save to file
        return save_image_heightmap(
            processed_map, 
            output_file, 
            bit_depth=bit_depth,
            normalize=normalize,
            **kwargs
        )
        
    except Exception as e:
        logger.error(f"Error exporting heightmap: {e}")
        import traceback
        traceback.print_exc()
        return None


def convert_heightmap_to_heightmap(
    height_map: np.ndarray,
    output_file: str,
    bit_depth: int = 16,
    normalize: bool = True,
    preserve_precision: bool = True,
    precision: str = "float32",
    use_metadata: bool = True,
    custom_range: Optional[Tuple[float, float]] = None,
    compression_level: int = 6,
    **kwargs
) -> Optional[str]:
    """
    Convert a heightmap to an image file with enhanced precision.
    
    This function processes a heightmap for saving as an image,
    applying requested normalization and precision handling.
    
    Args:
        height_map: 2D numpy array of height values
        output_file: Path to save the output image
        bit_depth: Bit depth of output image (8, 16, or 32)
        normalize: Whether to normalize the height values
        preserve_precision: Whether to use techniques to preserve precision
        precision: Precision for internal calculations ("float16", "float32", "float64")
        use_metadata: Whether to embed metadata in the output file
        custom_range: Optional custom range for normalization (min, max)
        compression_level: Compression level for output (0-9, where 9 is max compression)
        **kwargs: Additional options
        
    Returns:
        Path to the saved image or None if failed
    """
    try:
        # Ensure output directory exists
        if not ensure_directory_exists(output_file):
            logger.error(f"Failed to create output directory for {output_file}")
            return None
        
        # Process the heightmap
        processed_map = height_map.copy()
        
        # Convert to requested precision for internal operations
        if precision == "float64":
            processed_map = processed_map.astype(np.float64)
        elif precision == "float32":
            processed_map = processed_map.astype(np.float32)
        elif precision == "float16":
            processed_map = processed_map.astype(np.float16)
        
        # Handle NaN values if present
        if np.any(np.isnan(processed_map)):
            processed_map = handle_nan_values(processed_map)
        
        # Store original range before normalization for metadata
        original_min = float(np.nanmin(processed_map))
        original_max = float(np.nanmax(processed_map))
        original_range = (original_min, original_max)
        
        # Apply custom normalization range if specified
        normalization_range = custom_range if custom_range else original_range
        
        # Prepare metadata for precision preservation
        metadata = {}
        if use_metadata and preserve_precision:
            metadata = {
                'original_min': original_min,
                'original_max': original_max,
                'precision': precision,
                'normalization_range': normalization_range
            }
        
        # Apply high-precision normalization if requested
        if normalize and preserve_precision:
            # Use high-precision operations for normalization
            norm_min, norm_max = normalization_range
            if norm_max > norm_min:  # Avoid division by zero
                range_size = norm_max - norm_min
                processed_map = (processed_map - norm_min) / range_size
                
                # Scale to bit depth maximum
                if bit_depth == 16:
                    processed_map = processed_map * 65535.0
                elif bit_depth == 32:
                    # For 32-bit float output, keep as normalized float
                    pass
                else:  # 8-bit
                    processed_map = processed_map * 255.0
        
        # Configure output options
        output_options = {
            'bit_depth': bit_depth,
            'normalize': normalize and not preserve_precision,  # Skip normalization if we've already done it
            'compression': compression_level,
            'metadata': metadata,
            **kwargs
        }
        
        # Save to file with enhanced options
        return save_image_heightmap(
            processed_map, 
            output_file,
            **output_options
        )
        
    except Exception as e:
        logger.error(f"Error exporting heightmap: {e}")
        import traceback
        traceback.print_exc()
        return None

def export_heightmap(
    height_map: np.ndarray,
    output_file: str,
    **kwargs
) -> Optional[Dict[str, Any]]:
    """
    Export a heightmap to an image file with enhanced precision options.
    
    Args:
        height_map: 2D numpy array of height values
        output_file: Path to save the output file
        **kwargs: Additional options for convert_heightmap_to_heightmap
        
    Returns:
        Dictionary with export statistics or None if failed
    """
    result = convert_heightmap_to_heightmap(height_map, output_file, **kwargs)
    
    if result:
        # Calculate and return statistics about the export
        stats = {
            'output_file': result,
            'dimensions': height_map.shape,
            'height_range': (float(np.nanmin(height_map)), float(np.nanmax(height_map))),
            'mean_height': float(np.nanmean(height_map)),
            'std_dev': float(np.nanstd(height_map)),
            'file_size': os.path.getsize(result) if os.path.exists(result) else 0
        }
        return stats
    
    return None

def batch_export_heightmaps(
    heightmaps: Dict[str, np.ndarray],
    output_dir: str,
    file_format: str = 'tiff',
    **kwargs
) -> Dict[str, Any]:
    """
    Export multiple heightmaps in a batch operation.
    
    Args:
        heightmaps: Dictionary of {name: heightmap_array} pairs
        output_dir: Directory to save output files
        file_format: File format extension (tiff, png, etc.)
        **kwargs: Additional options for convert_heightmap_to_heightmap
        
    Returns:
        Dictionary with statistics for each exported file
    """
    results = {}
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each heightmap
    for name, heightmap in heightmaps.items():
        output_file = os.path.join(output_dir, f"{name}.{file_format}")
        result = export_heightmap(heightmap, output_file, **kwargs)
        results[name] = result
    
    return results
