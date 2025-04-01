"""
Utility functions for image processing operations.

This module provides core functionality for handling images, including
saving, normalization, resizing, and NaN handling.
"""
import os
import logging
from typing import Optional, Union, Tuple

import numpy as np

logger = logging.getLogger(__name__)

def normalize_array(
    array: np.ndarray, 
    min_val: float = 0.0, 
    max_val: float = 1.0
) -> np.ndarray:
    """
    Normalize an array to a specified range.
    
    Args:
        array: Input array
        min_val: Minimum value for output range
        max_val: Maximum value for output range
        
    Returns:
        Normalized array as float32
    """
    # Handle empty or invalid input
    if array is None or array.size == 0:
        return np.zeros((1, 1), dtype=np.float32)
    
    # Handle constant arrays (avoid division by zero)
    array_min = np.min(array)
    array_max = np.max(array)
    
    if array_min == array_max:
        return np.ones_like(array, dtype=np.float32) * min_val
    
    # Normalize array to the desired range
    normalized = min_val + (max_val - min_val) * (array - array_min) / (array_max - array_min)
    return normalized.astype(np.float32)

def handle_nan_values(
    array: np.ndarray, 
    strategy: str = 'mean'
) -> np.ndarray:
    """
    Handle NaN values in an array using the specified strategy.
    
    Args:
        array: Input array with potential NaN values
        strategy: Strategy to use ('zero', 'mean', or 'nearest')
        
    Returns:
        Array with NaN values replaced
    """
    # Quick return if no NaNs
    if not np.any(np.isnan(array)):
        return array
    
    result = array.copy()
    
    if strategy == 'zero':
        # Replace NaNs with zeros
        result = np.nan_to_num(result, nan=0.0)
    elif strategy == 'mean':
        # Replace NaNs with the mean value
        mean_val = np.nanmean(result)
        result = np.nan_to_num(result, nan=mean_val)
    elif strategy == 'nearest':
        try:
            from scipy import ndimage
            # Create a mask of NaN values
            mask = np.isnan(result)
            
            # Create a kernel for neighboring pixels
            kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
            
            # Count neighboring non-NaN values
            neighbor_count = ndimage.convolve(~mask, kernel)
            
            # Sum of neighboring non-NaN values
            neighbor_sum = ndimage.convolve(np.where(mask, 0, result), kernel)
            
            # Replace NaNs with average of neighbors where possible
            avg_neighbors = np.zeros_like(result)
            valid_mask = neighbor_count > 0
            avg_neighbors[valid_mask] = neighbor_sum[valid_mask] / neighbor_count[valid_mask]
            
            # Apply the replacement
            result = np.where(mask, avg_neighbors, result)
            
            # Fill remaining NaNs with zeros
            result = np.nan_to_num(result, nan=0.0)
        except ImportError:
            logger.warning("SciPy not available for nearest neighbor method. Using mean instead.")
            mean_val = np.nanmean(result)
            result = np.nan_to_num(result, nan=mean_val)
    else:
        # Default fallback
        result = np.nan_to_num(result, nan=0.0)
        
    return result

def prepare_height_map(
    height_map: np.ndarray,
    nan_strategy: str = 'mean',
    normalize: bool = True,
    min_height: float = 0.0,
    max_height: float = 1.0,
    blur_radius: int = 0,
    **kwargs
) -> np.ndarray:
    """
    Prepare a height map for processing by handling NaNs, normalizing, etc.
    
    Args:
        height_map: Input height map
        nan_strategy: Strategy for handling NaNs ('zero', 'mean', 'nearest')
        normalize: Whether to normalize the map
        min_height: Minimum value after normalization
        max_height: Maximum value after normalization
        blur_radius: Radius for Gaussian blur (0 for no blur)
        **kwargs: Additional parameters (ignored)
        
    Returns:
        Processed height map
    """
    # Ensure we have a numpy array
    height_data = np.array(height_map, dtype=np.float32)
    
    # Handle NaN values if present
    if np.any(np.isnan(height_data)):
        logger.debug(f"Handling NaN values with strategy: {nan_strategy}")
        height_data = handle_nan_values(height_data, strategy=nan_strategy)
    
    # Apply Gaussian blur if requested
    if blur_radius > 0:
        try:
            from scipy.ndimage import gaussian_filter
            logger.debug(f"Applying Gaussian blur with radius {blur_radius}")
            height_data = gaussian_filter(height_data, sigma=blur_radius)
        except ImportError:
            logger.warning("SciPy not available. Skipping Gaussian blur.")
    
    # Normalize height range
    if normalize:
        logger.debug(f"Normalizing height map to range [{min_height}, {max_height}]")
        height_data = normalize_array(height_data, min_val=min_height, max_val=max_height)
    
    return height_data

def save_image(
    image: np.ndarray,
    filepath: str,
    bit_depth: int = 8,
    normalize: bool = True,
    colormap: Optional[str] = None,
    **kwargs
) -> Optional[str]:
    """
    Save an array as an image using the best available method.
    
    Args:
        image: Array to save as image
        filepath: Output path
        bit_depth: 8 or 16 bits per pixel
        normalize: Whether to normalize the data before saving
        colormap: Optional colormap name
        **kwargs: Additional parameters
        
    Returns:
        Path to saved image or None if failed
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    
    # Process the array for saving
    if normalize:
        img_data = normalize_array(image)
    else:
        img_data = image.copy()
    
    # Try to save using PIL (Pillow)
    try:
        from PIL import Image
        
        # Convert array to appropriate format for PIL
        if bit_depth == 16:
            img_array = (img_data * 65535).astype(np.uint16)
        else:
            img_array = (img_data * 255).astype(np.uint8)
        
        # Create PIL image based on input dimensions
        if img_array.ndim == 2:
            pil_img = Image.fromarray(img_array)
        elif img_array.ndim == 3 and img_array.shape[2] == 3:
            pil_img = Image.fromarray(img_array, 'RGB')
        elif img_array.ndim == 3 and img_array.shape[2] == 4:
            pil_img = Image.fromarray(img_array, 'RGBA')
        else:
            # Unsupported format
            raise ValueError(f"Unsupported array shape: {img_array.shape}")
        
        # Apply colormap if specified
        if colormap and img_array.ndim == 2:
            try:
                import matplotlib.pyplot as plt
                
                # Get colormap
                cm = plt.get_cmap(colormap)
                
                # Apply colormap to normalized data
                colored_data = cm(img_data)
                
                # Convert to 8-bit RGB
                rgb_array = (colored_data[:, :, :3] * 255).astype(np.uint8)
                
                # Create new PIL image
                pil_img = Image.fromarray(rgb_array, 'RGB')
            except (ImportError, ValueError):
                logger.warning(f"Could not apply colormap '{colormap}', saving as grayscale")
        
        # Save the image
        pil_img.save(filepath)
        logger.debug(f"Image saved to {filepath}")
        return filepath
        
    except (ImportError, Exception) as e:
        logger.warning(f"PIL save failed: {e}, trying matplotlib...")
    
    # Fallback to matplotlib if PIL fails
    try:
        import matplotlib.pyplot as plt
        
        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0, 0, 1, 1])
        ax.set_axis_off()
        fig.add_axes(ax)
        
        if colormap:
            ax.imshow(img_data, cmap=colormap, aspect='equal')
        else:
            ax.imshow(img_data, aspect='equal')
        
        fig.savefig(filepath, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        
        logger.debug(f"Image saved to {filepath} using Matplotlib")
        return filepath
        
    except (ImportError, Exception) as e:
        logger.error(f"Failed to save image: {e}")
        return None

def resize_image(
    image: np.ndarray, 
    width: Optional[int] = None, 
    height: Optional[int] = None
) -> np.ndarray:
    """
    Resize an image to specified dimensions.
    
    Args:
        image: Input image array
        width: Target width
        height: Target height
        
    Returns:
        Resized image array
    """
    if width is None and height is None:
        return image
        
    try:
        from PIL import Image
        
        # Convert to PIL Image
        if image.dtype == np.float32 or image.dtype == np.float64:
            img_array = (image * 255).astype(np.uint8)
        else:
            img_array = image
            
        # Create PIL image based on input dimensions
        if img_array.ndim == 2:
            pil_img = Image.fromarray(img_array)
        elif img_array.ndim == 3 and img_array.shape[2] == 3:
            pil_img = Image.fromarray(img_array, 'RGB')
        elif img_array.ndim == 3 and img_array.shape[2] == 4:
            pil_img = Image.fromarray(img_array, 'RGBA')
        else:
            raise ValueError(f"Unsupported array shape for resize: {img_array.shape}")
            
        # Calculate dimensions to preserve aspect ratio if needed
        orig_width, orig_height = pil_img.size
        if width is None:
            width = int(orig_width * (height / orig_height))
        elif height is None:
            height = int(orig_height * (width / orig_width))
            
        # Resize the image
        resized_img = pil_img.resize((width, height), Image.LANCZOS)
        
        # Convert back to numpy array
        resized_array = np.array(resized_img)
        if image.dtype == np.float32 or image.dtype == np.float64:
            resized_array = resized_array.astype(np.float32) / 255.0
            
        return resized_array
        
    except ImportError:
        logger.warning("PIL not available for image resizing. Returning original image.")
        return image

def is_valid_height_map(height_map: np.ndarray) -> bool:
    """
    Check if a height map is valid for processing.
    
    Args:
        height_map: Array to check
        
    Returns:
        True if the array is valid, False otherwise
    """
    if not isinstance(height_map, np.ndarray):
        return False
        
    if height_map.size == 0:
        return False
        
    if not np.issubdtype(height_map.dtype, np.number):
        return False
        
    if np.all(np.isnan(height_map)):
        return False
        
    return True

def get_output_filepath(output_path: str, format: str = None) -> str:
    """
    Get a properly formatted output filepath.
    
    Args:
        output_path: Base output path
        format: Optional format extension to enforce
        
    Returns:
        Formatted output filepath
    """
    path = str(output_path)
    
    # If format is specified, ensure the file has the correct extension
    if format:
        format = format.lower().lstrip('.')
        if not path.lower().endswith(f'.{format}'):
            path = f"{path}.{format}"
    
    return path
