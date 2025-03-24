""".

Image I/O utilities for TMD.

This module provides functions for loading and saving height maps and other images
in various formats, including special handling for EXR, NPY, and NPZ files.
"""

import os
import logging
import numpy as np
from enum import Enum
from typing import Union, Optional, Tuple, List, Dict, Any

# Set up logging
logger = logging.getLogger(__name__)

# Check for available image processing libraries
try:
    import cv2
    has_opencv = True
except ImportError:
    has_opencv = False
    logger.warning("OpenCV not found. Some image formats may not be supported.")

try:
    from PIL import Image
    has_pil = True
except ImportError:
    has_pil = False
    logger.warning("PIL not found. Some image formats may not be supported.")

class ImageType(Enum):
    """Enum for image types.."""
    COLOR = "color"
    HEIGHTMAP = "heightmap"
    NORMAL_MAP = "normal_map"
    ROUGHNESS = "roughness"
    AMBIENT_OCCLUSION = "ambient_occlusion"
    MASK = "mask"

def load_image(filepath: str,
               image_type: Union[str, ImageType] = "auto",
               normalize: bool = True,
               channels: Optional[int] = None) -> np.ndarray:
    """.

    Load an image from various file formats, with special handling for EXR and other formats.

    Args:
        filepath: Path to the image file
        image_type: Type of image to load, or "auto" to determine from file extension
        normalize: Whether to normalize values to [0,1] range for float arrays
        channels: Number of channels to return (None=auto, 1=grayscale, 3=RGB, 4=RGBA)

    Returns:
        numpy.ndarray: Image data as a 2D or 3D array
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Image file not found: {filepath}")

    file_ext = os.path.splitext(filepath)[1].lower()

    # Determine expected image type from extension if set to auto
    if image_type == "auto":
        if file_ext in ['.exr', '.hdr']:
            image_type = ImageType.HEIGHTMAP
        elif file_ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
            # Determine from filename if it contains type hints
            filename = os.path.basename(filepath).lower()
            if any(x in filename for x in ['normal', 'norm']):
                image_type = ImageType.NORMAL_MAP
            elif any(x in filename for x in ['height', 'displacement', 'depth']):
                image_type = ImageType.HEIGHTMAP
            elif any(x in filename for x in ['rough']):
                image_type = ImageType.ROUGHNESS
            elif any(x in filename for x in ['ao', 'ambient', 'occlusion']):
                image_type = ImageType.AMBIENT_OCCLUSION
            elif any(x in filename for x in ['mask']):
                image_type = ImageType.MASK
            else:
                image_type = ImageType.COLOR
        elif file_ext in ['.npz', '.npy']:
            image_type = ImageType.HEIGHTMAP
        else:
            image_type = ImageType.COLOR

    # Convert string to enum if needed
    if isinstance(image_type, str):
        try:
            image_type = ImageType(image_type)
        except ValueError:
            logger.warning(f"Unknown image type '{image_type}', defaulting to 'color'")
            image_type = ImageType.COLOR

    try:
        # Special handling for NumPy formats
        if file_ext == '.npy':
            image_data = np.load(filepath)
            
            # Handle normalization
            if normalize and image_data.dtype != np.uint8:
                image_data = _normalize_array(image_data)
                
            return image_data
            
        if file_ext == '.npz':
            npz_data = np.load(filepath)
            
            # Look for height map in common keys
            height_map_keys = ['height_map', 'heightmap', 'heights', 'z', 'data']
            
            for key in height_map_keys:
                if key in npz_data:
                    image_data = npz_data[key]
                    
                    # Handle normalization
                    if normalize and image_data.dtype != np.uint8:
                        image_data = _normalize_array(image_data)
                        
                    return image_data
            
            # If no specific height map key found, use the first array
            for key in npz_data.keys():
                image_data = npz_data[key]
                
                # Handle normalization
                if normalize and image_data.dtype != np.uint8:
                    image_data = _normalize_array(image_data)
                    
                return image_data
            
            raise ValueError(f"Could not find valid height map data in NPZ file: {filepath}")
            
        # Load image according to type using available libraries
        if has_opencv:
            # Use OpenCV for image loading
            if file_ext == '.exr':
                # For EXR files, use specific flags
                flags = cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH
                if image_type == ImageType.HEIGHTMAP:
                    flags |= cv2.IMREAD_GRAYSCALE
                image_data = cv2.imread(filepath, flags)
                if image_data is None:
                    raise ValueError(f"Failed to load EXR file: {filepath}")

            elif file_ext in ['.hdr']:
                # For HDR files
                image_data = cv2.imread(filepath, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                if image_data is None:
                    raise ValueError(f"Failed to load HDR file: {filepath}")

            else:
                # For other image formats
                if image_type == ImageType.HEIGHTMAP:
                    image_data = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
                elif image_type == ImageType.MASK:
                    image_data = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                    # Convert to binary if needed
                    if image_data is not None:
                        image_data = (image_data > 127).astype(np.uint8) * 255
                else:
                    image_data = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)

                if image_data is None:
                    raise ValueError(f"Failed to load image file: {filepath}")
        elif has_pil:
            # Use PIL as fallback
            img = Image.open(filepath)
            
            if image_type == ImageType.HEIGHTMAP:
                # Convert to grayscale
                if img.mode != 'L' and img.mode != 'I' and img.mode != 'F':
                    img = img.convert('L')
            
            image_data = np.array(img)
        else:
            raise ImportError("No image processing library available (need OpenCV or PIL)")

        # Handle channel conversion
        if channels is not None and image_data.ndim > 2:
            if channels == 1 and image_data.shape[2] > 1:
                # Convert to grayscale
                if has_opencv:
                    image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
                else:
                    # Simple average for grayscale
                    image_data = np.mean(image_data, axis=2).astype(image_data.dtype)
            elif channels == 3 and image_data.shape[2] == 4:
                # Remove alpha channel
                image_data = image_data[:, :, :3]
            elif channels == 4 and image_data.shape[2] == 3:
                # Add alpha channel
                alpha = np.ones((image_data.shape[0], image_data.shape[1], 1), 
                               dtype=image_data.dtype)
                if image_data.dtype == np.uint8:
                    alpha *= 255
                image_data = np.concatenate([image_data, alpha], axis=2)

        # Normalize if requested and not already 8-bit
        if normalize and image_data.dtype != np.uint8:
            image_data = _normalize_array(image_data)

        return image_data

    except Exception as e:
        logger.error(f"Error loading image from {filepath}: {e}")
        raise

def save_image(image_data: np.ndarray,
              filepath: str,
              normalize: bool = False) -> str:
    """.

    Save an image to a file.

    Args:
        image_data: Image data as a numpy array
        filepath: Output file path
        normalize: Whether to normalize the image values before saving

    Returns:
        str: Path to the saved file
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    
    # Get file extension
    file_ext = os.path.splitext(filepath)[1].lower()
    
    try:
        # Handle NPY/NPZ formats directly
        if file_ext == '.npy':
            np.save(filepath, image_data)
            logger.debug(f"Saved array to {filepath}")
            return filepath
        
        if file_ext == '.npz':
            np.savez_compressed(filepath, height_map=image_data)
            logger.debug(f"Saved compressed array to {filepath}")
            return filepath
        
        # For other formats, use OpenCV or PIL
        if normalize:
            image_data = _normalize_array(image_data, target_type=np.uint8, target_range=(0, 255))
        
        if has_opencv:
            # Use OpenCV for saving
            if file_ext == '.exr':
                # For EXR, ensure data is float32
                if image_data.dtype != np.float32:
                    image_data = image_data.astype(np.float32)
                cv2.imwrite(filepath, image_data)
            else:
                # For other formats, ensure 8-bit
                if image_data.dtype != np.uint8:
                    image_data = (image_data * 255).clip(0, 255).astype(np.uint8)
                cv2.imwrite(filepath, image_data)
        elif has_pil:
            # Use PIL as fallback
            if image_data.dtype != np.uint8:
                image_data = (image_data * 255).clip(0, 255).astype(np.uint8)
            
            img = Image.fromarray(image_data)
            img.save(filepath)
        else:
            raise ImportError("No image processing library available (need OpenCV or PIL)")
        
        logger.debug(f"Saved image to {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Error saving image to {filepath}: {e}")
        raise

def _normalize_array(arr: np.ndarray,
                    target_type: np.dtype = np.float32,
                    target_range: Tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
    """.

    Normalize an array to a target range.

    Args:
        arr: Input array
        target_type: Target data type
        target_range: Target range as (min, max)

    Returns:
        np.ndarray: Normalized array
    """
    # Convert target range values to float
    target_min = float(target_range[0])
    target_max = float(target_range[1])
    
    # Get current range
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    
    # If the range is zero, return uniform array
    if arr_min == arr_max:
        uniform_val = target_min
        return np.full_like(arr, uniform_val, dtype=target_type)
    
    # Scale to target range
    normalized = ((arr - arr_min) / (arr_max - arr_min) * 
                 (target_max - target_min) + target_min)
    
    # Convert to target type
    return normalized.astype(target_type)

def save_heightmap(height_map: np.ndarray,
                  filepath: str,
                  normalize: bool = False) -> str:
    """.

    Save a height map to a file.

    Args:
        height_map: Height map data as a numpy array
        filepath: Output file path
        normalize: Whether to normalize the height values before saving

    Returns:
        str: Path to the saved file
    """
    return save_image(height_map, filepath, normalize)

def load_heightmap(filepath: str,
                  normalize: bool = True) -> np.ndarray:
    """.

    Load a height map from a file.

    Args:
        filepath: Path to the height map file
        normalize: Whether to normalize height values to [0,1] range

    Returns:
        np.ndarray: Height map data
    """
    return load_image(filepath, image_type=ImageType.HEIGHTMAP, normalize=normalize)

def load_normal_map(filepath: str,
                   normalize: bool = True) -> np.ndarray:
    """.

    Load a normal map from a file.

    Args:
        filepath: Path to the normal map file
        normalize: Whether to normalize values to [0,1] range

    Returns:
        np.ndarray: Normal map data
    """
    return load_image(filepath, image_type=ImageType.NORMAL_MAP, normalize=normalize)

def load_mask(filepath: str) -> np.ndarray:
    """.

    Load a binary mask from a file.

    Args:
        filepath: Path to the mask file

    Returns:
        np.ndarray: Binary mask (0 or 1 values)
    """
    mask = load_image(filepath, image_type=ImageType.MASK, normalize=False)
    
    # Ensure binary
    if mask.dtype != bool:
        mask = mask > 127
    
    return mask.astype(bool)
