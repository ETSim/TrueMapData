"""
RGBD (RGB+Depth) map generation module for TMD.

This module provides functions for generating combined RGB and depth maps
from height maps for use in depth-based rendering and visualization.
"""

import os
import numpy as np
import logging
from typing import Optional, Dict, Any, Union, Tuple

from .utils import ensure_directory_exists, normalize_heightmap, handle_nan_values
from .image_io import save_image

# Set up logger
logger = logging.getLogger(__name__)

def export_rgbd_map(
    height_map: np.ndarray,
    output_file: str,
    color_source: Union[str, np.ndarray] = "height",
    depth_scale: float = 1.0,
    format: str = "png",
    bit_depth: int = 8,
    **kwargs
) -> Optional[str]:
    """
    Export an RGBD (color + depth) map from a height map.
    
    Args:
        height_map: 2D numpy array of height values
        output_file: Path to save the output image
        color_source: Source for color data ('height', 'colormap', or numpy array)
        depth_scale: Scale factor for depth values
        format: Output format (png, exr, pfm)
        bit_depth: Bit depth for output image (8 or 16)
        **kwargs: Additional options including:
            - colormap: Name of colormap for 'colormap' color_source
            - rgb_alpha: Opacity of RGB layer (0.0-1.0)
            - blend_mode: How to blend RGB and depth ('composite', 'alpha', 'separate')
        
    Returns:
        Path to the saved image or None if failed
    """
    try:
        # Ensure output directory exists
        if not ensure_directory_exists(os.path.dirname(os.path.abspath(output_file))):
            logger.error(f"Failed to create output directory for {output_file}")
            return None
        
        # Handle NaN values if present
        if np.any(np.isnan(height_map)):
            height_map = handle_nan_values(height_map, strategy=kwargs.get("nan_strategy", "mean"))
        
        # Create RGBD data
        rgbd_data = create_rgbd_data(
            height_map=height_map,
            color_source=color_source,
            depth_scale=depth_scale,
            **kwargs
        )
        
        rgb_data = rgbd_data['rgb']
        depth_data = rgbd_data['depth']
        
        # How to combine RGB and depth
        blend_mode = kwargs.get("blend_mode", "composite")
        
        if blend_mode == "separate" or format.lower() == "exr":
            # Save as multi-channel image with separate RGB and depth channels
            channels = {
                "color": rgb_data[:, :, :3],  # Ensure 3 channels
                "depth": depth_data
            }
            
            # Use existing multi-channel exporter if available
            try:
                # Define a simple function to save channels as EXR if the import fails
                result = save_multi_channel_image(channels, output_file, bit_depth=bit_depth, **kwargs)
                
                if result:
                    logger.info(f"RGBD map saved to {output_file}")
                
                return result
                
            except (ImportError, AttributeError, NameError):
                # Fall back to basic export
                logger.warning("Multi-channel image export not available. Saving composite instead.")
                blend_mode = "composite"
        
        if blend_mode == "alpha":
            # Composite using alpha blending
            rgb_alpha = kwargs.get("rgb_alpha", 0.7)
            
            # Create RGBA with depth as alpha
            rgba = np.zeros((rgb_data.shape[0], rgb_data.shape[1], 4), dtype=np.float32)
            rgba[:, :, :3] = rgb_data[:, :, :3]
            rgba[:, :, 3] = depth_data  # Use depth as alpha
            
            # Save RGBA image
            result = save_image(
                rgba,
                output_file,
                bit_depth=bit_depth,
                **kwargs
            )
        
        else:  # composite or fallback
            # Create a composite RGB image with depth encoded in luminance
            composite = rgb_data.copy()
            
            # Adjust brightness based on depth
            depth_factor = kwargs.get("depth_factor", 0.5)
            composite = composite * (1.0 - depth_factor + depth_factor * np.expand_dims(depth_data, axis=-1))
            
            # Save composite image
            result = save_image(
                composite,
                output_file,
                bit_depth=bit_depth,
                **kwargs
            )
        
        if result:
            logger.info(f"RGBD map saved to {output_file}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error exporting RGBD map: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_rgbd_data(
    height_map: np.ndarray,
    color_source: Union[str, np.ndarray] = "height",
    depth_scale: float = 1.0,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Create RGBD data from height map and optional color data.
    
    Args:
        height_map: Input height map for depth data
        color_source: Source for RGB data ('height', 'colormap', or numpy array)
        depth_scale: Scale factor for depth values
        **kwargs: Additional options including:
            - colormap: Name of colormap for 'colormap' color_source
        
    Returns:
        Dictionary with 'rgb' and 'depth' arrays
    """
    # Handle NaN values if present
    if np.any(np.isnan(height_map)):
        height_map = handle_nan_values(height_map, strategy=kwargs.get("nan_strategy", "mean"))
    
    # Process depth data
    depth_data = normalize_heightmap(height_map) * depth_scale
    
    # Process color data
    if isinstance(color_source, str):
        if color_source == "height" or color_source == "colormap":
            # Apply colormap to height data
            colormap = kwargs.get("colormap", "viridis")
            rgb_data = apply_colormap(normalize_heightmap(height_map), colormap=colormap)
        else:
            # Unknown color source, use grayscale
            rgb_data = np.stack([normalize_heightmap(height_map)] * 3, axis=-1)
    else:
        # Use provided RGB data
        rgb_data = color_source
        
        # Ensure it has 3 channels and matches height_map dimensions
        if rgb_data.ndim == 2:
            rgb_data = np.stack([rgb_data] * 3, axis=-1)
        elif rgb_data.shape[0:2] != height_map.shape:
            # Resize to match height map
            try:
                from PIL import Image
                img = Image.fromarray((rgb_data * 255).astype(np.uint8))
                img = img.resize((height_map.shape[1], height_map.shape[0]))
                rgb_data = np.array(img).astype(np.float32) / 255.0
            except ImportError:
                # If PIL is not available, just repeat the height map
                logger.warning("PIL not available for image resizing. Using height map directly.")
                rgb_data = np.stack([normalize_heightmap(height_map)] * 3, axis=-1)
    
    return {
        'rgb': rgb_data,
        'depth': depth_data
    }

def save_multi_channel_image(
    channels: Dict[str, np.ndarray],
    output_path: str,
    bit_depth: int = 8,
    **kwargs
) -> bool:
    """
    Save a multi-channel image (especially useful for EXR format).
    
    Args:
        channels: Dictionary of channel names to channel data
        output_path: Path to save the image
        bit_depth: Bit depth of output image
        **kwargs: Additional parameters
        
    Returns:
        True if successful, False otherwise
    """
    # Check if output format is EXR
    if output_path.lower().endswith('.exr'):
        return export_to_exr(channels, output_path, **kwargs)
    
    # Otherwise composite channels into RGB
    try:
        # Start with first channel or create black image
        if 'color' in channels or 'rgb' in channels:
            main_channel = 'color' if 'color' in channels else 'rgb'
            rgb_data = channels[main_channel]
        else:
            # Use first available channel
            channel_name = next(iter(channels))
            channel_data = channels[channel_name]
            
            # Convert to RGB if needed
            if channel_data.ndim == 2:
                rgb_data = np.stack([channel_data] * 3, axis=-1)
            else:
                rgb_data = channel_data
        
        # Save as regular image
        return bool(save_image(rgb_data, output_path, bit_depth=bit_depth, **kwargs))
        
    except Exception as e:
        logger.error(f"Error saving multi-channel image: {e}")
        return False

def export_to_exr(channels: Dict[str, np.ndarray], filename: str, **kwargs) -> bool:
    """
    Export channels to OpenEXR format.
    
    Args:
        channels: Dictionary of named channels
        filename: Output file path
        **kwargs: Additional options
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Try to import OpenEXR
        import OpenEXR
        import Imath
    except ImportError:
        logger.error("OpenEXR and Imath are required for EXR export")
        return False
    
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        
        # Ensure .exr extension
        if not filename.lower().endswith('.exr'):
            filename += '.exr'
        
        # Get dimensions from the first channel
        height, width = next(iter(channels.values())).shape[:2]
        
        # Set up header
        header = OpenEXR.Header(width, height)
        pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
        header['channels'] = {}
        
        # Process each channel
        channel_data = {}
        for channel_name, channel_array in channels.items():
            if channel_array.ndim == 3 and channel_array.shape[2] == 3:
                # RGB channel
                for i, c in enumerate("RGB"):
                    sub_channel = channel_array[:, :, i].astype(np.float32).tobytes()
                    channel_key = f"{channel_name}.{c}"
                    header['channels'][channel_key] = Imath.Channel(pixel_type)
                    channel_data[channel_key] = sub_channel
            else:
                # Grayscale channel
                if channel_array.ndim == 3:
                    channel_array = channel_array[:, :, 0]  # Take first channel
                
                sub_channel = channel_array.astype(np.float32).tobytes()
                channel_data[channel_name] = sub_channel
                header['channels'][channel_name] = Imath.Channel(pixel_type)
        
        # Create and write the OpenEXR file
        exr_file = OpenEXR.OutputFile(filename, header)
        exr_file.writePixels(channel_data)
        exr_file.close()
        
        logger.info(f"OpenEXR file saved to {filename}")
        return True
    
    except Exception as e:
        logger.error(f"Error exporting to OpenEXR: {e}")
        return False

def convert_heightmap_to_rgbd(
    height_map: np.ndarray,
    output_file: str,
    **kwargs
) -> Optional[str]:
    """
    Convert a heightmap to an RGBD map.
    
    This is an alias for export_rgbd_map to maintain a consistent API
    with other converter functions.
    
    Args:
        height_map: 2D numpy array of height values
        output_file: Path to save the output image
        **kwargs: Additional options for export_rgbd_map
        
    Returns:
        Path to the saved image or None if failed
    """
    return export_rgbd_map(height_map, output_file, **kwargs)

# Helper function to apply a colormap (since moved from metallic_depth_maps.py)
def apply_colormap(data: np.ndarray, colormap: str = "viridis") -> np.ndarray:
    """
    Apply a colormap to data.
    
    Args:
        data: Input array
        colormap: Name of colormap
        
    Returns:
        RGB array with colormap applied
    """
    try:
        from matplotlib import cm
        import matplotlib.pyplot as plt
        
        # Normalize data to 0-1 range
        if np.max(data) > np.min(data):
            normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
        else:
            normalized_data = np.zeros_like(data)
        
        # Apply colormap
        cmap = plt.get_cmap(colormap)
        rgb_data = cmap(normalized_data)
        
        # Return RGB channels (drop alpha if present)
        return rgb_data[:, :, :3]
        
    except ImportError:
        # Fallback if matplotlib not available
        logger.warning("Matplotlib not available. Using grayscale instead.")
        gray_data = np.stack([data] * 3, axis=-1)
        return gray_data / np.max(gray_data) if np.max(gray_data) > 0 else gray_data
