""".

Multi-channel image export module for heightmaps.

This module provides functionality to create multi-channel images
from height maps, including normal maps, ambient occlusion, etc.
"""

import os
import numpy as np
import logging
from typing import Dict, Optional, Tuple, Union, List, Any

# Import local modules
from .utils import (
    normalize_height_map,
    generate_roughness_map,
    create_orm_map,
    generate_edge_map,
    save_texture,
    plot_textures,
    apply_colormap,
    apply_lighting,
    compose_multi_channel_image
)
from .normal_map import create_normal_map
from .ao_map import create_ambient_occlusion_map

# Set up logging
logger = logging.getLogger(__name__)

def export_multi_channel_image(
    height_map: np.ndarray,
    filename: str,
    include_channels: List[str] = ["color", "normal", "ao"],
    base_colormap: str = "viridis",
    z_scale: float = 1.0,
    normalize: bool = True,
    **kwargs
) -> bool:
    """.

    Export a height map as a multi-channel image with various material properties.
    
    Args:
        height_map: 2D numpy array of height values
        filename: Output filename
        include_channels: Channels to include in the output
                        ("color", "normal", "ao", "height", "roughness")
        base_colormap: Base colormap for the color channel
        z_scale: Z-scale factor for normal and ao maps
        normalize: Whether to normalize the height map to [0,1]
        **kwargs: Additional parameters for specific channels
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Normalize height map if requested
        if normalize:
            height_map_norm = normalize_height_map(height_map)
        else:
            height_map_norm = height_map.copy()
        
        # Process channels
        channels = {}
        
        # Base color channel (from height map)
        if "color" in include_channels:
            color_image = apply_colormap(height_map_norm, base_colormap)
            channels["color"] = color_image[:, :, :3]  # Remove alpha channel if present
        
        # Normal map channel
        if "normal" in include_channels:
            normal_map = create_normal_map(height_map_norm, z_scale=z_scale)
            # Convert from [-1,1] to [0,1] range for image
            normal_map = (normal_map + 1.0) * 0.5
            channels["normal"] = normal_map
        
        # Ambient occlusion channel
        if "ao" in include_channels:
            ao_map = create_ambient_occlusion_map(
                height_map_norm, 
                strength=kwargs.get("ao_strength", 1.0),
                samples=kwargs.get("ao_samples", 16)
            )
            channels["ao"] = ao_map
        
        # Height channel (grayscale)
        if "height" in include_channels:
            height_channel = height_map_norm[:, :, np.newaxis]
            height_channel = np.repeat(height_channel, 3, axis=2)
            channels["height"] = height_channel
        
        # Roughness channel (derived from height map)
        if "roughness" in include_channels:
            # Simple roughness derived from height variation
            roughness = create_roughness_map(height_map_norm, **kwargs)
            roughness_channel = roughness[:, :, np.newaxis]
            roughness_channel = np.repeat(roughness_channel, 3, axis=2)
            channels["roughness"] = roughness_channel
        
        # Save individual channel images if requested
        if kwargs.get("save_channels", False):
            save_individual_channels(channels, filename)
        
        # Export as appropriate format based on channels
        if len(channels) == 1:
            # Single channel export
            channel_name = list(channels.keys())[0]
            channel_data = channels[channel_name]
            return save_image(channel_data, filename, **kwargs)
        else:
            # Multi-channel export
            export_format = kwargs.get("format", "png").lower()
            
            if export_format == "exr" and "color" in channels and "normal" in channels:
                # OpenEXR format with multiple channels
                return export_to_exr(channels, filename, **kwargs)
            else:
                # Standard format - combine channels to RGB
                combined = combine_channels_to_rgb(channels, **kwargs)
                return save_image(combined, filename, **kwargs)
        
    except Exception as e:
        logger.error(f"Error exporting multi-channel image: {e}")
        return False

# Create alias with backward compatibility
convert_heightmap_to_multi_channel_map = export_multi_channel_image

def create_roughness_map(
    height_map: np.ndarray,
    method: str = "gradient",
    min_roughness: float = 0.1,
    max_roughness: float = 0.9,
    **kwargs
) -> np.ndarray:
    """.

    Create a roughness map from a height map.
    
    Args:
        height_map: 2D normalized height map in range [0,1]
        method: Method to use ('gradient', 'constant', 'inverse')
        min_roughness: Minimum roughness value
        max_roughness: Maximum roughness value
        **kwargs: Additional parameters
        
    Returns:
        2D numpy array of roughness values in range [0,1]
    """
    # Convert min_roughness and max_roughness to float to avoid type issues
    min_roughness = float(min_roughness)
    max_roughness = float(max_roughness)
    
    if method == "constant":
        roughness_value = float(kwargs.get("roughness_value", 0.5))
        return np.ones_like(height_map) * roughness_value
    
    elif method == "inverse":
        # Inverse relationship to height (higher = smoother)
        return min_roughness + (max_roughness - min_roughness) * (1.0 - height_map)
    
    else:  # Default: gradient method
        # For standard gradient method, use the optimized function from roughness_map
        from .roughness_map import generate_roughness_map
        
        # Get scale parameter - default to 1.0 if not specified
        scale = kwargs.get("scale", 1.0)
        kernel_size = kwargs.get("kernel_size", 3)
        
        # Generate roughness map using the standard function
        roughness = generate_roughness_map(height_map, kernel_size, scale)
        
        # Convert from [0-255] to [0-1] range
        roughness = roughness.astype(float) / 255.0
        
        # Apply min/max range
        if min_roughness > 0 or max_roughness < 1.0:
            roughness = min_roughness + roughness * (max_roughness - min_roughness)
            
        return roughness

def combine_channels_to_rgb(
    channels: Dict[str, np.ndarray],
    blend_mode: str = "overlay",
    weights: Optional[Dict[str, float]] = None,
    **kwargs
) -> np.ndarray:
    """.

    Combine multiple channels into a single RGB image.
    
    Args:
        channels: Dictionary of channel names to channel data
        blend_mode: How to blend channels ('overlay', 'multiply', 'add')
        weights: Optional dictionary of channel weights
        **kwargs: Additional parameters
        
    Returns:
        3D numpy array (H,W,3) of combined RGB values
    """
    if weights is None:
        # Default weights
        weights = {
            "color": 1.0,
            "normal": 0.0,
            "ao": 0.5,
            "height": 0.0,
            "roughness": 0.0
        }
    
    # Start with color or fallback to black
    if "color" in channels:
        result = channels["color"].copy()
    else:
        # Create black image
        h, w = next(iter(channels.values())).shape[:2]
        result = np.zeros((h, w, 3))
    
    # Apply ambient occlusion if present
    if "ao" in channels and weights.get("ao", 0.0) > 0:
        ao_strength = weights.get("ao", 0.5)
        ao_map = channels["ao"]
        
        if ao_map.ndim == 2:
            ao_map = ao_map[:, :, np.newaxis]
            ao_map = np.repeat(ao_map, 3, axis=2)
        
        if blend_mode == "multiply":
            # Multiply blend (darkens)
            result = result * (1.0 - ao_strength + ao_strength * ao_map)
        else:
            # Overlay blend
            result = result * (ao_map * ao_strength + (1.0 - ao_strength))
    
    # Apply normal map as lighting if requested
    if "normal" in channels and weights.get("normal", 0.0) > 0:
        normal_strength = weights.get("normal", 0.3)
        normal_map = channels["normal"]
        
        # Use normal map to calculate simple lighting
        light_dir = np.array(kwargs.get("light_direction", [0.5, 0.5, 1.0]))
        light_dir = light_dir / np.linalg.norm(light_dir)
        
        # Calculate dot product with light direction
        lighting = np.sum(normal_map * light_dir, axis=2)
        lighting = np.clip(lighting, 0, 1)
        lighting = lighting[:, :, np.newaxis]
        
        # Apply lighting effect
        result = result * (1.0 - normal_strength + normal_strength * lighting)
    
    # Apply roughness if present
    if "roughness" in channels and weights.get("roughness", 0.0) > 0:
        roughness_strength = weights.get("roughness", 0.2)
        roughness_map = channels["roughness"]
        
        if roughness_map.ndim == 2:
            roughness_map = roughness_map[:, :, np.newaxis]
            roughness_map = np.repeat(roughness_map, 3, axis=2)
        
        # Apply roughness (reduces contrast)
        mid_gray = np.ones_like(result) * 0.5
        result = result * (1.0 - roughness_strength) + mid_gray * roughness_strength * roughness_map
    
    # Ensure output is in range [0,1]
    return np.clip(result, 0, 1)

def save_individual_channels(
    channels: Dict[str, np.ndarray],
    base_filename: str
) -> Dict[str, str]:
    """.

    Save each channel as an individual image file.
    
    Args:
        channels: Dictionary of channel names to channel data
        base_filename: Base filename to use
        
    Returns:
        Dictionary mapping channel names to saved filenames
    """
    try:
        from PIL import Image
    except ImportError:
        logger.error("PIL/Pillow is required for saving images")
        return {}
    
    output_files = {}
    base_dir = os.path.dirname(base_filename)
    base_name = os.path.splitext(os.path.basename(base_filename))[0]
    ext = os.path.splitext(base_filename)[1] or ".png"
    
    for channel_name, channel_data in channels.items():
        # Create output filename
        output_path = os.path.join(base_dir, f"{base_name}_{channel_name}{ext}")
        
        # Ensure channel data is in expected format
        if channel_data.ndim == 2:
            # Convert grayscale to RGB
            img_data = np.repeat(channel_data[:, :, np.newaxis], 3, axis=2)
        else:
            img_data = channel_data
        
        # Convert to 8-bit
        img_data = (np.clip(img_data, 0, 1) * 255).astype(np.uint8)
        
        # Save the image
        Image.fromarray(img_data).save(output_path)
        output_files[channel_name] = output_path
        logger.info(f"Saved {channel_name} channel to {output_path}")
    
    return output_files

def save_image(image_data: np.ndarray, filename: str, **kwargs) -> bool:
    """
    Save an image array to a file.
    
    Args:
        image_data: Numpy array of image data (HxWxC)
        filename: Output filename
        **kwargs: Additional parameters
        
    Returns:
        True if successful, False otherwise
    """
    # Try to use the utils.save_image function if available
    try:
        from .utils import save_image as utils_save_image
        result = utils_save_image(image_data, filename, **kwargs)
        return result != ""
    except ImportError:
        pass
    
    # Check if PIL is available
    try:
        from PIL import Image
    except ImportError:
        logger.error("PIL/Pillow is required for saving images")
        return False
    
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        
        # Convert to 8-bit for standard formats
        if not filename.lower().endswith('.exr'):
            image_data = (np.clip(image_data, 0, 1) * 255).astype(np.uint8)
        
        # Save the image
        Image.fromarray(image_data).save(filename)
        logger.info(f"Image saved to {filename}")
        return True
    
    except Exception as e:
        logger.error(f"Error saving image: {e}")
        return False

def export_to_exr(channels: Dict[str, np.ndarray], filename: str, **kwargs) -> bool:
    """.

    Export channels to an OpenEXR file.
    
    Args:
        channels: Dictionary of channel names to channel data
        filename: Output filename
        **kwargs: Additional parameters
        
    Returns:
        True if successful, False otherwise
    """
    try:
        import OpenEXR
        import Imath
    except ImportError:
        logger.error("OpenEXR and Imath are required for EXR export")
        return False
    
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        
        # Ensure filename has .exr extension
        if not filename.lower().endswith('.exr'):
            filename += '.exr'
        
        # Get dimensions from the first channel
        height, width = next(iter(channels.values())).shape[:2]
        
        # Set up header
        header = OpenEXR.Header(width, height)
        
        # Prepare channel data
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

def create_pbr_material_set(
    height_map: np.ndarray,
    output_dir: str,
    base_name: str = "material",
    format: str = "png",
    z_scale: float = 1.0,
    **kwargs
) -> Dict[str, str]:
    """.

    Create a complete set of PBR (Physically Based Rendering) material maps.
    
    Args:
        height_map: 2D array of height values
        output_dir: Directory to save material maps
        base_name: Base name for output files
        format: Output format (e.g., 'png', 'jpg', 'exr')
        z_scale: Z-scale factor for normal and displacement maps
        **kwargs: Additional parameters for specific maps
        
    Returns:
        Dictionary mapping material map types to filenames
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Normalize height map
    height_map_norm = normalize_height_map(height_map)
    
    # Define the maps to generate
    material_maps = {
        "albedo": kwargs.get("albedo_method", "height_gradient"),
        "normal": True,
        "roughness": kwargs.get("roughness_method", "gradient"),
        "metallic": kwargs.get("metallic_method", "constant"),
        "ao": True,
        "height": True
    }
    
    output_files = {}
    
    # Process each material map
    for map_type, enabled in material_maps.items():
        if not enabled:
            continue
        
        # Generate output filename
        output_path = os.path.join(output_dir, f"{base_name}_{map_type}.{format}")
        
        # Generate the appropriate map
        if map_type == "albedo":
            # Color/albedo map
            if material_maps["albedo"] == "height_gradient":
                # Use height to determine color
                color_low = kwargs.get("albedo_color_low", [0.2, 0.2, 0.2])
                color_high = kwargs.get("albedo_color_high", [0.8, 0.8, 0.8])
                
                # Create gradient based on height
                albedo = np.zeros((height_map_norm.shape[0], height_map_norm.shape[1], 3))
                for i in range(3):
                    albedo[:, :, i] = color_low[i] + height_map_norm * (color_high[i] - color_low[i])
                
                # Save the albedo map
                save_image(albedo, output_path)
                output_files[map_type] = output_path
            
            elif material_maps["albedo"] == "colormap":
                # Use a matplotlib colormap
                colormap = kwargs.get("albedo_colormap", "viridis")
                albedo = apply_colormap(height_map_norm, colormap)
                
                # Save the albedo map (RGB only)
                save_image(albedo[:, :, :3], output_path)
                output_files[map_type] = output_path
        
        elif map_type == "normal":
            # Normal map
            normal_map = create_normal_map(height_map_norm, z_scale=z_scale)
            # Convert from [-1,1] to [0,1] range for image
            normal_map = (normal_map + 1.0) * 0.5
            
            # Save the normal map
            save_image(normal_map, output_path)
            output_files[map_type] = output_path
        
        elif map_type == "roughness":
            # Roughness map
            method = kwargs.get("roughness_method", "gradient")
            min_val = kwargs.get("roughness_min", 0.1)
            max_val = kwargs.get("roughness_max", 0.9)
            
            roughness = create_roughness_map(
                height_map_norm,
                method=method,
                min_roughness=min_val,
                max_roughness=max_val,
                **kwargs
            )
            
            # Convert to RGB for consistent output
            roughness_rgb = np.repeat(roughness[:, :, np.newaxis], 3, axis=2)
            
            # Save the roughness map
            save_image(roughness_rgb, output_path)
            output_files[map_type] = output_path
        
        elif map_type == "metallic":
            # Metallic map
            method = kwargs.get("metallic_method", "constant")
            
            if method == "constant":
                # Constant metallic value
                value = kwargs.get("metallic_value", 0.0)
                metallic = np.ones_like(height_map_norm) * value
            elif method == "height_threshold":
                # Areas above threshold are metallic
                threshold = kwargs.get("metallic_threshold", 0.7)
                value = kwargs.get("metallic_value", 1.0)
                metallic = np.where(height_map_norm > threshold, value, 0.0)
            else:
                # Default to non-metallic
                metallic = np.zeros_like(height_map_norm)
            
            # Convert to RGB for consistent output
            metallic_rgb = np.repeat(metallic[:, :, np.newaxis], 3, axis=2)
            
            # Save the metallic map
            save_image(metallic_rgb, output_path)
            output_files[map_type] = output_path
        
        elif map_type == "ao":
            # Ambient occlusion map
            strength = kwargs.get("ao_strength", 1.0)
            samples = kwargs.get("ao_samples", 16)
            
            ao_map = create_ambient_occlusion_map(
                height_map_norm,
                strength=strength,
                samples=samples
            )
            
            # Convert to RGB for consistent output
            ao_rgb = np.repeat(ao_map[:, :, np.newaxis], 3, axis=2)
            
            # Save the AO map
            save_image(ao_rgb, output_path)
            output_files[map_type] = output_path
        
        elif map_type == "height":
            # Height/displacement map
            scale = kwargs.get("height_scale", 1.0)
            
            # Scale the normalized height map
            height_disp = height_map_norm * scale
            
            # Convert to RGB for consistent output
            height_rgb = np.repeat(height_disp[:, :, np.newaxis], 3, axis=2)
            
            # Save the height/displacement map
            save_image(height_rgb, output_path)
            output_files[map_type] = output_path
    
    return output_files
