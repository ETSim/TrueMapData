"""
Core functionality for exporting height maps as various image formats.

This module provides functions to export height maps as different types of
visualization images, including normal maps, displacement maps, etc.
"""

import os
import logging
import numpy as np
from typing import Optional, Union, Tuple, Dict, Any

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

logger = logging.getLogger(__name__)

# Try to import OpenCV for additional functionality
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    logger.warning("OpenCV not found. Some functions may be limited.")


def export_heightmap_image(
    height_map: np.ndarray,
    filename: str,
    colormap: Optional[str] = None,
    normalize: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    dpi: int = 300,
    **kwargs
) -> str:
    """
    Export a height map as an image file.
    
    Args:
        height_map: 2D array of height values
        filename: Output filename
        colormap: Optional colormap name (None for grayscale)
        normalize: Whether to normalize height values
        vmin: Minimum value for normalization
        vmax: Maximum value for normalization
        dpi: Dots per inch for output image
        **kwargs: Additional arguments for plt.imsave
        
    Returns:
        Path to the saved file
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    
    # Create a copy to avoid modifying the original
    h_map = height_map.copy()
    
    # Normalize if requested
    if normalize:
        if vmin is None:
            vmin = np.nanmin(h_map)
        if vmax is None:
            vmax = np.nanmax(h_map)
            
        if vmax > vmin:
            h_map = (h_map - vmin) / (vmax - vmin)
        else:
            h_map = np.zeros_like(h_map)
    
    # Export using matplotlib
    plt.imsave(filename, h_map, cmap=colormap, dpi=dpi, **kwargs)
    logger.info(f"Heightmap image exported to {filename}")
    
    return filename


def export_normal_map(
    height_map: np.ndarray,
    filename: str,
    strength: float = 1.0,
    resolution: float = 1.0,
    output_format: str = 'RGB',
    normalize_z: bool = True,
    **kwargs
) -> str:
    """
    Calculate and export a normal map from a height map.
    
    Args:
        height_map: 2D array of height values
        filename: Output filename
        strength: Normal map strength factor (higher values exaggerate features)
        resolution: Resolution factor for normal calculation
        output_format: Output format ('RGB' or 'XYZ')
        normalize_z: Whether to normalize Z values
        **kwargs: Additional arguments for matplotlib or OpenCV
        
    Returns:
        Path to the saved file
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    
    # Calculate normal map
    if HAS_OPENCV:
        normal_map = _calculate_normal_map_cv2(height_map, strength, resolution, normalize_z)
    else:
        normal_map = _calculate_normal_map_numpy(height_map, strength, resolution, normalize_z)
    
    # Convert to output format if needed
    if output_format.upper() == 'RGB':
        # Convert -1,1 range to 0,1 range for RGB
        normal_map = (normal_map + 1.0) * 0.5
        
    # Export as image
    plt.imsave(filename, normal_map, **kwargs)
    logger.info(f"Normal map exported to {filename}")
    
    return filename


def _calculate_normal_map_cv2(
    height_map: np.ndarray,
    strength: float = 1.0,
    resolution: float = 1.0,
    normalize_z: bool = True
) -> np.ndarray:
    """
    Calculate normal map using OpenCV for better performance.
    """
    # Scale the strength by the resolution
    scaled_strength = strength / resolution
    
    # Create a copy and ensure correct data type
    h_map = height_map.astype(np.float32)
    
    # Calculate gradients using Sobel
    dx = cv2.Sobel(h_map, cv2.CV_32F, 1, 0, ksize=3) * scaled_strength
    dy = cv2.Sobel(h_map, cv2.CV_32F, 0, 1, ksize=3) * scaled_strength
    
    # Create normal map
    normal_map = np.zeros((height_map.shape[0], height_map.shape[1], 3), dtype=np.float32)
    normal_map[..., 0] = -dx
    normal_map[..., 1] = -dy
    normal_map[..., 2] = 1.0
    
    # Normalize vectors
    if normalize_z:
        norms = np.sqrt(np.sum(normal_map**2, axis=2, keepdims=True))
        np.divide(normal_map, norms, out=normal_map, where=norms != 0)
    
    return normal_map


def _calculate_normal_map_numpy(
    height_map: np.ndarray,
    strength: float = 1.0,
    resolution: float = 1.0,
    normalize_z: bool = True
) -> np.ndarray:
    """
    Calculate normal map using NumPy (fallback if OpenCV is not available).
    """
    # Scale the strength by the resolution
    scaled_strength = strength / resolution
    
    # Create gradient arrays
    h, w = height_map.shape
    dx = np.zeros((h, w))
    dy = np.zeros((h, w))
    
    # Calculate x gradients (left-right)
    dx[:, 1:-1] = (height_map[:, 2:] - height_map[:, :-2]) * 0.5
    dx[:, 0] = height_map[:, 1] - height_map[:, 0]
    dx[:, -1] = height_map[:, -1] - height_map[:, -2]
    
    # Calculate y gradients (up-down)
    dy[1:-1, :] = (height_map[2:, :] - height_map[:-2, :]) * 0.5
    dy[0, :] = height_map[1, :] - height_map[0, :]
    dy[-1, :] = height_map[-1, :] - height_map[-2, :]
    
    # Scale gradients
    dx *= scaled_strength
    dy *= scaled_strength
    
    # Create normal map
    normal_map = np.zeros((h, w, 3), dtype=np.float32)
    normal_map[..., 0] = -dx
    normal_map[..., 1] = -dy
    normal_map[..., 2] = 1.0
    
    # Normalize vectors
    if normalize_z:
        norms = np.sqrt(np.sum(normal_map**2, axis=2, keepdims=True))
        np.divide(normal_map, norms, out=normal_map, where=norms != 0)
    
    return normal_map


def export_displacement_map(
    height_map: np.ndarray,
    filename: str,
    invert: bool = False,
    bit_depth: int = 8,
    normalize: bool = True,
    **kwargs
) -> str:
    """
    Export a displacement map from a height map.
    
    Args:
        height_map: 2D array of height values
        filename: Output filename
        invert: Whether to invert the values (black=high, white=low)
        bit_depth: Bit depth for output (8 or 16)
        normalize: Whether to normalize height values
        **kwargs: Additional arguments for image export
        
    Returns:
        Path to the saved file
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    
    # Create a copy to avoid modifying the original
    h_map = height_map.copy()
    
    # Normalize if requested
    if normalize:
        h_min = np.nanmin(h_map)
        h_max = np.nanmax(h_map)
        
        if h_max > h_min:
            h_map = (h_map - h_min) / (h_max - h_min)
        else:
            h_map = np.zeros_like(h_map)
    
    # Invert if requested
    if invert:
        h_map = 1.0 - h_map
    
    # Save based on bit depth
    if HAS_OPENCV:
        if bit_depth == 16:
            # Convert to 16-bit
            h_map = (h_map * 65535).astype(np.uint16)
            cv2.imwrite(filename, h_map)
        else:
            # Default to 8-bit
            h_map = (h_map * 255).astype(np.uint8)
            cv2.imwrite(filename, h_map)
    else:
        # Fall back to matplotlib (only supports 8-bit)
        plt.imsave(filename, h_map, cmap='gray', **kwargs)
    
    logger.info(f"Displacement map exported to {filename}")
    return filename


def export_ambient_occlusion(
    height_map: np.ndarray,
    filename: str,
    strength: float = 1.0,
    samples: int = 16,
    radius: float = 0.1,
    **kwargs
) -> str:
    """
    Calculate and export an ambient occlusion map from a height map.
    
    Args:
        height_map: 2D array of height values
        filename: Output filename
        strength: Strength of the AO effect
        samples: Number of sample directions
        radius: Sampling radius relative to height map size
        **kwargs: Additional arguments for image export
        
    Returns:
        Path to the saved file
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    
    # Normalize height map
    h_map = height_map.copy()
    h_min = np.nanmin(h_map)
    h_max = np.nanmax(h_map)
    
    if h_max > h_min:
        h_map = (h_map - h_min) / (h_max - h_min)
    else:
        h_map = np.zeros_like(h_map)
    
    # Create ambient occlusion map
    ao_map = _calculate_ambient_occlusion(h_map, samples, radius, strength)
    
    # Save the image
    plt.imsave(filename, ao_map, cmap='gray', **kwargs)
    logger.info(f"Ambient occlusion map exported to {filename}")
    
    return filename


def _calculate_ambient_occlusion(
    height_map: np.ndarray,
    samples: int = 16,
    radius: float = 0.1,
    strength: float = 1.0
) -> np.ndarray:
    """
    Calculate ambient occlusion using a sampling approach.
    
    This is a simplified implementation that works by casting rays
    in different directions and checking for height differences.
    """
    h, w = height_map.shape
    actual_radius = int(max(h, w) * radius)
    ao_map = np.ones_like(height_map)
    
    # Generate sample directions around a hemisphere
    angles = np.linspace(0, 2 * np.pi, samples, endpoint=False)
    dirs_x = np.cos(angles) * actual_radius
    dirs_y = np.sin(angles) * actual_radius
    
    # For each direction, calculate occlusion
    for dx, dy in zip(dirs_x, dirs_y):
        dx_int, dy_int = int(dx), int(dy)
        
        # Skip if direction is too small
        if dx_int == 0 and dy_int == 0:
            continue
        
        # Create shifted height maps
        x_indices = np.clip(np.arange(w) + dx_int, 0, w - 1)
        y_indices = np.clip(np.arange(h) + dy_int, 0, h - 1)
        y_grid, x_grid = np.meshgrid(y_indices, x_indices, indexing='ij')
        
        # Get height at shifted positions
        shifted_heights = height_map[y_grid, x_grid]
        
        # Calculate occlusion factor
        height_diff = shifted_heights - height_map
        occlusion = np.maximum(0, height_diff) * strength
        
        # Apply to AO map
        ao_map -= occlusion / samples
    
    # Ensure values are in valid range
    ao_map = np.clip(ao_map, 0, 1)
    
    return ao_map


def batch_export_maps(
    height_map: np.ndarray,
    output_dir: str,
    base_name: str = "heightmap",
    formats: Optional[Dict[str, bool]] = None,
    **kwargs
) -> Dict[str, str]:
    """
    Export multiple map formats from a single height map.
    
    Args:
        height_map: 2D array of height values
        output_dir: Directory to save files
        base_name: Base filename to use
        formats: Dictionary of formats to export {format_name: enabled}
        **kwargs: Additional arguments for specific exporters
        
    Returns:
        Dictionary mapping format names to exported file paths
    """
    # Ensure directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Default formats to export
    if formats is None:
        formats = {
            "heightmap": True,
            "normal_map": True,
            "displacement_map": True,
            "ambient_occlusion": False,  # Disabled by default as it's slower
            "colored_heightmap": True
        }
    
    # Initialize results
    results = {}
    
    # Export each selected format
    if formats.get("heightmap", False):
        filename = os.path.join(output_dir, f"{base_name}.png")
        results["heightmap"] = export_heightmap_image(
            height_map, filename, 
            colormap=None,
            normalize=True,
            **kwargs.get("heightmap", {})
        )
    
    if formats.get("normal_map", False):
        filename = os.path.join(output_dir, f"{base_name}_normal.png")
        results["normal_map"] = export_normal_map(
            height_map, filename,
            **kwargs.get("normal_map", {})
        )
    
    if formats.get("displacement_map", False):
        filename = os.path.join(output_dir, f"{base_name}_displacement.png")
        results["displacement_map"] = export_displacement_map(
            height_map, filename,
            **kwargs.get("displacement_map", {})
        )
    
    if formats.get("ambient_occlusion", False):
        filename = os.path.join(output_dir, f"{base_name}_ao.png")
        results["ambient_occlusion"] = export_ambient_occlusion(
            height_map, filename,
            **kwargs.get("ambient_occlusion", {})
        )
    
    if formats.get("colored_heightmap", False):
        filename = os.path.join(output_dir, f"{base_name}_colored.png")
        results["colored_heightmap"] = export_heightmap_image(
            height_map, filename,
            colormap="terrain",
            normalize=True,
            **kwargs.get("colored_heightmap", {})
        )
    
    logger.info(f"Batch exported maps to {output_dir}")
    return results
