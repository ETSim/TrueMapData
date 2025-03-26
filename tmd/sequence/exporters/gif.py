"""
GIF exporter for height map sequences.

This module provides functionality to export height map sequences as animated GIF files.
"""

import os
import numpy as np
import logging
from typing import List, Optional, Union, Tuple

# Set up logger
logger = logging.getLogger(__name__)


def export_sequence_to_gif(
    frames: List[np.ndarray],
    output_file: str,
    fps: float = 10.0,
    colormap: str = "terrain",
    loop: int = 0,
    optimize: bool = True,
    duration: Optional[float] = None,
    show_progress: bool = True,
    **kwargs
) -> Optional[str]:
    """
    Export a sequence of height maps as an animated GIF.
    
    Args:
        frames: List of 2D numpy arrays representing height maps
        output_file: Path to save the GIF file
        fps: Frames per second (used to calculate duration)
        colormap: Matplotlib colormap name for rendering
        loop: Number of loops (0 = infinite)
        optimize: Whether to optimize the GIF
        duration: Duration per frame in milliseconds (overrides fps if provided)
        show_progress: Whether to show a progress bar
        **kwargs: Additional arguments passed to PIL's save method
        
    Returns:
        Path to the created file or None if failed
    """
    try:
        # Check for necessary libraries
        try:
            import matplotlib.pyplot as plt
            from matplotlib import cm
            from PIL import Image
            from io import BytesIO
            from tqdm import tqdm
        except ImportError as e:
            logger.error(f"Required package not found: {e}")
            logger.error("Please install matplotlib, Pillow and tqdm packages")
            return None
            
        # Check frames
        if not frames or len(frames) == 0:
            logger.error("No frames provided for GIF export")
            return None
            
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        # Ensure output file has .gif extension
        if not output_file.lower().endswith('.gif'):
            output_file += '.gif'
            
        # Calculate duration from fps if not provided
        if duration is None:
            duration = int(1000 / fps)  # Convert to milliseconds
            
        # Normalize data collectively for consistent color mapping
        all_min = min(np.min(frame) for frame in frames)
        all_max = max(np.max(frame) for frame in frames)
        norm_range = all_max - all_min
        
        if norm_range <= 0:
            norm_range = 1.0  # Avoid division by zero
            
        # Get colormap
        cmap = cm.get_cmap(colormap)
        
        # Process each frame into PIL images
        gif_frames = []
        
        # Use progress bar if requested
        frame_iterator = tqdm(frames, desc="Creating GIF") if show_progress else frames
        
        for frame in frame_iterator:
            # Normalize frame
            norm_frame = (frame - all_min) / norm_range
            
            # Convert to RGBA using colormap
            rgba_img = cmap(norm_frame)
            
            # Convert to 8-bit RGBA
            rgba_img_8bit = (rgba_img * 255).astype(np.uint8)
            
            # Create PIL image
            pil_img = Image.fromarray(rgba_img_8bit)
            
            gif_frames.append(pil_img)
            
        # Save as animated GIF
        if gif_frames:
            # First frame is used as the base
            first_frame = gif_frames[0]
            
            # Save with specified parameters
            first_frame.save(
                output_file,
                format='GIF',
                append_images=gif_frames[1:],
                save_all=True,
                duration=duration,
                loop=loop,
                optimize=optimize,
                **kwargs
            )
            
            logger.info(f"GIF animation saved to {output_file}")
            return output_file
        else:
            logger.error("No frames were processed for GIF export")
            return None
        
    except Exception as e:
        logger.error(f"Error exporting to GIF: {e}")
        import traceback
        traceback.print_exc()
        return None
