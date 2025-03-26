"""
Video exporter for height map sequences.

This module provides functionality to export height map sequences to video
formats like MP4, using matplotlib for visualization.
"""

import os
import numpy as np
import logging
from typing import List, Optional, Union, Tuple, Dict

# Set up logger
logger = logging.getLogger(__name__)


def export_sequence_to_video(
    frames: List[np.ndarray],
    output_file: str,
    fps: float = 30.0,
    colormap: str = "terrain",
    dpi: int = 100,
    quality: Optional[int] = None,
    show_progress: bool = True,
    bitrate: Optional[int] = None,
    codec: Optional[str] = None,
    **kwargs
) -> Optional[str]:
    """
    Export a sequence of height maps to a video file.
    
    Args:
        frames: List of 2D numpy arrays representing height maps
        output_file: Path to save the video file (should end with .mp4)
        fps: Frames per second
        colormap: Matplotlib colormap name for rendering
        dpi: Resolution for rendered frames
        quality: Optional video quality (0-10, higher is better)
        show_progress: Whether to show a progress bar
        bitrate: Optional bitrate for encoding
        codec: Optional video codec
        **kwargs: Additional arguments passed to matplotlib's animation.save
    
    Returns:
        Path to the created file or None if failed
    """
    try:
        # Check for necessary libraries
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            import matplotlib.animation as animation
            from matplotlib import cm
            from tqdm import tqdm
        except ImportError as e:
            logger.error(f"Required package not found: {e}")
            logger.error("Please install matplotlib and tqdm packages")
            return None
            
        # Check frames
        if not frames or len(frames) == 0:
            logger.error("No frames provided for video export")
            return None
        
        # Set non-interactive backend to avoid display
        matplotlib.use('Agg')
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        # Ensure output file has a video extension
        if not output_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            output_file += '.mp4'
            
        # Create figure and first frame
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Normalize data collectively for consistent color mapping
        all_min = min(np.min(frame) for frame in frames)
        all_max = max(np.max(frame) for frame in frames)
        norm_range = all_max - all_min
        
        if norm_range <= 0:
            norm_range = 1.0  # Avoid division by zero
        
        # Create initial plot
        norm_frame = (frames[0] - all_min) / norm_range
        im = ax.imshow(norm_frame, cmap=colormap, animated=True)
        ax.axis('off')  # Remove axes
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        # Title showing frame number
        title = ax.text(0.5, 1.05, 'Frame: 0', 
                        size=plt.rcParams["axes.titlesize"],
                        ha="center", transform=ax.transAxes)
        
        # Function to update figure for animation
        def update_frame(i, frames, im, title, all_min, norm_range):
            # Normalize frame
            norm_frame = (frames[i] - all_min) / norm_range
            
            # Update image and title
            im.set_array(norm_frame)
            title.set_text(f'Frame: {i}')
            return [im, title]
            
        # Create animation (with progress bar if requested)
        if show_progress:
            from functools import partial
            
            # Wrap iterator with tqdm for progress bar
            class UpdatingAnimation(animation.FuncAnimation):
                def __init__(self, *args, **kwargs):
                    self.n_frames = len(frames)
                    self.progress_bar = tqdm(total=self.n_frames, desc="Creating video")
                    super().__init__(*args, **kwargs)

                def _step(self, *args):
                    result = super()._step(*args)
                    self.progress_bar.update(1)
                    return result
                
                def finish(self):
                    self.progress_bar.close()
                    
            anim = UpdatingAnimation(
                fig, 
                partial(update_frame, frames=frames, im=im, title=title, 
                        all_min=all_min, norm_range=norm_range),
                frames=len(frames),
                interval=1000/fps,
                blit=True
            )
        else:
            anim = animation.FuncAnimation(
                fig, 
                lambda i: update_frame(i, frames, im, title, all_min, norm_range),
                frames=len(frames),
                interval=1000/fps,
                blit=True
            )
            
        # Set up writer with parameters
        writer_kwargs = {}
        if bitrate:
            writer_kwargs['bitrate'] = bitrate
        if quality:
            writer_kwargs['quality'] = quality / 10.0  # Convert 0-10 to 0-1 range
        
        writer_class = 'ffmpeg' if codec else None
        
        # Override for GIF output
        if output_file.lower().endswith('.gif'):
            writer_class = 'pillow'
            
        writer = animation.FFMpegWriter(
            fps=fps, 
            codec=codec,
            metadata=dict(title="Height Map Animation"),
            **writer_kwargs
        ) if writer_class == 'ffmpeg' else None
            
        # Save the animation
        save_kwargs = {'writer': writer} if writer else {}
        save_kwargs['dpi'] = dpi
        
        # Add additional kwargs
        for key, value in kwargs.items():
            if key not in save_kwargs:
                save_kwargs[key] = value
                
        # Save animation to file
        anim.save(output_file, **save_kwargs)
        
        # Close progress bar if used
        if show_progress and hasattr(anim, 'finish'):
            anim.finish()
        
        # Close figure to release resources
        plt.close(fig)
        
        logger.info(f"Video saved to {output_file}")
        return output_file
    
    except Exception as e:
        logger.error(f"Error exporting to video: {e}")
        import traceback
        traceback.print_exc()
        return None
