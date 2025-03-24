""".

Video exporter implementation for TMD sequence data.

This module provides functionality for exporting TMD sequence data as video files
using popular video codecs and container formats.
"""

import os
import logging
import numpy as np
import tempfile
from typing import List, Dict, Any, Optional, Union

from .base import BaseExporter

# Set up logging
logger = logging.getLogger(__name__)

class VideoExporter(BaseExporter):
    """Video exporter for TMD sequence data.."""
    
    def export_sequence(
        self,
        frames_data: List[np.ndarray],
        output_file: str,
        timestamps: Optional[List[Any]] = None,
        format: str = 'mp4',
        fps: int = 30,
        codec: Optional[str] = None,
        colormap: str = 'viridis',
        dpi: int = 150,
        **kwargs
    ) -> str:
        """.

        Export sequence frames as a video file.
        
        Args:
            frames_data: List of heightmap arrays
            output_file: Output video file path
            timestamps: Optional list of timestamps for each frame
            format: Video format ('mp4', 'avi', 'mov', etc.)
            fps: Frames per second
            codec: Video codec (None for default)
            colormap: Colormap for visualization
            dpi: Resolution in dots per inch
            **kwargs: Additional options
            
        Returns:
            Path to saved video file
        """
        # Ensure output directory exists
        output_dir = os.path.dirname(os.path.abspath(output_file))
        self.ensure_output_dir(output_dir)
        
        # Ensure correct extension
        if not output_file.lower().endswith(f'.{format.lower()}'):
            output_file = f"{os.path.splitext(output_file)[0]}.{format.lower()}"
        
        # Try to use MoviePy first, fall back to matplotlib animation
        if self._has_moviepy():
            return self._export_with_moviepy(
                frames_data=frames_data,
                output_file=output_file,
                fps=fps,
                codec=codec,
                colormap=colormap,
                **kwargs
            )
        elif self._has_matplotlib():
            return self._export_with_matplotlib(
                frames_data=frames_data,
                output_file=output_file,
                fps=fps,
                dpi=dpi,
                colormap=colormap,
                **kwargs
            )
        else:
            logger.error("No suitable video export library found. Install moviepy or matplotlib.")
            return ""
    
    def _has_moviepy(self) -> bool:
        """Check if MoviePy is available.."""
        try:
            import moviepy.editor
            return True
        except ImportError:
            return False
    
    def _has_matplotlib(self) -> bool:
        """Check if Matplotlib is available.."""
        try:
            import matplotlib
            import matplotlib.animation
            return True
        except ImportError:
            return False
    
    def _export_with_moviepy(
        self,
        frames_data: List[np.ndarray],
        output_file: str,
        fps: int = 30,
        codec: Optional[str] = None,
        colormap: str = 'viridis',
        **kwargs
    ) -> str:
        """.

        Export video using MoviePy library.
        
        Args:
            frames_data: List of heightmap arrays
            output_file: Output video file path
            fps: Frames per second
            codec: Video codec
            colormap: Colormap for visualization
            **kwargs: Additional options
            
        Returns:
            Path to saved video file
        """
        try:
            import moviepy.editor as mpy
            from matplotlib import cm
            
            # Process frames to make them suitable for video
            processed_frames = []
            
            for frame in frames_data:
                # Normalize the frame
                if np.max(frame) > 1.0 or np.min(frame) < 0.0:
                    frame_norm = (frame - np.min(frame)) / (np.max(frame) - np.min(frame))
                else:
                    frame_norm = frame.copy()
                
                # Apply colormap
                cmap = cm.get_cmap(colormap)
                colored_frame = cmap(frame_norm)
                
                # Convert to 8-bit RGB
                rgb_frame = (colored_frame[:, :, :3] * 255).astype(np.uint8)
                
                processed_frames.append(rgb_frame)
            
            # Create clip from frames
            clip = mpy.ImageSequenceClip(processed_frames, fps=fps)
            
            # Add audio if provided
            if 'audio' in kwargs:
                audio_file = kwargs.get('audio')
                if os.path.exists(audio_file):
                    audio_clip = mpy.AudioFileClip(audio_file)
                    clip = clip.set_audio(audio_clip)
            
            # Set codec parameters
            if codec:
                codec_params = {'codec': codec}
            else:
                codec_params = {}
                
            # Get additional parameters for specific formats
            if output_file.lower().endswith('.mp4'):
                codec_params.update({'codec': 'libx264', 'preset': 'medium'})
            elif output_file.lower().endswith('.webm'):
                codec_params.update({'codec': 'libvpx'})
            elif output_file.lower().endswith('.gif'):
                # Special handling for GIF
                clip.write_gif(output_file, fps=fps, opt='nq')
                logger.info(f"Saved video to {output_file}")
                return output_file
            
            # Write the clip to file
            clip.write_videofile(
                output_file,
                fps=fps,
                **codec_params
            )
            
            logger.info(f"Saved video to {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error exporting video with MoviePy: {e}")
            return ""
    
    def _export_with_matplotlib(
        self,
        frames_data: List[np.ndarray],
        output_file: str,
        fps: int = 30,
        dpi: int = 150,
        colormap: str = 'viridis',
        **kwargs
    ) -> str:
        """.

        Export video using Matplotlib animation.
        
        Args:
            frames_data: List of heightmap arrays
            output_file: Output video file path
            fps: Frames per second
            dpi: Resolution in dots per inch
            colormap: Colormap for visualization
            **kwargs: Additional options
            
        Returns:
            Path to saved video file
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.animation as animation
            
            # Create figure and axis
            fig, ax = plt.subplots(figsize=(10, 8))
            plt.tight_layout()
            
            # Initial display
            im = ax.imshow(frames_data[0], cmap=colormap)
            plt.colorbar(im, ax=ax)
            
            # Remove axes
            ax.set_axis_off()
            
            # Animation function
            def update(frame_idx):
                im.set_array(frames_data[frame_idx])
                return [im]
            
            # Create animation
            ani = animation.FuncAnimation(
                fig,
                update,
                frames=len(frames_data),
                interval=1000/fps,
                blit=True
            )
            
            # Determine writer
            if output_file.lower().endswith('.mp4'):
                writer = 'ffmpeg'
            elif output_file.lower().endswith('.gif'):
                writer = 'pillow'
            else:
                writer = 'ffmpeg'
            
            # Ensure output directory exists
            try:
                os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            except (PermissionError, OSError) as e:
                logger.error(f"Error creating directory for {output_file}: {e}")
                return ""
            
            # Save animation
            ani.save(
                output_file,
                writer=writer,
                fps=fps,
                dpi=dpi
            )
            
            # Close figure
            plt.close(fig)
            
            logger.info(f"Saved video to {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error exporting video with Matplotlib: {e}")
            return ""
    
    def export_sequence_with_overlay(
        self,
        frames_data: List[np.ndarray],
        output_file: str,
        overlay_text: List[str],
        fps: int = 30,
        font_size: int = 24,
        position: str = 'top',
        **kwargs
    ) -> str:
        """.

        Export sequence as video with text overlay.
        
        Args:
            frames_data: List of heightmap arrays
            output_file: Output video file path
            overlay_text: List of text to overlay on each frame
            fps: Frames per second
            font_size: Size of overlay text
            position: Position of text ('top', 'bottom', 'top-left', etc.)
            **kwargs: Additional options
            
        Returns:
            Path to saved video file
        """
        if not self._has_moviepy():
            logger.error("MoviePy is required for overlay functionality")
            return ""
        
        try:
            import moviepy.editor as mpy
            from matplotlib import cm
            import numpy as np
            from PIL import Image, ImageDraw, ImageFont
            
            # Process frames with overlay
            processed_frames = []
            
            # Get colormap
            colormap = kwargs.get('colormap', 'viridis')
            cmap = cm.get_cmap(colormap)
            
            # Determine text position
            pos_map = {
                'top': ('center', 30),
                'bottom': ('center', 'bottom'),
                'top-left': (30, 30),
                'top-right': ('right', 30),
                'bottom-left': (30, 'bottom'),
                'bottom-right': ('right', 'bottom')
            }
            pos = pos_map.get(position, (30, 30))
            
            # Process each frame
            for i, frame in enumerate(frames_data):
                # Normalize the frame
                if np.max(frame) > 1.0 or np.min(frame) < 0.0:
                    frame_norm = (frame - np.min(frame)) / (np.max(frame) - np.min(frame))
                else:
                    frame_norm = frame.copy()
                
                # Apply colormap
                colored_frame = cmap(frame_norm)
                
                # Convert to 8-bit RGB
                rgb_frame = (colored_frame[:, :, :3] * 255).astype(np.uint8)
                
                # Create PIL image
                img = Image.fromarray(rgb_frame)
                draw = ImageDraw.Draw(img)
                
                # Try to load a standard font
                try:
                    # Try some common fonts
                    font_options = [
                        "arial.ttf", "Arial.ttf",
                        "DejaVuSans.ttf", "DejaVuSans-Bold.ttf",
                        "times.ttf", "Times.ttf"
                    ]
                    
                    font = None
                    for font_name in font_options:
                        try:
                            font = ImageFont.truetype(font_name, font_size)
                            break
                        except IOError:
                            continue
                    
                    if font is None:
                        # Fallback to default
                        font = ImageFont.load_default()
                except Exception:
                    font = ImageFont.load_default()
                
                # Get text for this frame
                if i < len(overlay_text):
                    text = overlay_text[i]
                else:
                    text = ""
                
                # Calculate text position
                if pos[0] == 'center':
                    x = img.width // 2 - font.getsize(text)[0] // 2
                elif pos[0] == 'right':
                    x = img.width - font.getsize(text)[0] - 30
                else:
                    x = pos[0]
                
                if pos[1] == 'bottom':
                    y = img.height - font.getsize(text)[1] - 30
                else:
                    y = pos[1]
                
                # Draw text with shadow (for better visibility)
                draw.text((x+2, y+2), text, fill=(0, 0, 0), font=font)
                draw.text((x, y), text, fill=(255, 255, 255), font=font)
                
                # Convert back to numpy array
                frame_with_text = np.array(img)
                processed_frames.append(frame_with_text)
            
            # Create clip from frames
            clip = mpy.ImageSequenceClip(processed_frames, fps=fps)
            
            # Get codec parameters from kwargs
            codec = kwargs.get('codec', None)
            if codec:
                codec_params = {'codec': codec}
            else:
                codec_params = {}
            
            # Write the clip to file
            format = os.path.splitext(output_file)[1].lower().lstrip('.')
            if format == 'gif':
                clip.write_gif(output_file, fps=fps, opt='nq')
            else:
                clip.write_videofile(
                    output_file,
                    fps=fps,
                    **codec_params
                )
            
            logger.info(f"Saved video with overlay to {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error exporting video with overlay: {e}")
            return ""
