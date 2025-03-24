"""
GIF exporter for TMD sequence data.

This module provides functionality for exporting TMD sequence data as animated GIFs.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .base import BaseExporter

# Set up logging
logger = logging.getLogger(__name__)

class GifExporter(BaseExporter):
    """
    GIF exporter for TMD sequence data.
    
    This class provides functionality for exporting TMD sequence data as animated GIFs.
    """
    
    def export_sequence_to_gif(
        self,
        frames_data: List[np.ndarray],
        output_path: str,
        timestamps: Optional[List[Any]] = None,
        fps: int = 1,
        colormap: str = 'viridis',
        add_timestamp: bool = True,
        loop: int = 0,
        **kwargs
    ) -> Optional[str]:
        """
        Export a sequence of frames as an animated GIF.
        
        Args:
            frames_data: List of height map arrays
            output_path: Output path for the GIF file
            timestamps: Optional list of timestamps or labels for each frame
            fps: Frames per second (determines animation speed)
            colormap: Colormap to use for visualization
            add_timestamp: Whether to add timestamps as text on the frames
            loop: Number of times to loop the animation (0 = infinite)
            **kwargs: Additional export options
            
        Returns:
            Path to the exported GIF or None if export failed
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib import cm
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            logger.error("Matplotlib and PIL are required for GIF export")
            return None
        
        # Check if we have data to export
        if not frames_data:
            logger.warning("No frame data provided for GIF export")
            return None
        
        # Use indices as timestamps if none provided
        if timestamps is None:
            timestamps = [f"Frame {i+1}" for i in range(len(frames_data))]
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Create a list to store rendered frames
        gif_frames = []
        
        # Convert each frame to an image
        for i, (frame, timestamp) in enumerate(zip(frames_data, timestamps)):
            # Create matplotlib figure for this frame
            fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 8)))
            
            # Normalize data for consistent color mapping
            vmin = kwargs.get('vmin', np.nanmin(frame))
            vmax = kwargs.get('vmax', np.nanmax(frame))
            
            # Plot the height map
            im = ax.imshow(frame, cmap=colormap, origin='lower', vmin=vmin, vmax=vmax)
            
            # Add colorbar
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(kwargs.get('colorbar_label', 'Height'))
            
            # Add title with timestamp if requested
            if add_timestamp:
                ax.set_title(f"{timestamp}")
            
            # Instead of saving to file, save to memory buffer
            plt.tight_layout()
            
            # Convert matplotlib figure to PIL Image
            fig.canvas.draw()
            image = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
            
            # Add the image to our list of frames
            gif_frames.append(image)
            
            # Close the figure to free memory
            plt.close(fig)
        
        # Calculate duration for each frame in milliseconds
        duration = int(1000 / fps)
        
        # Save as GIF
        try:
            gif_frames[0].save(
                output_path,
                save_all=True,
                append_images=gif_frames[1:],
                duration=duration,
                loop=loop
            )
            logger.info(f"Saved GIF animation to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error saving GIF animation: {e}")
            return None
    
    def export_grouped_sequences_to_gifs(
        self,
        sequences: List[List[np.ndarray]],
        output_path: str,
        sequence_names: Optional[List[str]] = None,
        timestamps: Optional[List[Any]] = None,
        fps: int = 1,
        colormap: str = 'viridis',
        **kwargs
    ) -> Optional[str]:
        """
        Export multiple sequences side by side as a single GIF animation.
        
        Args:
            sequences: List of sequences, where each sequence is a list of frame arrays
            output_path: Output path for the GIF file
            sequence_names: Optional list of names for each sequence
            timestamps: Optional list of timestamps for each frame
            fps: Frames per second (determines animation speed)
            colormap: Colormap to use for visualization
            **kwargs: Additional export options
            
        Returns:
            Path to the exported GIF or None if export failed
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib import cm
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            logger.error("Matplotlib and PIL are required for GIF export")
            return None
        
        # Check if we have data to export
        if not sequences:
            logger.warning("No sequences provided for GIF export")
            return None
        
        # Default sequence names if none provided
        if sequence_names is None:
            sequence_names = [f"Sequence {i+1}" for i in range(len(sequences))]
        
        # Use indices as timestamps if none provided
        if timestamps is None:
            # Find the maximum sequence length
            max_frames = max(len(seq) for seq in sequences)
            timestamps = [f"Frame {i+1}" for i in range(max_frames)]
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Create a list to store rendered frames
        gif_frames = []
        
        # Find common min/max values for consistent color mapping (optional)
        if kwargs.get('normalize_all', True):
            all_values = np.concatenate([np.concatenate(seq) for seq in sequences])
            vmin, vmax = np.nanmin(all_values), np.nanmax(all_values)
        else:
            vmin, vmax = None, None
        
        # For each timestamp, create a frame showing all sequences side by side
        for frame_idx in range(len(timestamps)):
            # Create a figure with subplots for each sequence
            fig, axes = plt.subplots(
                1, len(sequences), 
                figsize=kwargs.get('figsize', (5*len(sequences), 5)),
                squeeze=False
            )
            
            # Plot each sequence at this time point
            for seq_idx, sequence in enumerate(sequences):
                ax = axes[0, seq_idx]
                
                # Only plot if this sequence has this frame
                if frame_idx < len(sequence):
                    frame = sequence[frame_idx]
                    
                    # Get min/max values for this frame if not normalizing across all
                    if vmin is None or vmax is None:
                        frame_vmin = np.nanmin(frame)
                        frame_vmax = np.nanmax(frame)
                    else:
                        frame_vmin, frame_vmax = vmin, vmax
                    
                    # Plot the height map
                    im = ax.imshow(frame, cmap=colormap, origin='lower', 
                                vmin=frame_vmin, vmax=frame_vmax)
                    
                    # Add colorbar
                    cbar = fig.colorbar(im, ax=ax)
                    cbar.set_label('Height')
                
                # Add sequence name as title
                ax.set_title(f"{sequence_names[seq_idx]}")
            
            # Add overall title with timestamp
            fig.suptitle(timestamps[frame_idx], fontsize=16)
            
            plt.tight_layout()
            
            # Convert matplotlib figure to PIL Image
            fig.canvas.draw()
            image = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
            
            # Add the image to our list of frames
            gif_frames.append(image)
            
            # Close the figure to free memory
            plt.close(fig)
        
        # Calculate duration for each frame in milliseconds
        duration = int(1000 / fps)
        
        # Save as GIF
        try:
            gif_frames[0].save(
                output_path,
                save_all=True,
                append_images=gif_frames[1:],
                duration=duration,
                loop=kwargs.get('loop', 0)
            )
            logger.info(f"Saved multi-sequence GIF animation to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error saving multi-sequence GIF animation: {e}")
            return None
    
    def export_normal_maps_to_gif(
        self,
        frames_data: List[np.ndarray],
        output_path: str,
        timestamps: Optional[List[Any]] = None,
        fps: int = 1,
        z_scale: float = 1.0,
        **kwargs
    ) -> Optional[str]:
        """
        Export a sequence of height maps as normal maps in an animated GIF.
        
        Args:
            frames_data: List of height map arrays
            output_path: Output path for the GIF file
            timestamps: Optional list of timestamps or labels for each frame
            fps: Frames per second (determines animation speed)
            z_scale: Z-scale factor for normal map calculation
            **kwargs: Additional export options
            
        Returns:
            Path to the exported GIF or None if export failed
        """
        try:
            import matplotlib.pyplot as plt
            from PIL import Image
        except ImportError:
            logger.error("Matplotlib and PIL are required for GIF export")
            return None
        
        # Check if we have data to export
        if not frames_data:
            logger.warning("No frame data provided for normal map GIF export")
            return None
        
        # Use indices as timestamps if none provided
        if timestamps is None:
            timestamps = [f"Frame {i+1}" for i in range(len(frames_data))]
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Create a list to store rendered frames
        gif_frames = []
        
        # Generate and convert each frame to a normal map
        for i, (frame, timestamp) in enumerate(zip(frames_data, timestamps)):
            # Generate normal map
            normal_map = self._generate_normal_map(frame, z_scale=z_scale)
            
            # Convert normals from [-1,1] to [0,1] for display
            normal_image = (normal_map + 1.0) * 0.5
            
            # Create matplotlib figure for this frame
            fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 8)))
            
            # Plot the normal map
            ax.imshow(normal_image)
            
            # Add title with timestamp
            ax.set_title(f"{timestamp}")
            
            plt.tight_layout()
            
            # Convert matplotlib figure to PIL Image
            fig.canvas.draw()
            image = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
            
            # Add the image to our list of frames
            gif_frames.append(image)
            
            # Close the figure to free memory
            plt.close(fig)
        
        # Calculate duration for each frame in milliseconds
        duration = int(1000 / fps)
        
        # Save as GIF
        try:
            gif_frames[0].save(
                output_path,
                save_all=True,
                append_images=gif_frames[1:],
                duration=duration,
                loop=kwargs.get('loop', 0)
            )
            logger.info(f"Saved normal map GIF animation to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error saving normal map GIF animation: {e}")
            return None
    
    def _generate_normal_map(
        self,
        height_map: np.ndarray,
        z_scale: float = 1.0
    ) -> np.ndarray:
        """
        Generate a normal map from a height map.
        
        Args:
            height_map: 2D array of height values
            z_scale: Z-scale factor for normal calculation
            
        Returns:
            3D array of normal vectors (RGB format)
        """
        # Compute the gradient using numpy
        dy, dx = np.gradient(height_map)
        
        # Scale the gradient
        dx = dx * (1.0 / z_scale)
        dy = dy * (1.0 / z_scale)
        
        # Create normal map array
        normal_map = np.zeros((height_map.shape[0], height_map.shape[1], 3), dtype=np.float32)
        
        # Compute normal vectors
        normal_map[:, :, 0] = -dx
        normal_map[:, :, 1] = -dy
        normal_map[:, :, 2] = 1.0
        
        # Normalize vectors
        norm = np.sqrt(np.sum(normal_map**2, axis=2, keepdims=True))
        normal_map = normal_map / (norm + 1e-10)  # Add small epsilon to avoid division by zero
        
        return normal_map
    
    def export_sequence_differences(
        self,
        frames_data: List[np.ndarray],
        output_dir: str,
        timestamps: Optional[List[Any]] = None,
        format: str = 'png',
        normalize: bool = True,
        colormap: str = 'RdBu',
        **kwargs
    ) -> List[str]:
        """
        Export a sequence of difference frames as images.
        
        Args:
            frames_data: List of difference arrays
            output_dir: Output directory for images
            timestamps: Optional list of timestamps or labels for each frame
            format: Image format (e.g., 'png', 'jpg')
            normalize: Whether to normalize the difference values
            colormap: Colormap to use for visualization
            **kwargs: Additional export options
            
        Returns:
            List of paths to exported images
        """
        # Ensure output directory exists
        self.ensure_output_dir(output_dir)
        
        # Use indices as timestamps if none provided
        if timestamps is None:
            timestamps = [f"diff_{i+1}" for i in range(len(frames_data))]
            
        # Ensure we have enough timestamps
        if len(timestamps) < len(frames_data):
            timestamps = list(timestamps) + [f"diff_{i+1}" for i in range(len(timestamps), len(frames_data))]
        
        output_files = []
        
        # Set up matplotlib for plotting
        try:
            import matplotlib.pyplot as plt
            from matplotlib import cm
        except ImportError:
            logger.error("Matplotlib is required for image export")
            return []
        
        # Export each difference frame as an image
        for i, (frame, timestamp) in enumerate(zip(frames_data, timestamps)):
            # Create a safe filename from the timestamp
            if isinstance(timestamp, str):
                safe_timestamp = self.sanitize_filename(timestamp)
            else:
                safe_timestamp = f"diff_{i+1}"
                
            # Create output filename
            filename = os.path.join(output_dir, f"diff_{safe_timestamp}.{format}")
            
            # Create figure and plot
            fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 8)))
            
            # Plot the difference map
            if normalize:
                # Center around zero with equal positive and negative range
                vmax = max(abs(np.nanmin(frame)), abs(np.nanmax(frame)))
                vmin = -vmax
                im = ax.imshow(frame, cmap=colormap, origin='lower', vmin=vmin, vmax=vmax)
            else:
                im = ax.imshow(frame, cmap=colormap, origin='lower')
            
            # Add colorbar
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(kwargs.get('colorbar_label', 'Difference'))
            
            # Add title
            ax.set_title(f"{timestamp}")
            
            # Save the figure
            plt.savefig(filename, dpi=kwargs.get('dpi', 300), bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"Saved difference image to {filename}")
            output_files.append(filename)
            
        return output_files
    
    def export_normal_maps(
        self,
        frames_data: List[np.ndarray],
        output_dir: str,
        timestamps: Optional[List[Any]] = None,
        format: str = 'png',
        z_scale: float = 1.0,
        **kwargs
    ) -> List[str]:
        """
        Export normal maps from height maps.
        
        Args:
            frames_data: List of height map arrays
            output_dir: Output directory for normal maps
            timestamps: Optional list of timestamps or labels for each frame
            format: Image format (e.g., 'png', 'jpg')
            z_scale: Z-scale factor for normal calculation
            **kwargs: Additional export options
            
        Returns:
            List of paths to exported normal maps
        """
        # Ensure output directory exists
        self.ensure_output_dir(output_dir)
        
        # Use indices as timestamps if none provided
        if timestamps is None:
            timestamps = [f"Frame_{i+1}" for i in range(len(frames_data))]
            
        # Ensure we have enough timestamps
        if len(timestamps) < len(frames_data):
            timestamps = list(timestamps) + [f"Frame_{i+1}" for i in range(len(timestamps), len(frames_data))]
        
        output_files = []
        
        # Try to import required modules
        try:
            from PIL import Image
        except ImportError:
            logger.error("PIL is required for normal map export")
            return []
        
        # Generate and export normal maps
        for i, (frame, timestamp) in enumerate(zip(frames_data, timestamps)):
            # Create a safe filename from the timestamp
            if isinstance(timestamp, str):
                safe_timestamp = self.sanitize_filename(timestamp)
            else:
                safe_timestamp = f"Frame_{i+1}"
                
            # Create output filename
            filename = os.path.join(output_dir, f"normal_{safe_timestamp}.{format}")
            
            # Generate normal map
            normal_map = self._generate_normal_map(frame, z_scale=z_scale)
            
            # Convert normals from [-1,1] to [0,255] for image
            normal_image = ((normal_map + 1.0) * 127.5).astype(np.uint8)
            
            # Save normal map
            Image.fromarray(normal_image).save(filename)
            
            logger.info(f"Saved normal map to {filename}")
            output_files.append(filename)
            
        return output_files


# Convenience functions
def export_sequence_to_gif(
    frames_data: List[np.ndarray],
    output_path: str,
    timestamps: Optional[List[Any]] = None,
    fps: int = 1,
    **kwargs
) -> Optional[str]:
    """
    Convenience function to export a sequence to GIF format.
    
    Args:
        frames_data: List of height map arrays
        output_path: Output path for the GIF file
        timestamps: Optional list of timestamps or labels for each frame
        fps: Frames per second (determines animation speed)
        **kwargs: Additional export options
        
    Returns:
        Path to the exported GIF or None if export failed
    """
    exporter = GifExporter()
    return exporter.export_sequence_to_gif(
        frames_data=frames_data,
        output_path=output_path,
        timestamps=timestamps,
        fps=fps,
        **kwargs
    )

def export_grouped_sequences_to_gifs(
    sequences: List[List[np.ndarray]],
    output_path: str,
    sequence_names: Optional[List[str]] = None,
    timestamps: Optional[List[Any]] = None,
    fps: int = 1,
    **kwargs
) -> Optional[str]:
    """
    Convenience function to export multiple sequences side by side as a GIF.
    
    Args:
        sequences: List of sequences, where each sequence is a list of frame arrays
        output_path: Output path for the GIF file
        sequence_names: Optional list of names for each sequence
        timestamps: Optional list of timestamps for each frame
        fps: Frames per second (determines animation speed)
        **kwargs: Additional export options
        
    Returns:
        Path to the exported GIF or None if export failed
    """
    exporter = GifExporter()
    return exporter.export_grouped_sequences_to_gifs(
        sequences=sequences,
        output_path=output_path,
        sequence_names=sequence_names,
        timestamps=timestamps,
        fps=fps,
        **kwargs
    )

def export_normal_maps_to_gif(
    frames_data: List[np.ndarray],
    output_path: str,
    timestamps: Optional[List[Any]] = None,
    fps: int = 1,
    z_scale: float = 1.0,
    **kwargs
) -> Optional[str]:
    """
    Convenience function to export a sequence of height maps as normal maps in a GIF.
    
    Args:
        frames_data: List of height map arrays
        output_path: Output path for the GIF file
        timestamps: Optional list of timestamps or labels for each frame
        fps: Frames per second (determines animation speed)
        z_scale: Z-scale factor for normal map calculation
        **kwargs: Additional export options
        
    Returns:
        Path to the exported GIF or None if export failed
    """
    exporter = GifExporter()
    return exporter.export_normal_maps_to_gif(
        frames_data=frames_data,
        output_path=output_path,
        timestamps=timestamps,
        fps=fps,
        z_scale=z_scale,
        **kwargs
    )
