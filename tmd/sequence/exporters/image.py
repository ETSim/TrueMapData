"""
Image exporter for TMD sequence data.

This module provides functionality for exporting TMD sequence data as images.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import tmd.sequence.exporters.npy as np

from .base import BaseExporter

# Set up logging
logger = logging.getLogger(__name__)

class ImageExporter(BaseExporter):
    """
    Image exporter for TMD sequence data.
    
    This class provides functionality for exporting TMD sequence data as images.
    """
    
    def export_images(
        self,
        frames_data: List[np.ndarray],
        output_dir: str,
        timestamps: Optional[List[Any]] = None,
        format: str = 'png',
        colormap: str = 'viridis',
        dpi: int = 300,
        **kwargs
    ) -> List[str]:
        """
        Export a sequence of frames as images.
        
        Args:
            frames_data: List of height map arrays
            output_dir: Output directory for images
            timestamps: Optional list of timestamps or labels for each frame
            format: Image format (e.g., 'png', 'jpg')
            colormap: Colormap to use for visualization
            dpi: Resolution in dots per inch
            **kwargs: Additional export options
            
        Returns:
            List of paths to exported images
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
        
        # Set up matplotlib for plotting
        try:
            import matplotlib.pyplot as plt
            from matplotlib import cm
        except ImportError:
            logger.error("Matplotlib is required for image export")
            return []
        
        # Export each frame as an image
        for i, (frame, timestamp) in enumerate(zip(frames_data, timestamps)):
            # Create a safe filename from the timestamp
            if isinstance(timestamp, str):
                safe_timestamp = self.sanitize_filename(timestamp)
            else:
                safe_timestamp = f"Frame_{i+1}"
                
            # Create output filename
            filename = os.path.join(output_dir, f"{safe_timestamp}.{format}")
            
            # Create figure and plot
            fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 8)))
            
            # Plot the height map
            im = ax.imshow(frame, cmap=colormap, origin='lower')
            
            # Add colorbar
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(kwargs.get('colorbar_label', 'Height'))
            
            # Add title
            ax.set_title(f"{timestamp}")
            
            # Save the figure
            plt.savefig(filename, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"Saved image to {filename}")
            output_files.append(filename)
            
        return output_files
    
    # Alias for backward compatibility
    export_sequence = export_images
    
    def export_sequence_differences(
        self,
        frames_data: List[np.ndarray],
        output_dir: str,
        timestamps: Optional[List[Any]] = None,
        format: str = 'png',
        normalize: bool = True,
        colormap: str = 'RdBu',
        dpi: int = 300,
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
            dpi: Resolution in dots per inch
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
            plt.savefig(filename, dpi=dpi, bbox_inches='tight')
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

def export_sequence_to_images(
    frames: List[np.ndarray],
    output_directory: str,
    filename_pattern: str = "frame_{:04d}.png",
    colormap: str = "terrain",
    dpi: int = 100,
    format: str = None,
    show_progress: bool = True,
    **kwargs
) -> Optional[List[str]]:
    """
    Export a sequence of height maps as individual image files.
    
    Args:
        frames: List of 2D numpy arrays representing height maps
        output_directory: Directory to save the images
        filename_pattern: Pattern for naming files with frame number placeholder
        colormap: Matplotlib colormap name for rendering
        dpi: Resolution for rendered images
        format: Image format (png, jpg, etc.) - overrides extension in pattern
        show_progress: Whether to show a progress bar
        **kwargs: Additional arguments passed to matplotlib's savefig
        
    Returns:
        List of paths to created files or None if failed
    """
    try:
        # Check for necessary libraries
        try:
            import matplotlib.pyplot as plt
            from matplotlib import cm
            from tqdm import tqdm
        except ImportError as e:
            logger.error(f"Required package not found: {e}")
            logger.error("Please install matplotlib and tqdm packages")
            return None
            
        # Check frames
        if not frames or len(frames) == 0:
            logger.error("No frames provided for image sequence export")
            return None
        
        # Ensure output directory exists
        os.makedirs(output_directory, exist_ok=True)
        
        # Normalize data collectively for consistent color mapping
        all_min = min(np.min(frame) for frame in frames)
        all_max = max(np.max(frame) for frame in frames)
        norm_range = all_max - all_min
        
        if norm_range <= 0:
            norm_range = 1.0  # Avoid division by zero
        
        # Create colormap
        cmap = cm.get_cmap(colormap)
        
        # Process each frame (with progress bar if requested)
        output_files = []
        frame_iterator = tqdm(enumerate(frames), total=len(frames), desc="Creating images") if show_progress else enumerate(frames)
        
        for i, frame in frame_iterator:
            # Create output filename
            filename = filename_pattern.format(i)
            
            # Override format if specified
            if format:
                base, _ = os.path.splitext(filename)
                filename = f"{base}.{format}"
                
            output_path = os.path.join(output_directory, filename)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Normalize frame
            norm_frame = (frame - all_min) / norm_range
            
            # Display as image
            im = ax.imshow(norm_frame, cmap=cmap)
            ax.axis('off')  # Remove axes
            
            # Add colorbar
            plt.colorbar(im, ax=ax)
            
            # Add frame number as title
            ax.set_title(f"Frame {i}")
            
            # Save image
            fig.savefig(output_path, dpi=dpi, bbox_inches='tight', **kwargs)
            
            # Close figure to release memory
            plt.close(fig)
            
            output_files.append(output_path)
        
        logger.info(f"Image sequence saved to {output_directory}")
        return output_files
        
    except Exception as e:
        logger.error(f"Error exporting to image sequence: {e}")
        import traceback
        traceback.print_exc()
        return None
