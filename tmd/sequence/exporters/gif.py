import importlib.util
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np

from tmd.utils.files import TMDFileUtilities

from ..base import BaseExporter

logger = logging.getLogger(__name__)


def _has_dep(root: str) -> bool:
    return importlib.util.find_spec(root) is not None


def _iter_progress(items: List[np.ndarray], show: bool, desc: str) -> List[np.ndarray]:
    return items


class GifExporter(BaseExporter):
    """
    Exporter for creating animated GIFs from height map sequences.
    """

    def export(self, **kwargs) -> Optional[str]:
        """
        Expects the following kwargs:
            - frames: List of 2D numpy arrays (required)
            - output_file: Destination path for the GIF (defaults to 'output.gif')
            - fps: Frames per second (default: 10.0)
            - colormap: Matplotlib colormap name (default: 'terrain')
            - loop: Loop count for the GIF (default: 0 for infinite)
            - optimize: Whether to optimize the GIF (default: True)
            - duration: Duration per frame in milliseconds (optional)
            - show_progress: Whether to display progress (default: True)
            - Additional kwargs passed to PIL.Image.save
        """
        if not (_has_dep("matplotlib") and _has_dep("PIL")):
            logger.error("Required packages (matplotlib and Pillow) not available")
            return None

        import matplotlib.cm as cm
        from PIL import Image

        # Retrieve parameters
        frames: List[np.ndarray] = kwargs.get("frames", [])
        output_file: str = kwargs.get("output_file", "output.gif")
        fps: float = kwargs.get("fps", 10.0)
        colormap: str = kwargs.get("colormap", "terrain")
        loop: int = kwargs.get("loop", 0)
        optimize: bool = kwargs.get("optimize", True)
        duration: Optional[float] = kwargs.get("duration", None)
        show_progress: bool = kwargs.get("show_progress", True)
        extra_kwargs: Dict[str, Any] = kwargs.get("extra_kwargs", {})

        if not frames:
            logger.error("No frames provided for GIF export")
            return None

        try:
            # Ensure output directory exists
            TMDFileUtilities.ensure_directory_exists(os.path.dirname(os.path.abspath(output_file)))
            if not output_file.lower().endswith(".gif"):
                output_file += ".gif"

            # Calculate duration from fps if not provided
            if duration is None:
                duration = int(1000 / fps)  # in milliseconds

            # Normalize data across all frames for consistent color mapping
            all_min = min(np.nanmin(frame) for frame in frames)
            all_max = max(np.nanmax(frame) for frame in frames)
            norm_range = all_max - all_min if all_max > all_min else 1.0

            cmap = cm.get_cmap(colormap)
            gif_frames = []

            frame_iterator = _iter_progress(frames, show_progress, "Creating GIF")

            for frame in frame_iterator:
                norm_frame = (frame - all_min) / norm_range
                rgba_img = (cmap(norm_frame) * 255).astype(np.uint8)
                gif_frames.append(Image.fromarray(rgba_img))

            if gif_frames:
                gif_frames[0].save(
                    output_file,
                    format="GIF",
                    append_images=gif_frames[1:],
                    save_all=True,
                    duration=duration,
                    loop=loop,
                    optimize=optimize,
                    **extra_kwargs,
                )
                logger.info(f"GIF animation with {len(frames)} frames saved to {output_file}")
                return output_file
            else:
                logger.error("No frames were processed for GIF export")
                return None
        except Exception as e:
            logger.error(f"Error exporting to GIF: {e}")
            return None
