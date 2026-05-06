"""
Map exporter functionality.

This module provides the main exporter class for generating and exporting
maps from height maps.
"""
import logging
import math
import os
from typing import Optional

import numpy as np

from ..core.image_utils import save_image
from ..core.exceptions import MapGeneratorNotFoundError
from .registry import MapRegistry

logger = logging.getLogger(__name__)


def _cli_console():
    """Import Rich console lazily to avoid tmd.image <-> tmd.cli circular imports."""
    from ...cli.core.ui import console

    return console

class MapExporter:
    """
    Exporter for generating and saving maps from height maps.
    
    This class provides a unified interface for generating different types of maps
    from height maps and saving them to files.
    """
    
    @staticmethod
    def export_map(
        height_map: np.ndarray,
        output_file: str,
        map_type: str,
        **kwargs
    ) -> Optional[str]:
        """
        Generate and export a map from a height map.
        
        Args:
            height_map: Input height map
            output_file: Path to save the output file
            map_type: Type of map to generate
            **kwargs: Additional parameters for generation and saving
            
        Returns:
            Path to the saved file, or None if failed
            
        Raises:
            MapGeneratorNotFoundError: If no generator is found for the specified map type
        """
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        try:
            # Extract parameters from kwargs
            compress = kwargs.pop('compress', 75) 
            format = kwargs.pop('format', 'png')
            bit_depth = kwargs.pop('bit_depth', 8)
            colormap = kwargs.pop('colormap', None)
            normalize = kwargs.pop('normalize', True)
            
            # Get original metadata - ensure we don't pass it twice
            metadata = kwargs.pop('metadata', None)
            
            # Get generator class
            generator_cls = MapRegistry.get(map_type)
            if not generator_cls:
                raise MapGeneratorNotFoundError(f"No generator found for map type: {map_type}")
            
            # Create generator instance
            generator = generator_cls()
            
            # Generate the map with all remaining parameters
            map_data = generator.generate(height_map, metadata=metadata, **kwargs)
            
            # Save the map with compression
            saved_path = save_image(
                map_data,
                output_file,
                bit_depth=bit_depth,
                colormap=colormap,
                normalize=normalize,
                compress=compress,
                format=format
            )
            
            if saved_path:
                # Get file size and format
                size_kb = os.path.getsize(saved_path) / 1024
                _cli_console().print(
                    f"[green]Saved {map_type} map ({size_kb:.1f} KB) with {compress}% compression[/]"
                )
            
            return saved_path
            
        except Exception as e:
            logger.error(f"Failed to export {map_type} map: {e}")
            _cli_console().print(f"[red]Error exporting {map_type} map: {e}[/]")
            import traceback
            traceback.print_exc()
            return None

    @staticmethod
    def export_material_binding_maps(
        height_map: np.ndarray,
        output_dir: str,
        stem: str,
        *,
        compress: int = 75,
        normalize: bool = True,
        metadata=None,
    ) -> dict[str, str]:
        """Export canonical material-binding maps and return slot->path mapping."""
        slot_types = {
            "map_kd": "height",
            "map_bump": "normal",
            "map_disp": "displacement",
            "map_pr": "roughness",
        }
        out: dict[str, str] = {}
        for slot, map_type in slot_types.items():
            path = os.path.join(output_dir, f"{stem}_{map_type}.png")
            saved = MapExporter.export_map(
                height_map,
                path,
                map_type,
                compress=compress,
                format="png",
                normalize=normalize,
                metadata=metadata,
            )
            if saved:
                out[slot] = saved
        return out

    @staticmethod
    def export_material_binding_maps_with_physical_tiling(
        height_map: np.ndarray,
        output_dir: str,
        stem: str,
        *,
        tile_size_px: tuple[int, int],
        target_size_px: tuple[int, int],
        compress: int = 75,
        normalize: bool = True,
        metadata=None,
    ) -> dict[str, str]:
        """Export canonical binding maps using physical tile and atlas sizes."""
        slot_types = {
            "map_kd": "height",
            "map_bump": "normal",
            "map_disp": "displacement",
            "map_pr": "roughness",
        }

        tile_w, tile_h = tile_size_px
        target_w, target_h = target_size_px
        if tile_w <= 0 or tile_h <= 0:
            raise ValueError(f"tile_size_px must be positive, got {tile_size_px}")
        if target_w <= 0 or target_h <= 0:
            raise ValueError(f"target_size_px must be positive, got {target_size_px}")

        os.makedirs(output_dir, exist_ok=True)
        out: dict[str, str] = {}
        for slot, map_type in slot_types.items():
            generator_cls = MapRegistry.get(map_type)
            if not generator_cls:
                raise MapGeneratorNotFoundError(f"No generator found for map type: {map_type}")

            generator = generator_cls()
            map_data = generator.generate(height_map, metadata=metadata)
            tiled = MapExporter._resample_tile_and_crop(map_data, tile_w, tile_h, target_w, target_h)

            path = os.path.join(output_dir, f"{stem}_{map_type}.png")
            saved = save_image(
                tiled,
                path,
                bit_depth=8,
                colormap=None,
                normalize=normalize,
                compress=compress,
                format="png",
            )
            if saved:
                out[slot] = saved
        return out

    @staticmethod
    def _resample_tile_and_crop(
        map_data: np.ndarray,
        tile_w: int,
        tile_h: int,
        target_w: int,
        target_h: int,
    ) -> np.ndarray:
        """Resample one-capture map to tile size, then tile and crop atlas."""
        from PIL import Image

        def _to_pil(array: np.ndarray) -> Image.Image:
            if array.ndim == 2:
                if np.issubdtype(array.dtype, np.floating):
                    arr = np.clip(array, 0.0, 1.0)
                    return Image.fromarray((arr * 255).astype(np.uint8), mode="L")
                return Image.fromarray(array.astype(np.uint8), mode="L")
            if array.ndim == 3 and array.shape[2] in (3, 4):
                if np.issubdtype(array.dtype, np.floating):
                    arr = np.clip(array, 0.0, 1.0)
                    arr = (arr * 255).astype(np.uint8)
                else:
                    arr = array.astype(np.uint8)
                mode = "RGB" if array.shape[2] == 3 else "RGBA"
                return Image.fromarray(arr, mode=mode)
            raise ValueError(f"Unsupported map shape for physical tiling: {array.shape}")

        def _from_pil(img: Image.Image, reference_dtype: np.dtype) -> np.ndarray:
            arr = np.array(img)
            if np.issubdtype(reference_dtype, np.floating):
                return (arr.astype(np.float32) / 255.0).astype(reference_dtype)
            return arr.astype(reference_dtype)

        tile_img = _to_pil(map_data).resize((tile_w, tile_h), Image.LANCZOS)
        tile = _from_pil(tile_img, map_data.dtype)
        reps_x = int(math.ceil(target_w / tile_w))
        reps_y = int(math.ceil(target_h / tile_h))
        tiled = np.tile(tile, (reps_y, reps_x) + (() if tile.ndim == 2 else (1,)))
        return tiled[:target_h, :target_w] if tile.ndim == 2 else tiled[:target_h, :target_w, :]
