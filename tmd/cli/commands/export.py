"""Export commands for TMD CLI."""
from pathlib import Path
from typing import Optional, List

from ..core.ui import console, print_error
from ...image import MapExporter, get_available_map_types
from ...model.formats import get_available_formats
from ...model.factory import ModelExporterFactory
from ...model.config import ExportConfig

def export_command(
    tmd_file: Path,
    output: Optional[Path] = None,
    format: str = "npz",
    **kwargs
) -> bool:
    """Export a TMD file to another format."""
    try:
        if format in get_available_map_types():
            MapExporter.export(tmd_file, output, format, **kwargs)
        elif format in get_available_formats():
            ModelExporter.export(tmd_file, output, format, **kwargs)
        else:
            print_error(f"Unknown export format: {format}")
            return False
        return True
    except Exception as e:
        print_error(f"Export failed: {e}")
        return False

def export_maps_command(
    tmd_file: Path,
    output_dir: Optional[Path] = None,
    types: Optional[List[str]] = None,
    **kwargs
) -> bool:
    """Export a TMD file to multiple map types."""
    try:
        if types is None:
            types = get_available_map_types()
            
        for map_type in types:
            output = output_dir / f"{tmd_file.stem}_{map_type}.png" if output_dir else None
            MapExporter.export_map(tmd_file, output, map_type, **kwargs)
        return True
    except Exception as e:
        print_error(f"Map export failed: {e}")
        return False
