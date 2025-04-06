"""Terrain generation commands."""
from pathlib import Path
from typing import Optional, List
import time
from rich.console import Console
from rich.table import Table

from ..core.ui import print_error, print_success
from ...surface.terrain import TMDTerrain
from .export import export_maps_command

console = Console()

def generate_synthetic_terrain(
    pattern: str,
    width: int,
    height: int,
    output_dir: Optional[Path] = None,
    types: Optional[List[str]] = None,
    compress: int = 0,
    format: str = "png",
    noise_level: float = 0.0,
    seed: Optional[int] = None,
    x_length: Optional[float] = None,
    y_length: Optional[float] = None,
    mmpp: Optional[float] = None,
    wave_height: float = 1.0,
    z_value: float = 0.5,  # Add default z_value
    **kwargs
) -> bool:
    """Generate synthetic terrain and export maps."""
    try:
        # Set up output directories
        output_dir = output_dir or Path("terrain")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Show generation configuration
        config_table = Table(title="Terrain Generation Config")
        config_table.add_column("Setting", style="cyan")
        config_table.add_column("Value", style="green")
        config_table.add_row("Pattern", pattern)
        config_table.add_row("Size", f"{width}x{height}")
        config_table.add_row("Output Dir", str(output_dir))
        
        if x_length and y_length:
            config_table.add_row("Physical Size", f"{x_length:.2f}mm × {y_length:.2f}mm")
        if mmpp:
            config_table.add_row("Resolution", f"{mmpp:.6f} mm/pixel")
            
        console.print(config_table)
        
        # Generate terrain
        tmd_path = output_dir / f"{pattern}.tmd"
        
        TMDTerrain.generate_synthetic_tmd(
            str(tmd_path),
            width=width,
            height=height,
            pattern=pattern,
            noise_level=noise_level,
            seed=seed,
            x_length=x_length,
            y_length=y_length,
            mmpp=mmpp,
            wave_height=wave_height,
            z_value=z_value,  # Pass z_value
            **kwargs
        )
        
        # Export maps if requested
        if types is not None or types is None:  # Export all if None
            maps_dir = output_dir / "maps"
            export_maps_command(
                tmd_path,
                maps_dir,
                types=types,
                compress=compress,
                format=format
            )
            
        return True
        
    except Exception as e:
        print_error(f"Failed to generate terrain: {e}")
        return False
