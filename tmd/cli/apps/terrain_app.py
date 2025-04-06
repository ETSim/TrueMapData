"""Terrain generation functionality for the CLI."""
import typer
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from ..commands.terrain import generate_synthetic_terrain
from ...image import get_available_map_types

def create_terrain_app() -> typer.Typer:
    """Create the terrain generation app."""
    app = typer.Typer(help="Generate synthetic terrain and textures")
    
    @app.command("generate")
    def generate(
        pattern: str = typer.Argument(
            "waves", 
            help="Pattern type (waves, peak, dome, ramp, combined, flat, random, perlin, fbm)"
        ),
        width: int = typer.Option(1024, "--width", "-w", help="Width of height map"),
        height: int = typer.Option(1024, "--height", "-h", help="Height of height map"),
        output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
        types: Optional[List[str]] = typer.Option(None, "--types", "-t", help="Map types to generate"),
        compress: int = typer.Option(0, "--compress", "-c", help="Compression (0-100)"),
        format: str = typer.Option("png", "--format", "-f", help="Output format"),
        noise: float = typer.Option(0.1, "--noise", "-n", help="Noise level (0-1)"),
        seed: Optional[int] = typer.Option(None, "--seed", "-s", help="Random seed"),
        x_length: Optional[float] = typer.Option(None, "--x-length", help="Physical width in mm"),
        y_length: Optional[float] = typer.Option(None, "--y-length", help="Physical height in mm"),
        mmpp: Optional[float] = typer.Option(None, "--mmpp", help="Millimeters per pixel"),
        wave_height: float = typer.Option(8, "--wave-height", help="Height of wave pattern"),
        wave_scale: float = typer.Option(2.0, "--wave-scale", help="Scale factor for wave pattern"),
        frequency: float = typer.Option(0.3, "--frequency", help="Wave frequency (lower = bigger waves)"),
        phase: float = typer.Option(0.0, "--phase", help="Wave phase shift"),
        smoothing: float = typer.Option(1.0, "--smoothing", help="Wave smoothing factor"),
        contrast: float = typer.Option(1.5, "--contrast", help="Wave contrast enhancement"),
        z_value: float = typer.Option(60, "--z-value", help="Base height value (0-1)")
    ):
        """Generate synthetic terrain patterns."""
        if not generate_synthetic_terrain(
            pattern=pattern,
            width=width,
            height=height,
            output_dir=output_dir,
            types=types,
            compress=compress,
            format=format,
            noise_level=noise,
            seed=seed,
            x_length=x_length,
            y_length=y_length,
            mmpp=mmpp,
            wave_height=wave_height,
            wave_scale=wave_scale,
            frequency=frequency,
            phase=phase,
            smoothing=smoothing,
            contrast=contrast,
            z_value=z_value
        ):
            raise typer.Exit(1)
    
    return app
