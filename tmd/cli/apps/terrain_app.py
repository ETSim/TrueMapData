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
        z_value: float = typer.Option(60, "--z-value", help="Base height value (0-1)"),
        scale: float = typer.Option(100.0, "--scale", help="Scale of terrain features"),
        octaves: int = typer.Option(8, "--octaves", help="Number of noise octaves"),
        persistence: float = typer.Option(0.5, "--persistence", help="Persistence between octaves"),
        lacunarity: float = typer.Option(2.0, "--lacunarity", help="Lacunarity between octaves"),
        ridge_factor: float = typer.Option(1.2, "--ridge-factor", help="Ridge formation factor"),
        plateau_threshold: float = typer.Option(0.7, "--plateau-threshold", help="Height threshold for plateaus"),
        enhance: bool = typer.Option(True, "--enhance/--no-enhance", help="Enhance terrain features"),
        plateaus: bool = typer.Option(True, "--plateaus/--no-plateaus", help="Add plateau effect"),
        power: float = typer.Option(1.2, "--power", help="Terrain enhancement power"),
        base_frequency: float = typer.Option(1.0, "--base-frequency", help="Base frequency for fBm"),
        ridge_weight: float = typer.Option(0.3, "--ridge-weight", help="Ridge formation weight"),
        river_valleys: bool = typer.Option(True, "--river-valleys/--no-river-valleys", help="Enable river valley formation"),
        valley_threshold: float = typer.Option(0.3, "--valley-threshold", help="Height threshold for valley formation"),
        valley_depth: float = typer.Option(0.4, "--valley-depth", help="Relative depth of valleys"),
        mountain_peaks: bool = typer.Option(True, "--mountain-peaks/--no-mountain-peaks", help="Enable mountain peak enhancement"),
        peak_threshold: float = typer.Option(0.7, "--peak-threshold", help="Height threshold for peak enhancement"),
        peak_factor: float = typer.Option(1.5, "--peak-factor", help="Mountain peak enhancement factor")
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
            z_value=z_value,
            scale=scale,
            octaves=octaves,
            persistence=persistence,
            lacunarity=lacunarity,
            ridge_factor=ridge_factor,
            plateau_threshold=plateau_threshold,
            enhance=enhance,
            plateaus=plateaus,
            power=power,
            base_frequency=base_frequency,
            ridge_weight=ridge_weight,
            river_valleys=river_valleys,
            valley_threshold=valley_threshold,
            valley_depth=valley_depth,
            mountain_peaks=mountain_peaks,
            peak_threshold=peak_threshold,
            peak_factor=peak_factor
        ):
            raise typer.Exit(1)
    
    return app
