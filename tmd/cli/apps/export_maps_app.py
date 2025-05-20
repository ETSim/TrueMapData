import typer
import json
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

from tmd import TMD
from tmd.cli.core.ui import console
from ..commands.export import export_maps_command
from ..commands.terrain import generate_synthetic_terrain
from ...image import get_available_map_types
from ...surface.metadata import create_metadata_file


def create_export_maps_app() -> typer.Typer:
    """Create the maps export app."""
    app = typer.Typer(help="Export TMD files to various map types")

    def parse_metadata(md_str: Optional[str]) -> Optional[Dict[str, Any]]:
        """Turn a JSON string into a dict, or None if empty."""
        if not md_str:
            return None
        try:
            return json.loads(md_str)
        except json.JSONDecodeError as e:
            console.print(f"[red]Invalid metadata JSON:[/] {e}")
            raise typer.Exit(1)
            
    def parse_tuple_range(range_str: Optional[str]) -> Optional[Tuple[float, float]]:
        """Parse a string like '30:45' into a tuple of (30.0, 45.0)."""
        if not range_str:
            return None
        try:
            min_val, max_val = map(float, range_str.split(':'))
            return (min_val, max_val)
        except ValueError:
            console.print(f"[red]Invalid range format (use min:max, e.g. '30:45'): {range_str}[/]")
            raise typer.Exit(1)

    @app.command("batch")
    def batch_export(
        input_dir: Path = typer.Argument(Path("data"), help="Input directory containing TMD files"),
        output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory (default: ./textures)"),
        types: Optional[List[str]] = typer.Option(None, "--types", "-t", help="Map types (default: all)"),
        recursive: bool = typer.Option(False, "--recursive", "-r", help="Search subdirectories"),
        compress: int = typer.Option(0, "--compress", "-c", help="Compression (0-100)"),
        format: str = typer.Option("png", "--format", "-f", help="Output format"),
        pattern: str = typer.Option("*.tmd", "--pattern", "-p", help="File pattern to match"),
        normalize: bool = typer.Option(False, "--normalize", "-n", help="Normalize height map before processing"),
        metadata: Optional[str] = typer.Option(None, "--metadata", "-m", help="Additional parameters as JSON string"),
        save_metadata: bool = typer.Option(False, "--save-metadata", help="Save comprehensive metadata to JSON")
    ):
        """Export maps for multiple TMD files in a directory."""
        cli_md = parse_metadata(metadata)

        if output_dir is None:
            output_dir = Path("textures")
        files = list(input_dir.rglob(pattern) if recursive else input_dir.glob(pattern))
        if not files:
            console.print(f"[yellow]No TMD files found in {input_dir}[/]")
            raise typer.Exit(1)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Use context manager for progress bar
        with typer.progressbar(files, label="Processing files") as progress:
            for file in progress:
                console.print(f"\n[cyan]Processing {file.name}…[/]")
                file_output_dir = output_dir / file.stem
                file_output_dir.mkdir(parents=True, exist_ok=True)

                # Load TMD to get built-in metadata
                tmd_data = TMD.load(str(file))
                file_md = tmd_data.metadata or {}

                # Optionally save out full metadata JSON
                if save_metadata:
                    try:
                        metadata_path = create_metadata_file(tmd_data, file, file_output_dir)
                        console.print(f"[green]Metadata saved to {metadata_path}[/]")
                    except Exception as e:
                        console.print(f"[red]Error saving metadata:[/] {e}")

                # Merge CLI overrides with file metadata
                merged_md = {**file_md, **(cli_md or {})}

                export_maps_command(
                    file,
                    file_output_dir,
                    types,
                    compress=compress,
                    format=format,
                    normalize=normalize,  # Added normalize parameter
                    metadata=merged_md
                )

    # Apply same metadata handling for single-map commands
    @app.command("normal")
    def normal(
        input_file: Path = typer.Argument(..., help="Input TMD file", exists=True),
        output_file: Optional[Path] = typer.Option(None, "--output", "-o", help="Output filename"),
        strength: float = typer.Option(1.0, "--strength", "-s", help="Normal map strength"),
        compress: int = typer.Option(0, "--compress", "-c", help="Compression (0-100)"),
        format: str = typer.Option("png", "--format", "-f", help="Output format"),
        normalize: bool = typer.Option(False, "--normalize", "-n", help="Normalize height map before processing"),
        metadata: Optional[str] = typer.Option(None, "--metadata", "-m", help="Additional parameters as JSON string")
    ):
        """Export a normal map."""
        cli_md = parse_metadata(metadata)
        # Load & merge metadata
        tmd_data = TMD.load(str(input_file))
        file_md = tmd_data.metadata or {}
        merged_md = {**file_md, **(cli_md or {})}

        out_dir = output_file.parent if output_file else None
        export_maps_command(
            input_file,
            out_dir,
            ["normal"],
            strength=strength,
            compress=compress,
            format=format,
            normalize=normalize,
            metadata=merged_md
        )

    @app.command("list")
    def list_maps():
        """List all available map types with descriptions."""
        console.print("\n[bold cyan]Available Map Types:[/]")
        
        map_info = [
            ("normal", "Normal maps for surface detail"),
            ("height", "Height maps for displacement"),
            ("ao", "Ambient occlusion maps for depth cues"),
            ("parallax_ao", "Enhanced parallax ambient occlusion with slope awareness"),
            ("bump", "Bump maps for simple bumps and dents"),
            ("roughness", "Roughness maps for PBR materials"),
            ("metallic", "Metallic maps for PBR materials"),
            ("displacement", "Displacement maps for mesh displacement"),
            ("hillshade", "Hillshade maps for terrain visualization"),
            ("curvature", "Enhanced curvature maps with multiple visualization modes"),
            ("angle", "Enhanced angle/slope maps with multiple visualization modes"),
            ("depth", "Depth maps showing distance from viewpoint to surfaces"),
        ]
        
        table = typer.rich.table.Table()
        table.add_column("Type", style="cyan")
        table.add_column("Description", style="green")
        
        for map_type, description in map_info:
            table.add_row(map_type, description)
            
        console.print(table)
        console.print("\nUse 'tmd maps <map_type> --help' for detailed options for each map type.")

    @app.command("ao")
    def ao(
        input_file: Path = typer.Argument(..., help="Input TMD file", exists=True),
        output_file: Optional[Path] = typer.Option(None, help="Output filename"),
        samples: int = typer.Option(16, help="Number of AO samples"),
        strength: float = typer.Option(1.0, help="AO effect strength"),
        compress: int = typer.Option(0, help="Compression level (0-100)", min=0, max=100),
        format: str = typer.Option("png", help="Output format (png, jpg, webp)"),
        normalize: bool = typer.Option(False, "--normalize", "-n", help="Normalize height map before processing"),
        metadata: Optional[str] = typer.Option(
            None, "--metadata", "-m", help="Additional parameters as JSON string")
    ):
        """Export an ambient occlusion map."""
        output_dir = Path(output_file).parent if output_file else input_file.parent
        export_maps_command(input_file, output_dir, ["ao"], samples=samples, strength=strength,
                            compress=compress, format=format, normalize=normalize, metadata=metadata)

    @app.command("bump")
    def bump(
        input_file: Path = typer.Argument(..., help="Input TMD file", exists=True),
        output_file: Optional[Path] = typer.Option(None, help="Output filename"),
        strength: float = typer.Option(1.0, help="Bump map strength"),
        compress: int = typer.Option(0, help="Compression level (0-100)", min=0, max=100),
        format: str = typer.Option("png", help="Output format (png, jpg, webp)"),
        normalize: bool = typer.Option(False, "--normalize", "-n", help="Normalize height map before processing"),
        metadata: Optional[str] = typer.Option(
            None, "--metadata", "-m", help="Additional parameters as JSON string")
    ):
        """Export a bump map."""
        output_dir = Path(output_file).parent if output_file else input_file.parent
        export_maps_command(input_file, output_dir, ["bump"], strength=strength,
                            compress=compress, format=format, normalize=normalize, metadata=metadata)

    @app.command("roughness")
    def roughness(
        input_file: Path = typer.Argument(..., help="Input TMD file", exists=True),
        output_file: Optional[Path] = typer.Option(None, help="Output filename"),
        strength: float = typer.Option(1.0, help="Roughness map strength"),
        compress: int = typer.Option(0, help="Compression level (0-100)", min=0, max=100),
        format: str = typer.Option("png", help="Output format (png, jpg, webp)"),
        normalize: bool = typer.Option(False, "--normalize", "-n", help="Normalize height map before processing"),
        metadata: Optional[str] = typer.Option(
            None, "--metadata", "-m", help="Additional parameters as JSON string")
    ):
        """Export a roughness map."""
        output_dir = Path(output_file).parent if output_file else input_file.parent
        export_maps_command(input_file, output_dir, ["roughness"], strength=strength,
                            compress=compress, format=format, normalize=normalize, metadata=metadata)

    @app.command("metallic")
    def metallic(
        input_file: Path = typer.Argument(..., help="Input TMD file", exists=True),
        output_file: Optional[Path] = typer.Option(None, help="Output filename"),
        strength: float = typer.Option(1.0, help="Metallic map strength"),
        compress: int = typer.Option(0, help="Compression level (0-100)", min=0, max=100),
        format: str = typer.Option("png", help="Output format (png, jpg, webp)"),
        normalize: bool = typer.Option(False, "--normalize", "-n", help="Normalize height map before processing"),
        metadata: Optional[str] = typer.Option(
            None, "--metadata", "-m", help="Additional parameters as JSON string")
    ):
        """Export a metallic map."""
        output_dir = Path(output_file).parent if output_file else input_file.parent
        export_maps_command(input_file, output_dir, ["metallic"], strength=strength,
                            compress=compress, format=format, normalize=normalize, metadata=metadata)

    @app.command("displacement")
    def displacement(
        input_file: Path = typer.Argument(..., help="Input TMD file", exists=True),
        output_file: Optional[Path] = typer.Option(None, help="Output filename"),
        intensity: float = typer.Option(1.0, help="Displacement intensity"),
        compress: int = typer.Option(0, help="Compression level (0-100)", min=0, max=100),
        format: str = typer.Option("png", help="Output format (png, jpg, webp)"),
        normalize: bool = typer.Option(False, "--normalize", "-n", help="Normalize height map before processing"),
        metadata: Optional[str] = typer.Option(
            None, "--metadata", "-m", help="Additional parameters as JSON string")
    ):
        """Export a displacement map."""
        output_dir = Path(output_file).parent if output_file else input_file.parent
        export_maps_command(input_file, output_dir, ["displacement"], intensity=intensity,
                            compress=compress, format=format, normalize=normalize, metadata=metadata)

    @app.command("height")
    def height(
        input_file: Path = typer.Argument(..., help="Input TMD file", exists=True),
        output_file: Optional[Path] = typer.Option(None, help="Output filename"),
        colormap: str = typer.Option("viridis", help="Colormap to use"),
        compress: int = typer.Option(0, help="Compression level (0-100)", min=0, max=100),
        format: str = typer.Option("png", help="Output format (png, jpg, webp)"),
        normalize: bool = typer.Option(False, "--normalize", "-n", help="Normalize height map before processing"),
        metadata: Optional[str] = typer.Option(
            None, "--metadata", "-m", help="Additional parameters as JSON string")
    ):
        """Export a height map."""
        output_dir = Path(output_file).parent if output_file else input_file.parent
        export_maps_command(input_file, output_dir, ["height"], colormap=colormap,
                            compress=compress, format=format, normalize=normalize, metadata=metadata)

    @app.command("hillshade")
    def hillshade(
        input_file: Path = typer.Argument(..., help="Input TMD file", exists=True),
        output_file: Optional[Path] = typer.Option(None, help="Output filename"),
        azimuth: float = typer.Option(315.0, help="Light source azimuth in degrees"),
        altitude: float = typer.Option(45.0, help="Light source altitude in degrees"),
        compress: int = typer.Option(0, help="Compression level (0-100)", min=0, max=100),
        format: str = typer.Option("png", help="Output format (png, jpg, webp)"),
        normalize: bool = typer.Option(False, "--normalize", "-n", help="Normalize height map before processing"),
        metadata: Optional[str] = typer.Option(
            None, "--metadata", "-m", help="Additional parameters as JSON string")
    ):
        """Export a hillshade map."""
        output_dir = Path(output_file).parent if output_file else input_file.parent
        export_maps_command(input_file, output_dir, ["hillshade"], azimuth=azimuth, altitude=altitude,
                            compress=compress, format=format, normalize=normalize, metadata=metadata)

    @app.command("all")
    def all_maps(
        input_file: Path = typer.Argument(..., help="Input TMD file", exists=True),
        output_dir: Optional[Path] = typer.Option(
            None, "--output", "-o", help="Output directory (default: ./textures)"),
        types: Optional[List[str]] = typer.Option(
            None, "--types", "-t", help="Map types to generate"),
        compress: int = typer.Option(0, "--compress", "-c", help="Compression (0-100)"),
        format: str = typer.Option("png", "--format", "-f", help="Output format"),
        strength: float = typer.Option(1.0, "--strength", "-s", help="Map strength"),
        normalize: bool = typer.Option(False, "--normalize", "-n", help="Normalize height map before processing"),
        metadata: Optional[str] = typer.Option(
            None, "--metadata", "-m", help="Additional parameters as JSON string")
    ):
        """Export all or specified map types."""
        # If no output dir specified, use textures subdirectory
        if output_dir is None:
            output_dir = Path("textures")

        export_maps_command(input_file, output_dir, types,
                            compress=compress, format=format, strength=strength,
                            normalize=normalize, metadata=metadata)

    @app.command("synthetic")
    def synthetic(
        pattern: str = typer.Argument(
            "waves",
            help="Pattern type (waves, peak, dome, ramp, combined, flat, random, perlin, fbm, square, sawtooth)"
        ),
        width: int = typer.Option(1024, "--width", "-w", help="Width of the height map"),
        height: int = typer.Option(1024, "--height", "-h", help="Height of the height map"),
        output_dir: Optional[Path] = typer.Option(
            None, "--output", "-o", help="Output directory (default: ./textures)"),
        types: Optional[List[str]] = typer.Option(
            None, "--types", "-t", help="Map types to generate"),
        compress: int = typer.Option(0, "--compress", "-c", help="Compression (0-100)"),
        format: str = typer.Option("png", "--format", "-f", help="Output format"),
        normalize: bool = typer.Option(False, "--normalize", "-n", help="Normalize height map before processing"),
        metadata: Optional[str] = typer.Option(
            None, "--metadata", "-m", help="Additional parameters as JSON string")
    ):
        """Generate and export maps from synthetic TMD data."""
        if not generate_synthetic_terrain(pattern, width, height, output_dir, types, compress, format, 
                                        normalize=normalize, metadata=metadata):
            raise typer.Exit(1)

    return app

    @app.command("parallax_ao")
    def parallax_ao(
        input_file: Path = typer.Argument(..., help="Input TMD file", exists=True),
        output_file: Optional[Path] = typer.Option(None, help="Output filename"),
        strength: float = typer.Option(1.0, help="Overall effect strength"),
        samples: int = typer.Option(16, help="Number of sample directions for AO calculation"),
        steep_threshold: float = typer.Option(45.0, help="Angle in degrees above which terrain is considered steep"),
        steep_multiplier: float = typer.Option(2.0, help="Effect multiplier for steep areas"),
        slope_sensitivity: float = typer.Option(0.5, help="How much slope affects the AO intensity (0-1)"),
        shadow_softness: float = typer.Option(1.0, help="Softness of shadow transitions"),
        max_distance: float = typer.Option(0.05, help="Maximum sampling distance as percentage of image size"),
        directional_bias: float = typer.Option(0.0, help="Bias strength toward a specific direction (0-1)"),
        bias_direction: float = typer.Option(315.0, help="Direction angle for bias in degrees (0=N, 90=E, etc.)"),
        cavity_emphasis: float = typer.Option(1.0, help="Factor to emphasize concave areas (crevices, valleys)"),
        multi_scale: bool = typer.Option(True, help="Use multi-resolution sampling for better quality"),
        compress: int = typer.Option(0, help="Compression level (0-100)", min=0, max=100),
        format: str = typer.Option("png", help="Output format (png, jpg, webp)"),
        normalize: bool = typer.Option(False, "--normalize", "-n", help="Normalize height map before processing"),
        metadata: Optional[str] = typer.Option(
            None, "--metadata", "-m", help="Additional parameters as JSON string")
    ):
        """Export a parallax ambient occlusion map with enhanced slope and steepness features."""
        output_dir = Path(output_file).parent if output_file else input_file.parent
        export_maps_command(
            input_file, output_dir, ["parallax_ao"], 
            strength=strength, samples=samples, 
            steep_threshold=steep_threshold, steep_multiplier=steep_multiplier,
            slope_sensitivity=slope_sensitivity, shadow_softness=shadow_softness,
            max_distance=max_distance, directional_bias=directional_bias, 
            bias_direction=bias_direction, cavity_emphasis=cavity_emphasis,
            multi_scale=multi_scale, compress=compress, format=format, 
            normalize=normalize, metadata=metadata
        )

    @app.command("angle")
    def angle(
        input_file: Path = typer.Argument(..., help="Input TMD file", exists=True),
        output_file: Optional[Path] = typer.Option(None, help="Output filename"),
        mode: str = typer.Option("gradient", help="Visualization mode (gradient, binary, hypsometric, aspect, classified, contour, custom)"),
        max_angle: float = typer.Option(90.0, help="Maximum angle in degrees (white in output)"),
        min_angle: float = typer.Option(0.0, help="Minimum angle in degrees (black in output)"),
        smoothing: float = typer.Option(0.5, help="Amount of smoothing to apply (0-5)"),
        highlight_range: Optional[str] = typer.Option(None, help="Range of angles to highlight as min:max (e.g. 30:45)"),
        aspect_direction: bool = typer.Option(False, help="Include directional information (aspect)"),
        terrain_class: bool = typer.Option(False, help="Use terrain classification mode"),
        edge_preservation: float = typer.Option(0.5, help="How much to preserve edges when smoothing (0-1)"),
        compress: int = typer.Option(0, help="Compression level (0-100)", min=0, max=100),
        format: str = typer.Option("png", help="Output format (png, jpg, webp)"),
        normalize: bool = typer.Option(False, "--normalize", "-n", help="Normalize height map before processing"),
        metadata: Optional[str] = typer.Option(
            None, "--metadata", "-m", help="Additional parameters as JSON string")
    ):
        """Export an enhanced angle/slope map showing terrain steepness with multiple visualization modes."""
        output_dir = Path(output_file).parent if output_file else input_file.parent
        
        # Parse highlight_range to a tuple if provided
        highlight_range_tuple = parse_tuple_range(highlight_range)
        
        export_maps_command(
            input_file, output_dir, ["angle"], 
            mode=mode, max_angle=max_angle, min_angle=min_angle,
            smoothing=smoothing, highlight_range=highlight_range_tuple,
            aspect_direction=aspect_direction, terrain_class=terrain_class,
            edge_preservation=edge_preservation,
            compress=compress, format=format, normalize=normalize, metadata=metadata
        )

    @app.command("curvature")
    def curvature(
        input_file: Path = typer.Argument(..., help="Input TMD file", exists=True),
        output_file: Optional[Path] = typer.Option(None, help="Output filename"),
        mode: str = typer.Option("mean", help="Curvature type (mean, gaussian, profile, planform, maximal, minimal)"),
        visualization: str = typer.Option("grayscale", help="Visualization type (grayscale, color, classified, edges, features, multi, divergent)"),
        scale: float = typer.Option(1.0, help="Scaling factor for curvature values"),
        sigma: float = typer.Option(1.0, help="Gaussian smoothing radius for preprocessing"),
        multi_scale: bool = typer.Option(False, help="Use multi-scale analysis for better feature detection"),
        highlight_features: bool = typer.Option(False, help="Highlight detected features"),
        feature_threshold: float = typer.Option(0.1, help="Threshold for feature detection (0-1)"),
        enhance_contrast: float = typer.Option(1.0, help="Contrast enhancement factor"),
        colormap: str = typer.Option("coolwarm", help="Colormap for color visualization"),
        percentile_clip: str = typer.Option("2:98", help="Percentiles for range clipping as min:max (e.g. 2:98)"),
        edge_width: int = typer.Option(1, help="Width of edges in 'edges' visualization mode"),
        compress: int = typer.Option(0, help="Compression level (0-100)", min=0, max=100),
        format: str = typer.Option("png", help="Output format (png, jpg, webp)"),
        normalize: bool = typer.Option(False, "--normalize", "-n", help="Normalize height map before processing"),
        metadata: Optional[str] = typer.Option(
            None, "--metadata", "-m", help="Additional parameters as JSON string")
    ):
        """Export an enhanced curvature map with multiple visualization modes."""
        output_dir = Path(output_file).parent if output_file else input_file.parent
        
        # Parse percentile_clip to a tuple
        percentile_clip_tuple = parse_tuple_range(percentile_clip)
        
        export_maps_command(
            input_file, output_dir, ["curvature"], 
            mode=mode, visualization=visualization, scale=scale,
            sigma=sigma, multi_scale=multi_scale, 
            highlight_features=highlight_features, feature_threshold=feature_threshold,
            enhance_contrast=enhance_contrast, colormap=colormap,
            percentile_clip=percentile_clip_tuple, edge_width=edge_width,
            compress=compress, format=format, normalize=normalize, metadata=metadata
        )

    @app.command("depth")
    def depth(
        input_file: Path = typer.Argument(..., help="Input TMD file", exists=True),
        output_file: Optional[Path] = typer.Option(None, help="Output filename"),
        mode: str = typer.Option("linear", help="Depth calculation mode (linear, inverse, focal)"),
        reverse: bool = typer.Option(False, help="Reverse depth values (far=dark, near=light)"),
        min_depth: float = typer.Option(0.0, help="Minimum depth value in output"),
        max_depth: float = typer.Option(1.0, help="Maximum depth value in output"),
        focal_plane: float = typer.Option(0.5, help="Relative position of focal plane (0-1) for 'focal' mode"),
        focal_range: float = typer.Option(0.2, help="Range around focal plane that appears in focus"),
        smoothing: float = typer.Option(0.0, help="Amount of smoothing to apply to depth values"),
        visualization: str = typer.Option("grayscale", help="Output type (grayscale, color, heatmap)"),
        colormap: str = typer.Option("plasma", help="Colormap for color visualizations"),
        enhance_contrast: float = typer.Option(1.0, help="Contrast enhancement factor"),
        compress: int = typer.Option(0, help="Compression level (0-100)", min=0, max=100),
        format: str = typer.Option("png", help="Output format (png, jpg, webp)"),
        normalize: bool = typer.Option(True, "--normalize", "-n", help="Normalize height map before processing"),
        metadata: Optional[str] = typer.Option(
            None, "--metadata", "-m", help="Additional parameters as JSON string")
    ):
        """Export a depth map showing distance from viewpoint to surfaces."""
        output_dir = Path(output_file).parent if output_file else input_file.parent
        export_maps_command(
            input_file, output_dir, ["depth"], 
            mode=mode, reverse=reverse, min_depth=min_depth, max_depth=max_depth,
            focal_plane=focal_plane, focal_range=focal_range, smoothing=smoothing,
            visualization=visualization, colormap=colormap, enhance_contrast=enhance_contrast,
            compress=compress, format=format, normalize=normalize, metadata=metadata
        )