"""Mesh export functionality for the CLI"""
import typer
from pathlib import Path
from typing import Optional, List
import logging

from rich.console import Console
from rich.panel import Panel

# Import core functionality from commands
from ..commands.model import (
    export_model,
    batch_export_models, 
    list_model_formats,
    get_quality_params,
    MeshMethod,
    QualityPreset
)

# Set up logging
logger = logging.getLogger(__name__)
console = Console()

def create_export_mesh_app() -> typer.Typer:
    """Create the mesh export app with both legacy and modern commands."""
    app = typer.Typer(
        help="Export TMD files to 3D model formats (Legacy and Modern)",
        short_help="Export to 3D model formats"
    )

    # =============================================================================
    # LEGACY COMMANDS (with deprecation warnings)
    # =============================================================================
    
    @app.command("list-legacy")
    def list_formats_legacy():
        """List all available mesh export formats (Legacy)."""
        console.print("[yellow]Note: This is the legacy mesh export command.[/yellow]")
        console.print("[cyan]Please use 'tmd model formats' for the updated command.[/cyan]")
        
        formats = ["stl", "obj", "ply", "gltf", "usd"]
        for format_name in formats:
            console.print(f"  - {format_name}")

    @app.command("stl-legacy")
    def export_stl_legacy(
        input_file: Path = typer.Argument(..., help="Input TMD file"),
        output_file: Optional[Path] = typer.Option(None, help="Output filename"),
        scale: float = typer.Option(1.0, help="Mesh scale factor"),
        method: MeshMethod = typer.Option(MeshMethod.ADAPTIVE, help="Mesh generation method"),
        quality: QualityPreset = typer.Option(QualityPreset.NORMAL, help="Quality preset"),
        binary: bool = typer.Option(True, help="Use binary format")
    ):
        """Export as STL mesh (Legacy - use 'tmd model generate' instead)."""
        console.print("[yellow]Warning: This is a legacy command.[/yellow]")
        console.print("[cyan]Please use: tmd model generate --format stl[/cyan]")
        
        # Set output file if not specified
        if output_file is None:
            output_file = input_file.with_suffix(".stl")
        
        console.print(f"Would export {input_file} to {output_file}")
        console.print("Use the new 'tmd model generate' command for actual export functionality.")

    # =============================================================================
    # MODERN COMMANDS (full functionality)
    # =============================================================================

    @app.command("generate")
    def generate_command(
        tmd_file: Path = typer.Argument(..., help="Input TMD file"),
        format: str = typer.Option("stl", help="Output format (stl, obj, ply, gltf, usd)"),
        output_file: Optional[Path] = typer.Option(None, help="Output filename"),
        method: MeshMethod = typer.Option(MeshMethod.QUADTREE, help="Mesh generation method"),
        quality: QualityPreset = typer.Option(QualityPreset.HIGH, help="Quality preset"),
        scale: float = typer.Option(5.0, help="Scale factor for height values"),
        save_heightmap: bool = typer.Option(True, help="Save heightmap visualization"),
        colormap: str = typer.Option("terrain", help="Colormap for heightmap visualization"),
        base_height: float = typer.Option(0.0, help="Height of solid base below model"),
        max_triangles: Optional[int] = typer.Option(None, help="Override maximum triangle count"),
        error_threshold: Optional[float] = typer.Option(None, help="Override error threshold"),
        max_subdivisions: int = typer.Option(12, help="Maximum quadtree subdivisions"),
        binary: Optional[bool] = typer.Option(None, help="Use binary format if supported"),
        coordinate_system: str = typer.Option("right-handed", help="Coordinate system orientation"),
        optimize: bool = typer.Option(True, help="Optimize mesh after generation")
    ):
        """Generate a 3D model from a TMD file."""
        try:
            # Get quality parameters
            quality_params = get_quality_params(quality)
            
            # Override with command line parameters if provided
            if error_threshold is not None:
                quality_params['error_threshold'] = error_threshold
            if max_triangles is not None:
                quality_params['max_triangles'] = max_triangles
        
            # Determine output filename
            if not output_file:
                output_file = tmd_file.with_suffix(f".{format.lower()}")
        
            # Export the model
            success = export_model(
                input_file=tmd_file,
                output_file=output_file,
                format=format,
                method=method,
                scale=scale,
                base_height=base_height,
                binary=binary,
                coordinate_system=coordinate_system,
                optimize=optimize,
                save_heightmap=save_heightmap,
                colormap=colormap,
                **quality_params
            )
        
            if not success:
                raise typer.Exit(code=1)

        except Exception as e:
            logger.error(f"Export failed: {e}", exc_info=True)
            console.print(f"[red]Export failed: {e}[/red]")
            raise typer.Exit(code=1)

    @app.command("formats")
    def list_formats():
        """List all available model export formats."""
        list_model_formats()

    @app.command("batch")
    def batch_command(
        input_dir: Path = typer.Argument(..., help="Directory containing TMD files"),
        output_dir: Optional[Path] = typer.Option(None, help="Output directory"),
        format: str = typer.Option("stl", help="Output format"),
        pattern: str = typer.Option("*.tmd", help="File pattern to match"),
        quality: QualityPreset = typer.Option(QualityPreset.NORMAL, help="Quality preset"),
        scale: float = typer.Option(5.0, help="Scale factor for height values"),
        max_workers: int = typer.Option(1, help="Number of parallel workers"),
        recursive: bool = typer.Option(False, help="Search recursively in subdirectories"),
        max_triangles: Optional[int] = typer.Option(None, help="Override maximum triangle count"),
        error_threshold: Optional[float] = typer.Option(None, help="Override error threshold"),
        binary: Optional[bool] = typer.Option(None, help="Use binary format if supported"),
        coordinate_system: str = typer.Option("right-handed", help="Coordinate system orientation"),
        optimize: bool = typer.Option(True, help="Optimize mesh after generation")
    ):
        """Batch process multiple TMD files."""
        try:
            # Prepare additional kwargs
            kwargs = {}
            if max_triangles is not None:
                kwargs['max_triangles'] = max_triangles
            if error_threshold is not None:
                kwargs['error_threshold'] = error_threshold
            if binary is not None:
                kwargs['binary'] = binary
            kwargs['coordinate_system'] = coordinate_system
            kwargs['optimize'] = optimize
            
            success = batch_export_models(
                input_dir=input_dir,
                output_dir=output_dir,
                format=format,
                pattern=pattern,
                quality=quality,
                scale=scale,
                max_workers=max_workers,
                recursive=recursive,
                **kwargs
            )
            
            if not success:
                raise typer.Exit(code=1)
                
        except Exception as e:
            logger.error(f"Batch processing failed: {e}", exc_info=True)
            console.print(f"[red]Batch processing failed: {e}[/red]")
            raise typer.Exit(code=1)

    # =============================================================================
    # FORMAT-SPECIFIC CONVENIENCE COMMANDS
    # =============================================================================

    @app.command("stl")
    def export_stl(
        input_file: Path = typer.Argument(..., help="Input TMD file"),
        output_file: Optional[Path] = typer.Option(None, help="Output filename"),
        scale: float = typer.Option(5.0, help="Mesh scale factor"),
        quality: QualityPreset = typer.Option(QualityPreset.HIGH, help="Quality preset"),
        binary: bool = typer.Option(True, help="Use binary format"),
        max_triangles: Optional[int] = typer.Option(None, help="Override maximum triangle count"),
        error_threshold: Optional[float] = typer.Option(None, help="Override error threshold")
    ):
        """Export as STL mesh."""
        # Get quality parameters and override if specified
        quality_params = get_quality_params(quality)
        if error_threshold is not None:
            quality_params['error_threshold'] = error_threshold
        if max_triangles is not None:
            quality_params['max_triangles'] = max_triangles
    
        # Set output file if not specified
        if output_file is None:
            output_file = input_file.with_suffix(".stl")
        
        success = export_model(
            input_file=input_file,
            output_file=output_file,
            format="stl",
            scale=scale,
            binary=binary,
            **quality_params
        )
        
        if not success:
            raise typer.Exit(code=1)

    @app.command("obj")
    def export_obj(
        input_file: Path = typer.Argument(..., help="Input TMD file"),
        output_file: Optional[Path] = typer.Option(None, help="Output filename"),
        scale: float = typer.Option(5.0, help="Mesh scale factor"),
        quality: QualityPreset = typer.Option(QualityPreset.HIGH, help="Quality preset"),
        max_triangles: Optional[int] = typer.Option(None, help="Override maximum triangle count"),
        error_threshold: Optional[float] = typer.Option(None, help="Override error threshold")
    ):
        """Export as OBJ mesh."""
        # Get quality parameters and override if specified
        quality_params = get_quality_params(quality)
        if error_threshold is not None:
            quality_params['error_threshold'] = error_threshold
        if max_triangles is not None:
            quality_params['max_triangles'] = max_triangles
    
        # Set output file if not specified
        if output_file is None:
            output_file = input_file.with_suffix(".obj")
        
        success = export_model(
            input_file=input_file,
            output_file=output_file,
            format="obj",
            scale=scale,
            **quality_params
        )
        
        if not success:
            raise typer.Exit(code=1)

    @app.command("ply")
    def export_ply(
        input_file: Path = typer.Argument(..., help="Input TMD file"),
        output_file: Optional[Path] = typer.Option(None, help="Output filename"),
        scale: float = typer.Option(5.0, help="Mesh scale factor"),
        quality: QualityPreset = typer.Option(QualityPreset.HIGH, help="Quality preset"),
        binary: bool = typer.Option(True, help="Use binary format"),
        max_triangles: Optional[int] = typer.Option(None, help="Override maximum triangle count"),
        error_threshold: Optional[float] = typer.Option(None, help="Override error threshold")
    ):
        """Export as PLY mesh."""
        # Get quality parameters and override if specified
        quality_params = get_quality_params(quality)
        if error_threshold is not None:
            quality_params['error_threshold'] = error_threshold
        if max_triangles is not None:
            quality_params['max_triangles'] = max_triangles
    
        # Set output file if not specified
        if output_file is None:
            output_file = input_file.with_suffix(".ply")
        
        success = export_model(
            input_file=input_file,
            output_file=output_file,
            format="ply",
            scale=scale,
            binary=binary,
            **quality_params
        )
        
        if not success:
            raise typer.Exit(code=1)

    @app.command("gltf")
    def export_gltf(
        input_file: Path = typer.Argument(..., help="Input TMD file"),
        output_file: Optional[Path] = typer.Option(None, help="Output filename"),
        scale: float = typer.Option(5.0, help="Mesh scale factor"),
        quality: QualityPreset = typer.Option(QualityPreset.HIGH, help="Quality preset"),
        binary: bool = typer.Option(True, help="Use GLB format"),
        max_triangles: Optional[int] = typer.Option(None, help="Override maximum triangle count"),
        error_threshold: Optional[float] = typer.Option(None, help="Override error threshold")
    ):
        """Export as GLTF/GLB mesh."""
        # Get quality parameters and override if specified
        quality_params = get_quality_params(quality)
        if error_threshold is not None:
            quality_params['error_threshold'] = error_threshold
        if max_triangles is not None:
            quality_params['max_triangles'] = max_triangles
    
        # Set output file if not specified
        if output_file is None:
            suffix = ".glb" if binary else ".gltf"
            output_file = input_file.with_suffix(suffix)
        
        success = export_model(
            input_file=input_file,
            output_file=output_file,
            format="gltf",
            scale=scale,
            binary=binary,
            **quality_params
        )
        
        if not success:
            raise typer.Exit(code=1)

    @app.command("usd")
    def export_usd(
        input_file: Path = typer.Argument(..., help="Input TMD file"),
        output_file: Optional[Path] = typer.Option(None, help="Output filename"),
        scale: float = typer.Option(5.0, help="Mesh scale factor"),
        quality: QualityPreset = typer.Option(QualityPreset.HIGH, help="Quality preset"),
        binary: bool = typer.Option(True, help="Use binary USD format"),
        max_triangles: Optional[int] = typer.Option(None, help="Override maximum triangle count"),
        error_threshold: Optional[float] = typer.Option(None, help="Override error threshold")
    ):
        """Export as USD mesh."""
        # Get quality parameters and override if specified
        quality_params = get_quality_params(quality)
        if error_threshold is not None:
            quality_params['error_threshold'] = error_threshold
        if max_triangles is not None:
            quality_params['max_triangles'] = max_triangles
    
        # Set output file if not specified
        if output_file is None:
            suffix = ".usd" if binary else ".usda"
            output_file = input_file.with_suffix(suffix)
        
        success = export_model(
            input_file=input_file,
            output_file=output_file,
            format="usd",
            scale=scale,
            binary=binary,
            **quality_params
        )
        
        if not success:
            raise typer.Exit(code=1)

    return app