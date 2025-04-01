#!/usr/bin/env python3
"""
Visualization app for TMD CLI.
"""

from pathlib import Path
from typing import Optional, List
import typer
from enum import Enum

from tmd.cli.core.ui import console, print_error, print_warning, print_success
from tmd.cli.core.io import auto_open_file
from tmd.cli.utils.visualization import create_visualization

class PlotterChoice(str, Enum):
    """Supported plotter backends."""
    MATPLOTLIB = "matplotlib"
    PLOTLY = "plotly"
    SEABORN = "seaborn"
    POLYSCOPE = "polyscope"
    AUTO = "auto"  # Automatically choose best available

class ColorMapChoice(str, Enum):
    """Common colormaps across plotting backends."""
    VIRIDIS = "viridis"
    PLASMA = "plasma"
    INFERNO = "inferno"
    MAGMA = "magma"
    JET = "jet"
    COOLWARM = "coolwarm"
    RAINBOW = "rainbow"
    TERRAIN = "terrain"
    BLUES = "Blues"
    REDS = "Reds"
    GREENS = "Greens"
    CIVIDIS = "cividis"

def create_visualize_app():
    """Create the visualization app with all commands."""
    visualize_app = typer.Typer(help="Visualization tools for TMD files")
    
    # Basic commands
    visualize_app.command(name="basic")(visualize_basic)
    visualize_app.command(name="3d")(visualize_3d)
    visualize_app.command(name="profile")(visualize_profile)
    
    # Advanced visualization options
    visualize_app.command(name="contour")(visualize_contour)
    visualize_app.command(name="fancy")(visualize_enhanced)
    visualize_app.command(name="compare")(visualize_comparison)
    
    # Polyscope 3D visualization commands
    visualize_app.command(name="ps-3d")(visualize_polyscope_3d)
    visualize_app.command(name="ps-pointcloud")(visualize_polyscope_pointcloud)
    visualize_app.command(name="ps-mesh")(visualize_polyscope_mesh)
    
    # Utilities
    visualize_app.command(name="backends")(list_backends)
    visualize_app.command(name="examples")(show_examples)
    
    return visualize_app

def visualize_basic(
    tmd_file: Path = typer.Argument(..., help="Path to TMD file", exists=True),
    plotter: PlotterChoice = typer.Option(PlotterChoice.AUTO, help="Visualization backend to use"),
    colormap: ColorMapChoice = typer.Option(ColorMapChoice.VIRIDIS, help="Colormap name"),
    output: Optional[Path] = typer.Option(None, help="Output filename"),
    auto_open: bool = typer.Option(False, help="Automatically open the output file"),
    cache: bool = typer.Option(True, help="Use cache if available")
):
    """Create a basic 2D visualization of a TMD file."""
    plotter_name = _resolve_plotter(plotter)
    
    try:
        success = create_visualization(
            tmd_file_or_data=tmd_file,
            mode="2d",
            plotter=plotter_name,
            colormap=colormap.value,
            output=output,
            use_cache=cache
        )
        
        if success and auto_open and output:
            auto_open_file(output)
            
        return 0 if success else 1
    except (NameError, ImportError):
        print_error("Visualization functionality is not available")
        return 1

def visualize_3d(
    tmd_file: Path = typer.Argument(..., help="Path to TMD file", exists=True),
    plotter: PlotterChoice = typer.Option(PlotterChoice.AUTO, help="Visualization backend to use"),
    colormap: ColorMapChoice = typer.Option(ColorMapChoice.VIRIDIS, help="Colormap name"),
    output: Optional[Path] = typer.Option(None, help="Output filename"),
    z_scale: float = typer.Option(1.0, help="Z-axis scaling factor"),
    wireframe: bool = typer.Option(False, help="Show wireframe (for supported backends)"),
    auto_open: bool = typer.Option(False, help="Automatically open the output file"),
    cache: bool = typer.Option(True, help="Use cache if available")
):
    """Create a 3D surface visualization of a TMD file."""
    plotter_name = _resolve_plotter(plotter)
    
    try:
        success = create_visualization(
            tmd_file_or_data=tmd_file,
            mode="3d",
            plotter=plotter_name,
            colormap=colormap.value,
            output=output,
            z_scale=z_scale,
            wireframe=wireframe,
            use_cache=cache
        )
        
        if success and auto_open and output:
            auto_open_file(output)
            
        return 0 if success else 1
    except (NameError, ImportError):
        print_error("Visualization functionality is not available")
        return 1

def visualize_profile(
    tmd_file: Path = typer.Argument(..., help="Path to TMD file", exists=True),
    row: int = typer.Option(-1, help="Row index for profile (-1 for middle row)"),
    plotter: PlotterChoice = typer.Option(PlotterChoice.AUTO, help="Visualization backend to use"),
    colormap: ColorMapChoice = typer.Option(ColorMapChoice.VIRIDIS, help="Colormap name"),
    output: Optional[Path] = typer.Option(None, help="Output filename"),
    show_markers: bool = typer.Option(True, help="Show markers on the profile line"),
    show_grid: bool = typer.Option(True, help="Show grid lines"),
    marker_size: int = typer.Option(5, help="Size of markers (if shown)"),
    fill: bool = typer.Option(True, help="Fill area under the profile line"),
    auto_open: bool = typer.Option(False, help="Automatically open the output file"),
    cache: bool = typer.Option(True, help="Use cache if available")
):
    """Create a profile (cross-section) visualization of a TMD file."""
    plotter_name = _resolve_plotter(plotter)
    
    # Use a profile_row of None to indicate middle row should be used
    profile_row = row if row >= 0 else None
    
    # Extra options for seaborn profiles
    extra_opts = {}
    if plotter_name.lower() == "seaborn":
        extra_opts = {
            "marker_size": marker_size,
            "fill": fill
        }
    
    try:
        success = create_visualization(
            tmd_file_or_data=tmd_file,
            mode="profile",
            plotter=plotter_name,
            colormap=colormap.value,
            output=output,
            profile_row=profile_row,
            show_markers=show_markers,
            show_grid=show_grid,
            use_cache=cache,
            **extra_opts
        )
        
        if success and auto_open and output:
            auto_open_file(output)
            
        return 0 if success else 1
    except (NameError, ImportError):
        print_error("Visualization functionality is not available")
        return 1

def visualize_contour(
    tmd_file: Path = typer.Argument(..., help="Path to TMD file", exists=True),
    plotter: PlotterChoice = typer.Option(PlotterChoice.AUTO, help="Visualization backend to use"),
    colormap: ColorMapChoice = typer.Option(ColorMapChoice.VIRIDIS, help="Colormap name"),
    output: Optional[Path] = typer.Option(None, help="Output filename"),
    levels: int = typer.Option(20, help="Number of contour levels"),
    show_lines: bool = typer.Option(True, help="Show contour lines"),
    auto_open: bool = typer.Option(False, help="Automatically open the output file"),
    cache: bool = typer.Option(True, help="Use cache if available")
):
    """Create a contour plot visualization of a TMD file."""
    plotter_name = _resolve_plotter(plotter)
    
    try:
        success = create_visualization(
            tmd_file_or_data=tmd_file,
            mode="contour",
            plotter=plotter_name,
            colormap=colormap.value,
            output=output,
            levels=levels,
            show_lines=show_lines,
            use_cache=cache
        )
        
        if success and auto_open and output:
            auto_open_file(output)
            
        return 0 if success else 1
    except (NameError, ImportError):
        print_error("Visualization functionality is not available")
        return 1

def visualize_enhanced(
    tmd_file: Path = typer.Argument(..., help="Path to TMD file", exists=True),
    plotter: PlotterChoice = typer.Option("seaborn", help="Visualization backend to use"),
    colormap: ColorMapChoice = typer.Option(ColorMapChoice.VIRIDIS, help="Colormap name"),
    output: Optional[Path] = typer.Option(None, help="Output filename"),
    mode: str = typer.Option("enhanced", help="Enhanced mode: 'enhanced', 'distribution', or 'joint'"),
    auto_open: bool = typer.Option(False, help="Automatically open the output file"),
    cache: bool = typer.Option(True, help="Use cache if available")
):
    """Create an enhanced visualization using Seaborn's statistical features."""
    plotter_name = _resolve_plotter(plotter)
    
    # Warn if user didn't select seaborn or plotly
    if plotter_name not in ["seaborn", "plotly"]:
        print_warning(f"Enhanced visualizations work best with seaborn or plotly, not {plotter_name}")
    
    try:
        # For different enhanced visualization types
        if mode == "distribution":
            extra_params = {"enhanced": True, "viz_type": "distribution", "kde": True, "bins": 50}
        elif mode == "joint":
            extra_params = {"enhanced": True, "viz_type": "joint"}
        else:  # default to enhanced heatmap
            extra_params = {"enhanced": True}
        
        success = create_visualization(
            tmd_file_or_data=tmd_file,
            mode="enhanced",
            plotter=plotter_name,
            colormap=colormap.value,
            output=output,
            use_cache=cache,
            **extra_params
        )
        
        if success and auto_open and output:
            auto_open_file(output)
            
        return 0 if success else 1
    except (NameError, ImportError):
        print_error("Enhanced visualization functionality is not available. Try installing seaborn.")
        return 1

def visualize_comparison(
    tmd_file: Path = typer.Argument(..., help="Path to TMD file", exists=True),
    second_file: Optional[Path] = typer.Option(None, help="Second TMD file to compare with"),
    plotter: PlotterChoice = typer.Option(PlotterChoice.AUTO, help="Visualization backend to use"),
    colormap: ColorMapChoice = typer.Option(ColorMapChoice.VIRIDIS, help="Colormap name"),
    output: Optional[Path] = typer.Option(None, help="Output filename"),
    auto_open: bool = typer.Option(False, help="Automatically open the output file"),
    cache: bool = typer.Option(True, help="Use cache if available")
):
    """Create a comparison visualization between two TMD files or between multiple profiles in the same file."""
    plotter_name = _resolve_plotter(plotter)
    
    try:
        # If second file is provided, create a multi-file comparison
        if second_file and second_file.exists():
            success = create_visualization(
                tmd_file_or_data=tmd_file,
                mode="comparison",
                plotter=plotter_name,
                colormap=colormap.value,
                output=output,
                second_file=second_file,
                use_cache=cache
            )
        else:
            # Otherwise create a multi-profile comparison from a single file
            # Use three rows at different positions (25%, 50%, 75%)
            success = create_visualization(
                tmd_file_or_data=tmd_file,
                mode="multi_profile",
                plotter=plotter_name,
                colormap=colormap.value,
                output=output,
                profile_rows=[0.25, 0.5, 0.75],  # Use percentages for different profiles
                use_cache=cache
            )
        
        if success and auto_open and output:
            auto_open_file(output)
            
        return 0 if success else 1
    except (NameError, ImportError):
        print_error("Comparison visualization functionality is not available")
        return 1

def visualize_polyscope_3d(
    tmd_file: Path = typer.Argument(..., help="Path to TMD file", exists=True),
    z_scale: float = typer.Option(1.0, help="Z-axis scaling factor"),
    colormap: ColorMapChoice = typer.Option(ColorMapChoice.VIRIDIS, help="Colormap name"),
    wireframe: bool = typer.Option(False, help="Show wireframe"),
    smooth: bool = typer.Option(True, help="Use smooth shading"),
    output: Optional[Path] = typer.Option(None, help="Output screenshot filename"),
    width: int = typer.Option(1024, help="Screenshot width in pixels"),
    height: int = typer.Option(768, help="Screenshot height in pixels"),
    interactive: bool = typer.Option(True, help="Show interactive window"),
    cache: bool = typer.Option(True, help="Use cache if available")
):
    """Create a 3D visualization of a TMD file using Polyscope (interactive 3D viewer)."""
    plotter_name = "polyscope"
    
    # Print warning if polyscope isn't available
    if not _check_polyscope_available():
        return 1
    
    try:
        success = create_visualization(
            tmd_file_or_data=tmd_file,
            mode="3d",
            plotter=plotter_name,
            colormap=colormap.value,
            output=output,
            z_scale=z_scale,
            wireframe=wireframe,
            smooth=smooth,
            width=width,
            height=height,
            show=interactive,
            use_cache=cache,
            auto_open=False  # Explicitly disable auto-open
        )
        
        return 0 if success else 1
    except (NameError, ImportError) as e:
        print_error(f"Visualization failed: {e}")
        return 1

def visualize_polyscope_pointcloud(
    tmd_file: Path = typer.Argument(..., help="Path to TMD file", exists=True),
    z_scale: float = typer.Option(1.0, help="Z-axis scaling factor"),
    colormap: ColorMapChoice = typer.Option(ColorMapChoice.VIRIDIS, help="Colormap name"),
    point_size: float = typer.Option(2.0, help="Point size"),
    sample_rate: int = typer.Option(1, help="Sample rate (1=all points, 2=every other point, etc.)"),
    output: Optional[Path] = typer.Option(None, help="Output screenshot filename"),
    width: int = typer.Option(1024, help="Screenshot width in pixels"),
    height: int = typer.Option(768, help="Screenshot height in pixels"),
    interactive: bool = typer.Option(True, help="Show interactive window"),
    cache: bool = typer.Option(True, help="Use cache if available"),
    use_fallback: bool = typer.Option(False, help="Use fallback plotter if polyscope fails")
):
    """Create a point cloud visualization of a TMD file using Polyscope."""
    # Check polyscope availability once before proceeding
    if not _check_polyscope_available():
        return 1
    
    try:
        success = create_visualization(
            tmd_file_or_data=tmd_file,
            mode="point_cloud",
            plotter="polyscope",
            colormap=colormap.value,
            output=output,
            z_scale=z_scale,
            point_size=point_size,
            sample_rate=sample_rate,
            width=width,
            height=height,
            show=interactive,
            use_cache=cache,
            use_fallback=use_fallback,
            auto_open=False  # Explicitly disable auto-open
        )
        
        return 0 if success else 1
    except (NameError, ImportError) as e:
        print_error(f"Visualization failed: {e}")
        return 1

def visualize_polyscope_mesh(
    tmd_file: Path = typer.Argument(..., help="Path to TMD file", exists=True),
    z_scale: float = typer.Option(1.0, help="Z-axis scaling factor"),
    colormap: ColorMapChoice = typer.Option(ColorMapChoice.VIRIDIS, help="Colormap name"),
    wireframe: bool = typer.Option(True, help="Show wireframe"),
    smooth: bool = typer.Option(False, help="Use smooth shading"),
    output: Optional[Path] = typer.Option(None, help="Output screenshot filename"),
    width: int = typer.Option(1024, help="Screenshot width in pixels"),
    height: int = typer.Option(768, help="Screenshot height in pixels"),
    interactive: bool = typer.Option(True, help="Show interactive window"),
    cache: bool = typer.Option(True, help="Use cache if available")
):
    """Create a triangle mesh visualization of a TMD file using Polyscope."""
    plotter_name = "polyscope"
    
    # Print warning if polyscope isn't available
    if not _check_polyscope_available():
        return 1
    
    try:
        success = create_visualization(
            tmd_file_or_data=tmd_file,
            mode="mesh",
            plotter=plotter_name,
            colormap=colormap.value,
            output=output,
            z_scale=z_scale,
            wireframe=wireframe,
            smooth=smooth,
            width=width,
            height=height,
            show=interactive,
            use_cache=cache,
            auto_open=False  # Explicitly disable auto-open
        )
        
        return 0 if success else 1
    except (NameError, ImportError) as e:
        print_error(f"Visualization failed: {e}")
        return 1

def list_backends():
    """Display information about available visualization backends."""
    try:
        from tmd.plotters import get_registered_plotters, get_available_plotters
        
        console.print("[bold]Registered plotters:[/bold]")
        registered = get_registered_plotters()
        
        available = get_available_plotters()
        
        for name, status in registered.items():
            status_str = "[green]Available[/green]" if status else "[red]Not available[/red]"
            console.print(f"  - {name}: {status_str}")
            
        console.print("\n[bold]Recommended plotter:[/bold]")
        if available:
            console.print(f"  - [green]{available[0]}[/green]")
        else:
            console.print("  - [red]No visualization backends available[/red]")
            
        console.print("\n[bold]Installation instructions:[/bold]")
        console.print("  - matplotlib: [dim]pip install matplotlib[/dim]")
        console.print("  - plotly: [dim]pip install plotly[/dim]")
        console.print("  - seaborn: [dim]pip install seaborn matplotlib[/dim]")
        console.print("  - polyscope: [dim]pip install polyscope[/dim]")
        
        return 0
    except ImportError:
        print_error("Visualization backend detection failed")
        return 1

def show_examples():
    """Display example commands for visualizing TMD files."""
    console.print("[bold]TMD Visualization Examples:[/bold]\n")
    
    console.print("[bold cyan]Basic 2D Visualization:[/bold cyan]")
    console.print("  python tmd_cli.py visualize basic Dime.tmd --colormap viridis")
    console.print("  python tmd_cli.py visualize basic Dime.tmd --plotter plotly --output dime_visualization.html\n")
    
    console.print("[bold cyan]3D Surface Visualization:[/bold cyan]")
    console.print("  python tmd_cli.py visualize 3d Dime.tmd --z-scale 2.0")
    console.print("  python tmd_cli.py visualize 3d Dime.tmd --plotter plotly --z-scale 1.5 --colormap plasma\n")
    
    console.print("[bold cyan]Profile Visualization:[/bold cyan]")
    console.print("  python tmd_cli.py visualize profile Dime.tmd --row 50")
    console.print("  python tmd_cli.py visualize profile Dime.tmd --plotter seaborn --row 100 --no-show-markers\n")
    
    console.print("[bold cyan]Contour Visualization:[/bold cyan]")
    console.print("  python tmd_cli.py visualize contour Dime.tmd --levels 15")
    console.print("  python tmd_cli.py visualize contour Dime.tmd --plotter plotly --colormap terrain\n")
    
    console.print("[bold cyan]Enhanced Visualizations (Seaborn/Plotly):[/bold cyan]")
    console.print("  python tmd_cli.py visualize fancy Dime.tmd --plotter seaborn")
    console.print("  python tmd_cli.py visualize fancy Dime.tmd --mode distribution --plotter seaborn")
    console.print("  python tmd_cli.py visualize fancy Dime.tmd --mode joint --plotter plotly\n")
    
    console.print("[bold cyan]Comparison Visualizations:[/bold cyan]")
    console.print("  python tmd_cli.py visualize compare Dime.tmd")
    console.print("  python tmd_cli.py visualize compare Dime.tmd --second-file Quarter.tmd --plotter seaborn\n")
    
    console.print("[bold cyan]Polyscope 3D Visualizations:[/bold cyan]")
    console.print("  python tmd_cli.py visualize ps-3d Dime.tmd --z-scale 2.0")
    console.print("  python tmd_cli.py visualize ps-pointcloud Dime.tmd --sample-rate 2 --point-size 3.0")
    console.print("  python tmd_cli.py visualize ps-mesh Dime.tmd --wireframe --no-smooth\n")
    
    console.print("[bold]Tips:[/bold]")
    console.print("- Use [cyan]--auto-open[/cyan] to automatically open the saved visualization")
    console.print("- Use [cyan]--plotter auto[/cyan] to let TMD choose the best available visualization backend")
    console.print("- Run [cyan]visualize backends[/cyan] to see all available visualization backends")
    
    return 0

def _resolve_plotter(plotter: PlotterChoice) -> str:
    """
    Resolves the plotter choice, using the best available plotter if AUTO is selected.
    
    Args:
        plotter: The plotter choice from the enum
        
    Returns:
        String name of the plotter to use
    """
    if plotter == PlotterChoice.AUTO:
        try:
            from tmd.plotters import get_available_plotters
            available = get_available_plotters()
            if available:
                print_warning(f"Auto-selecting {available[0]} visualization backend")
                return available[0]
            else:
                print_warning("No visualization backends available, defaulting to matplotlib")
                return "matplotlib"
        except ImportError:
            print_warning("Could not detect available plotters, defaulting to matplotlib")
            return "matplotlib"
    
    return plotter.value

def _check_polyscope_available() -> bool:
    """Check if polyscope is available and print appropriate message."""
    try:
        import polyscope
        return True
    except ImportError:
        print_error("Polyscope is not available. Install with: pip install polyscope")
        return False
