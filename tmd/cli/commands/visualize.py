#!/usr/bin/env python3
"""
Visualization commands for TMD CLI.

This module provides functions for creating various visualizations of TMD files
using different plotting backends.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np

# Import TMD core
from tmd import TMD

# Import CLI utilities
from tmd.cli.core import (
    console,
    print_warning,
    print_error,
    print_success,
    load_tmd_file,
    auto_open_file,
    create_output_dir,
    get_output_filename,
    get_file_extension
)

# Import caching utilities
from tmd.cli.utils.caching import get_cache_stats, clear_cache

# Import logging
import logging
logger = logging.getLogger(__name__)


def get_available_plotters() -> Dict[str, bool]:
    """
    Get a dictionary of available plotters and their status.
    
    Returns:
        Dict[str, bool]: Dictionary with plotter names as keys and 
                        availability status as values
    """
    try:
        from tmd.plotters import get_registered_plotters
        return get_registered_plotters()
    except ImportError:
        # Fallback if plotter factories aren't available
        plotters = {
            "matplotlib": False,
            "plotly": False,
            "seaborn": False,
            "polyscope": False
        }
        
        # Check matplotlib
        try:
            import matplotlib
            plotters["matplotlib"] = True
        except ImportError:
            pass
            
        # Check plotly
        try:
            import plotly
            plotters["plotly"] = True
        except ImportError:
            pass
            
        # Check seaborn
        try:
            import seaborn
            plotters["seaborn"] = True
        except ImportError:
            pass
            
        # Check polyscope
        try:
            import polyscope
            plotters["polyscope"] = True
        except ImportError:
            pass
            
        return plotters

def select_plotter(requested: str, viz_type: str = "2d") -> str:
    """
    Select an appropriate plotter based on availability.
    
    Args:
        requested: Name of the requested plotter
        viz_type: Type of visualization (e.g., "2d", "3d", "profile")
        
    Returns:
        Name of the selected plotter
    """
    available_plotters = get_available_plotters()
    
    # Check if requested plotter is available
    if requested.lower() in available_plotters and available_plotters[requested.lower()]:
        return requested.lower()
    
    # Otherwise find an available alternative
    for plotter, available in available_plotters.items():
        if available:
            print_warning(f"Plotter '{requested}' not available. Using '{plotter}' instead.")
            return plotter
    
    # No plotters available, return the original and let caller handle the error
    print_warning(f"No visualization backends available. Install matplotlib, plotly, or other visualization libraries.")
    return requested.lower()

def visualize_tmd_file(
    tmd_file: Path,
    mode: str = "2d",
    plotter: str = "matplotlib",
    colormap: str = "viridis",
    output: Optional[Path] = None,
    z_scale: float = 1.0,
    profile_row: Optional[int] = None,
    auto_open: bool = False,  # Default to False
    **kwargs
) -> bool:
    """
    Create a visualization of a TMD file.
    
    Args:
        tmd_file: Path to the TMD file
        mode: Visualization mode ("2d", "3d", "profile")
        plotter: Plotter backend to use
        colormap: Colormap to use
        output: Output file path
        z_scale: Z-axis scaling factor (for 3D plots)
        profile_row: Row to use for profile plot (for profile mode)
        auto_open: Whether to automatically open the output file
        
    Returns:
        True if visualization was successful, False otherwise
    """
    # Load TMD file
    data = load_tmd_file(tmd_file, with_console_status=True)
    if not data:
        print_error(f"Failed to load TMD file: {tmd_file}")
        return False
    
    # Get height map
    height_map = data.height_map()
    
    # Try to use the CLI visualization utility function directly
    try:
        from tmd.cli.utils.visualization import create_visualization
        
        success = create_visualization(
            tmd_file_or_data=data,
            mode=mode,
            plotter=plotter,
            colormap=colormap,
            output=output,
            z_scale=z_scale,
            profile_row=profile_row,
            title=f"{tmd_file.name} - {mode.upper()} Visualization",
            auto_open=auto_open and plotter.lower() != "polyscope",  # Don't auto-open polyscope visuals
            **kwargs
        )
        
        # Disable auto-opening for polyscope visualizations
        if success and auto_open and output and plotter.lower() != "polyscope":
            auto_open_file(output)
            
        return success
    except (ImportError, Exception) as e:
        logger.warning(f"Could not use create_visualization: {e}. Falling back to direct approach.")
    
    # Select appropriate plotter
    selected_plotter = select_plotter(plotter, mode)
    
    # Generate output file path if not specified
    if not output:
        output_dir = create_output_dir("visualizations")
        suffix = ""
        if mode == "profile" and profile_row is not None:
            suffix = f"_row{profile_row}"
        output = output_dir / f"{tmd_file.stem}_{mode}{suffix}_{selected_plotter}{get_file_extension(selected_plotter)}"
    
    try:
        # Try to use factory-based plotters
        try:
            from tmd.plotters import TMDPlotterFactory
            
            with console.status(f"Creating {mode} visualization with {selected_plotter}..."):
                # Create plotter instance
                plotter_instance = TMDPlotterFactory.create_plotter(selected_plotter)
                
                # Create visualization based on mode
                if mode == "3d":
                    fig = plotter_instance.plot(
                        height_map,
                        mode="3d",
                        title=f"{tmd_file.name} - 3D Visualization",
                        colormap=colormap,
                        z_scale=z_scale,
                        **kwargs
                    )
                elif mode == "profile":
                    # Use middle row if not specified
                    if profile_row is None:
                        profile_row = height_map.shape[0] // 2
                    
                    # Ensure row is within bounds
                    if profile_row < 0 or profile_row >= height_map.shape[0]:
                        print_error(f"Row index {profile_row} out of bounds (max: {height_map.shape[0]-1})")
                        return False
                    
                    # Plot profile
                    fig = plotter_instance.plot(
                        height_map,
                        mode="profile",
                        profile_row=profile_row,
                        title=f"{tmd_file.name} - Height Profile (Row {profile_row})",
                        colormap=colormap,
                        **kwargs
                    )
                else:  # Default to 2D
                    fig = plotter_instance.plot(
                        height_map,
                        mode="2d",
                        title=f"{tmd_file.name} - 2D Visualization",
                        colormap=colormap,
                        **kwargs
                    )
                
                # Save figure
                saved_path = plotter_instance.save(fig, str(output))
                if saved_path:
                    print_success(f"Visualization saved to {saved_path}")
                else:
                    print_error(f"Failed to save visualization to {output}")
                    return False
                
                # Auto-open if requested
                if auto_open and plotter.lower() != "polyscope":
                    auto_open_file(output)
                
                return True
                
        except (ImportError, AttributeError) as e:
            # Fall back to direct implementation
            logger.warning(f"Factory approach failed: {e}. Using direct implementation.")
            
    except Exception as e:
        print_error(f"Error creating visualization: {e}")
        return False

def check_available_visualization_backends():
    """Check which visualization backends are available and print the results."""
    print_success("Checking available visualization backends...")
    
    plotters = get_available_plotters()
    
    for name, available in plotters.items():
        if available:
            print_success(f"✓ {name.capitalize()} is available")
            
            # Check extra features for matplotlib
            if name == "matplotlib":
                try:
                    from mpl_toolkits.mplot3d import Axes3D
                    print_success("  ✓ 3D plotting support available")
                except ImportError:
                    print_warning("  ✗ 3D plotting support not available")
        else:
            print_warning(f"✗ {name.capitalize()} is not available")
    
    # Display installation instructions for missing plotters
    missing = [name for name, available in plotters.items() if not available]
    if missing:
        print_warning("\nInstallation instructions for missing plotters:")
        
        for name in missing:
            if name == "matplotlib":
                print_warning("  matplotlib: pip install matplotlib")
            elif name == "plotly":
                print_warning("  plotly: pip install plotly")
            elif name == "seaborn":
                print_warning("  seaborn: pip install seaborn matplotlib")
            elif name == "polyscope":
                print_warning("  polyscope: pip install polyscope")
    
    return plotters
