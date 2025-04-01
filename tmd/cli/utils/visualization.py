#!/usr/bin/env python3
"""
Visualization utilities for TMD CLI.

This module provides functions for creating visualizations with different plotting backends.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple

# Set up logger
logger = logging.getLogger(__name__)
import numpy as np

from tmd.plotters import TMDPlotterFactory, TMDSequencePlotterFactory
from tmd.cli.utils.caching import get_cache_stats, clear_cache

# Delayed imports for CLI-specific modules to prevent circular dependencies
_io_module = None
_ui_module = None
_utils_module = None  # New lazy-loaded module
_mesh_converter_module = None  # New lazy-loaded module

def _get_io_module():
    """Lazily import io module to avoid circular imports."""
    global _io_module
    if _io_module is None:
        from tmd.cli.core import io
        _io_module = io
    return _io_module

def _get_ui_module():
    """Lazily import ui module to avoid circular imports."""
    global _ui_module
    if _ui_module is None:
        from tmd.cli.core import ui
        _ui_module = ui
    return _ui_module

def _get_utils_module():
    """Lazily import utils module to avoid circular imports."""
    global _utils_module
    if _utils_module is None:
        from tmd.utils import utils
        _utils_module = utils
    return _utils_module

def _get_mesh_converter_module():
    """Lazily import mesh_converter module to avoid circular imports."""
    global _mesh_converter_module
    if _mesh_converter_module is None:
        from tmd.utils import mesh_converter
        _mesh_converter_module = mesh_converter
    return _mesh_converter_module

def check_available_visualization_backends() -> Dict[str, bool]:
    """
    Check which visualization backends are available.
    
    Returns:
        Dictionary mapping backend names to availability.
    """
    ui = _get_ui_module()
    
    # Define backends to check
    backends = {
        "matplotlib": False,
        "plotly": False,
        "seaborn": False,
        "polyscope": False
    }
    
    # Check each backend
    for backend in backends:
        try:
            if backend == "matplotlib":
                import matplotlib
                backends[backend] = True
            elif backend == "plotly":
                import plotly
                backends[backend] = True
            elif backend == "seaborn":
                import seaborn
                backends[backend] = True
            elif backend == "polyscope":
                import polyscope
                backends[backend] = True
                
            if backends[backend]:
                ui.print_success(f"✓ {backend} is available")
        except ImportError:
            ui.print_warning(f"✗ {backend} is not available")
    
    return backends

def _get_height_map(tmd_obj):
    """
    Safely get the height map from a TMD object, handling both property and method implementations.
    
    Args:
        tmd_obj: TMD object with either height_map property or height_map() method
        
    Returns:
        NumPy array containing height map data
    """
    if tmd_obj is None:
        ui_module = _get_ui_module()
        ui_module.print_error("TMD object is None")
        return None
        
    if hasattr(tmd_obj, 'height_map'):
        if callable(tmd_obj.height_map):
            return tmd_obj.height_map()
        else:
            return tmd_obj.height_map
    else:
        ui_module = _get_ui_module()
        ui_module.print_error("TMD object has no height_map property or method")
        return None

def create_visualization(
    tmd_file_or_data: Union[Path, 'TMD'],
    mode: str,
    plotter: str,
    output: Optional[Path] = None,
    profile_row: Optional[int] = None,
    title: Optional[str] = None,
    colormap: str = "viridis",
    z_scale: float = 1.0,
    show_axes: bool = False,
    transparent: bool = False,
    use_cache: bool = True,
    auto_open: bool = False,  # Default to False to disable browser opening
    **kwargs
) -> bool:
    """
    Create a visualization of TMD data.
    
    Args:
        tmd_file_or_data: Path to TMD file or TMD object
        mode: Visualization mode ("2d", "3d", "profile", "contour", "enhanced", etc.)
        plotter: Plotter name ("matplotlib", "plotly", "seaborn", etc.)
        output: Output file path
        profile_row: Row index for profile visualization
        title: Plot title
        colormap: Colormap name
        z_scale: Z-axis scaling factor for 3D visualizations
        show_axes: Whether to show axes
        transparent: Whether to use transparent background
        use_cache: Whether to use cached data (if available)
        auto_open: Whether to automatically open the output file
        **kwargs: Additional options for plotter
        
    Returns:
        True if successful, False otherwise
    """
    io_module = _get_io_module()
    ui_module = _get_ui_module()
    
    try:
        # Load data if a path was provided
        height_map, file_path, title = _prepare_data_and_title(
            tmd_file_or_data, mode, title, use_cache, io_module, ui_module, **kwargs
        )
        
        if height_map is None:
            return False
                
        # Determine output path if not specified
        if output is None and file_path is not None:
            output = io_module.get_output_filename(
                file_path, 
                plotter, 
                viz_type=mode,
                subdir="visualizations"
            )
        
        # Special handling for seaborn profile visualization
        if mode == "profile" and plotter.lower() == "seaborn":
            success = _handle_seaborn_profile(
                height_map, profile_row, title, output, kwargs, ui_module, io_module, auto_open
            )
            if success is not None:  # If handled by special case
                return success
        
        # Create plotter instance using the factory
        with ui_module.console.status(f"Creating {mode} visualization with {plotter}..."):
            # Use the factory pattern to create the appropriate plotter
            try:
                plotter_instance = TMDPlotterFactory.create_plotter(plotter)
                
                # Special handling for polyscope
                if plotter.lower() == "polyscope":
                    auto_open = False  # Disable auto-open for polyscope
                
                # Set common visualization parameters
                common_params = {
                    "mode": mode,
                    "title": title,
                    "colormap": colormap,
                    "z_scale": z_scale,
                    "show_axes": show_axes,
                    "transparent": transparent
                }
                
                # Add mode-specific parameters
                if mode == "profile" and profile_row is not None:
                    # Validate profile row
                    if profile_row < 0 or profile_row >= height_map.shape[0]:
                        ui_module.print_error(f"Profile row {profile_row} out of bounds (max: {height_map.shape[0]-1})")
                        return False
                    common_params["profile_row"] = profile_row
                
                # Create visualization with all parameters
                fig = plotter_instance.plot(height_map, **common_params, **kwargs)
                
            except (ImportError, ValueError) as e:
                ui_module.print_error(f"Could not create plotter: {e}")
                
                # Only try fallback for non-polyscope plotters or if explicitly requested
                if plotter.lower() != "polyscope" or kwargs.get("use_fallback", True):
                    # Try to fall back to the best available plotter
                    from tmd.plotters import get_best_plotter
                    plotter_instance = get_best_plotter()
                    if not plotter_instance:
                        return False
                    ui_module.print_warning(f"Using {plotter_instance.NAME} as fallback plotter")
                    
                    # Try again with the fallback plotter
                    fig = plotter_instance.plot(height_map, **common_params, **kwargs)
                else:
                    # For polyscope with no fallback requested, just report the error
                    return False
            
            # Save the visualization
            if output is not None:
                # Ensure directory exists
                output.parent.mkdir(parents=True, exist_ok=True)
                saved_path = plotter_instance.save(fig, str(output))
                if saved_path:
                    ui_module.print_success(f"Visualization saved to {saved_path}")
                    
                    # Only auto-open if specifically requested and it's not a polyscope visualization
                    if auto_open and plotter.lower() != "polyscope":
                        io_module.auto_open_file(output)
                else:
                    ui_module.print_error(f"Failed to save visualization to {output}")
                    return False
            
            return True
            
    except Exception as e:
        ui_module.print_error(f"Error creating visualization: {e}")
        logger.error(f"Visualization error: {e}", exc_info=True)
        
        # Try using fallback method if we have the height map
        if 'height_map' in locals() and height_map is not None:
            return _try_fallback_visualization(
                height_map, 
                mode,
                plotter, 
                output, 
                profile_row, 
                title, 
                colormap, 
                z_scale, 
                show_axes, 
                transparent, 
                **kwargs
            )
        return False

def _prepare_data_and_title(
    tmd_file_or_data, mode, title, use_cache, io_module, ui_module, **kwargs
):
    """Helper function to prepare height map data and title."""
    if isinstance(tmd_file_or_data, Path):
        tmd_data = io_module.load_tmd_file(
            tmd_file_or_data, 
            with_console_status=True, 
            use_cache=use_cache
        )
        if tmd_data is None:
            return None, None, None
        
        height_map = _get_height_map(tmd_data)
        if height_map is None:
            return None, None, None
            
        file_path = tmd_file_or_data
        
        # Load second file if provided (for comparison mode)
        if mode == "comparison" and "second_file" in kwargs:
            second_file = kwargs.get("second_file")
            if second_file:
                second_data = io_module.load_tmd_file(
                    second_file,
                    with_console_status=True,
                    use_cache=use_cache
                )
                if second_data:
                    second_height_map = _get_height_map(second_data)
                    if second_height_map is not None:
                        kwargs["second_height_map"] = second_height_map
        
        # Auto-generate title if not provided
        if title is None:
            title = f"{file_path.stem} - {mode.upper()} Visualization"
    else:
        # Assume it's already a TMD object
        tmd_data = tmd_file_or_data
        height_map = _get_height_map(tmd_data)
        if height_map is None:
            return None, None, None
            
        file_path = None
        
        # Auto-generate title if not provided
        if title is None:
            title = f"TMD Data - {mode.upper()} Visualization"
            
    return height_map, file_path, title

def _handle_seaborn_profile(
    height_map, profile_row, title, output, kwargs, ui_module, io_module, auto_open
):
    """Handle special case for seaborn profile visualization."""
    try:
        from tmd.plotters.seaborn import SeabornProfilePlotter
        
        # Initialize the profile plotter
        profile_plotter = SeabornProfilePlotter()
        
        # Set default profile row if not specified
        if profile_row is None:
            profile_row = height_map.shape[0] // 2
        
        # Ensure row is within bounds
        if profile_row < 0 or profile_row >= height_map.shape[0]:
            ui_module.print_error(f"Profile row {profile_row} out of bounds (max: {height_map.shape[0]-1})")
            return False
        
        # Extract the profile data
        profile_data = height_map[profile_row, :]
        
        # Create the profile plot
        fig = profile_plotter.plot_profile(
            profile_data,
            title=title or f"Height Profile (Row {profile_row})",
            show_markers=kwargs.get("show_markers", True),
            show_grid=kwargs.get("show_grid", True),
            **kwargs
        )
        
        # Save the visualization
        if output is not None:
            # Ensure directory exists
            output.parent.mkdir(parents=True, exist_ok=True)
            
            # Save the figure
            fig.savefig(str(output), bbox_inches='tight', dpi=300)
            ui_module.print_success(f"Visualization saved to {output}")
            
            # Only auto-open if specifically requested
            if auto_open:
                io_module.auto_open_file(output)
        
        return True
    except Exception as e:
        ui_module.print_warning(f"Error with seaborn profile plotter: {e}. Falling back to standard method.")
        return None  # Continue with standard visualization approach

def _try_fallback_visualization(
    height_map: np.ndarray,
    mode: str,
    plotter: str,
    output: Optional[Path],
    profile_row: Optional[int],
    title: Optional[str],
    colormap: str,
    z_scale: float,
    show_axes: bool,
    transparent: bool,
    **kwargs
) -> bool:
    """
    Attempt a fallback visualization if the plotter factory method fails.
    
    Args:
        height_map: The height map data to visualize
        mode: Visualization mode ("2d", "3d", "profile", etc.)
        plotter: The original plotter that failed
        output: Output file path
        profile_row: Row index for profile visualization
        title: Plot title
        colormap: Colormap name
        z_scale: Z-axis scaling factor
        show_axes: Whether to show axes
        transparent: Whether to use transparent background
        **kwargs: Additional options
        
    Returns:
        True if fallback was successful, False otherwise
    """
    ui_module = _get_ui_module()
    io_module = _get_io_module()
    
    ui_module.print_warning(f"Attempting fallback visualization using built-in methods...")
    
    try:
        # Try matplotlib as a fallback
        import matplotlib.pyplot as plt
        
        if mode == "3d":
            try:
                from mpl_toolkits.mplot3d import Axes3D
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection="3d")
                
                # Create meshgrid for 3D plot
                rows, cols = height_map.shape
                x = np.arange(cols)
                y = np.arange(rows)
                x, y = np.meshgrid(x, y)
                
                # Create surface plot
                surf = ax.plot_surface(
                    x, y, height_map * z_scale,
                    cmap=colormap,
                    linewidth=0,
                    antialiased=True
                )
                
                # Add colorbar
                fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
                
                # Set labels and title
                if title:
                    ax.set_title(title)
                if show_axes:
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Height')
                else:
                    ax.set_axis_off()
                    
            except ImportError:
                # Fall back to 2D if 3D isn't available
                ui_module.print_warning("3D plotting not available - falling back to 2D")
                fig, ax = plt.subplots(figsize=(10, 8))
                im = ax.imshow(height_map, cmap=colormap)
                plt.colorbar(im)
                if title:
                    ax.set_title(title)
                if not show_axes:
                    ax.set_axis_off()
        
        elif mode == "profile":
            # Set default profile row if not specified
            if profile_row is None:
                profile_row = height_map.shape[0] // 2
            
            # Create profile plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(height_map[profile_row, :])
            
            if title:
                ax.set_title(title)
            else:
                ax.set_title(f"Height Profile (Row {profile_row})")
                
            if show_axes:
                ax.set_xlabel('Column')
                ax.set_ylabel('Height')
                ax.grid(True, linestyle='--', alpha=0.7)
            else:
                ax.set_axis_off()
        
        else:  # Default to 2D
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(height_map, cmap=colormap)
            plt.colorbar(im)
            if title:
                ax.set_title(title)
            if not show_axes:
                ax.set_axis_off()
        
        # Save the figure if an output path is provided
        if output:
            # Ensure directory exists
            output.parent.mkdir(parents=True, exist_ok=True)
            
            # Save figure
            plt.savefig(
                output,
                transparent=transparent,
                bbox_inches='tight',
                dpi=300
            )
            
            ui_module.print_success(f"Fallback visualization saved to {output}")
            
            # Auto-open the file if requested
            if kwargs.get("auto_open", False):
                io_module.auto_open_file(output)
        
        plt.close(fig)
        return True
    
    except Exception as e:
        ui_module.print_error(f"Fallback visualization failed: {e}")
        return False
