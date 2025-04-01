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
    **kwargs
) -> bool:
    """
    Create a visualization of TMD data.
    
    Args:
        tmd_file_or_data: Path to TMD file or TMD object
        mode: Visualization mode ("2d", "3d", "profile", etc.)
        plotter: Plotter name ("matplotlib", "plotly", etc.)
        output: Output file path
        profile_row: Row index for profile visualization
        title: Plot title
        colormap: Colormap name
        z_scale: Z-axis scaling factor for 3D visualizations
        show_axes: Whether to show axes
        transparent: Whether to use transparent background
        use_cache: Whether to use cached data (if available)
        **kwargs: Additional options for plotter
        
    Returns:
        True if successful, False otherwise
    """
    io_module = _get_io_module()
    ui_module = _get_ui_module()
    
    try:
        # Load data if a path was provided
        if isinstance(tmd_file_or_data, Path):
            tmd_data = io_module.load_tmd_file(
                tmd_file_or_data, 
                with_console_status=True, 
                use_cache=use_cache
            )
            if tmd_data is None:
                return False
            
            height_map = tmd_data.height_map()
            file_path = tmd_file_or_data
            
            # Auto-generate title if not provided
            if title is None:
                title = f"{file_path.stem} - {mode.upper()} Visualization"
        else:
            # Assume it's already a TMD object
            tmd_data = tmd_file_or_data
            height_map = tmd_data.height_map()
            file_path = None
            
            # Auto-generate title if not provided
            if title is None:
                title = f"TMD Data - {mode.upper()} Visualization"
        
        # Determine output path if not specified
        if output is None and file_path is not None:
            output = io_module.get_output_filename(
                file_path, 
                plotter, 
                viz_type=mode,
                subdir="visualizations"
            )
            
        # Create plotter instance using the factory
        with ui_module.console.status(f"Creating {mode} visualization with {plotter}..."):
            # Create plotter instance
            plotter_instance = TMDPlotterFactory.create_plotter(plotter)
            
            # Special handling for profile mode
            if mode == "profile":
                if profile_row is None:
                    profile_row = height_map.shape[0] // 2
                
                # Ensure row is within bounds
                if profile_row < 0 or profile_row >= height_map.shape[0]:
                    ui_module.print_error(f"Profile row {profile_row} out of bounds (max: {height_map.shape[0]-1})")
                    return False
                
                # Create profile plot
                fig = plotter_instance.plot_profile(
                    height_map,
                    profile_row=profile_row,
                    title=title or f"Height Profile (Row {profile_row})",
                    colormap=colormap,
                    show_axes=show_axes,
                    transparent=transparent,
                    **kwargs
                )
            
            # 3D surface plot
            elif mode == "3d":
                fig = plotter_instance.plot_3d(
                    height_map,
                    title=title,
                    colormap=colormap,
                    z_scale=z_scale,
                    show_axes=show_axes,
                    transparent=transparent,
                    **kwargs
                )
            
            # Default to 2D plot
            else:
                fig = plotter_instance.plot_2d(
                    height_map,
                    title=title,
                    colormap=colormap,
                    show_axes=show_axes,
                    transparent=transparent,
                    **kwargs
                )
            
            # Save the visualization
            if output is not None:
                # Ensure directory exists
                output.parent.mkdir(parents=True, exist_ok=True)
                plotter_instance.save(fig, str(output))
                ui_module.print_success(f"Visualization saved to {output}")
                
            # Show the visualization if no output path is specified
            else:
                plotter_instance.show(fig)
                ui_module.print_success("Visualization displayed")
            
            return True
            
    except Exception as e:
        ui_module.print_error(f"Error creating visualization: {e}")
        logger.error(f"Visualization error: {e}", exc_info=True)
        
        # Fallback to direct implementation if factory fails
        try:
            if plotter == "matplotlib":
                return _create_matplotlib_visualization(
                    height_map, 
                    mode, 
                    output, 
                    profile_row, 
                    title, 
                    colormap, 
                    z_scale, 
                    show_axes, 
                    transparent, 
                    **kwargs
                )
            elif plotter == "plotly":
                return _create_plotly_visualization(
                    height_map, 
                    mode, 
                    output, 
                    profile_row, 
                    title, 
                    colormap, 
                    z_scale, 
                    show_axes, 
                    transparent, 
                    **kwargs
                )
            else:
                ui_module.print_error(f"Visualization with {plotter} not implemented in fallback mode")
                return False
        except Exception as fallback_error:
            ui_module.print_error(f"Fallback visualization also failed: {fallback_error}")
            return False

def _create_matplotlib_visualization(
    height_map: np.ndarray,
    mode: str,
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
    Create visualization with matplotlib.
    
    Helper function used as fallback when TMDPlotterFactory is not available.
    """
    ui_module = _get_ui_module()
    
    try:
        import matplotlib.pyplot as plt
        from matplotlib import cm
        
        if mode == "3d":
            from mpl_toolkits.mplot3d import Axes3D
            
            with ui_module.console.status("Creating 3D visualization..."):
                # Create 3D surface plot
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                
                # Create coordinate grid
                rows, cols = height_map.shape
                x = np.arange(0, cols, 1)
                y = np.arange(0, rows, 1)
                x, y = np.meshgrid(x, y)
                
                # Plot surface
                surf = ax.plot_surface(
                    x, y, height_map * z_scale,
                    cmap=colormap,
                    linewidth=0.2,
                    antialiased=True
                )
                
                # Set title and labels
                if title:
                    ax.set_title(title)
                
                # Show/hide axes
                if not show_axes:
                    ax.set_axis_off()
                
                # Add colorbar
                fig.colorbar(surf, shrink=0.5, aspect=5)
                
        elif mode == "profile":
            with ui_module.console.status("Creating profile visualization..."):
                # Get profile data
                if profile_row is None:
                    profile_row = height_map.shape[0] // 2
                
                profile_data = height_map[profile_row, :]
                
                # Create figure
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Plot profile
                ax.plot(profile_data, linewidth=2)
                ax.fill_between(
                    range(len(profile_data)), 
                    profile_data.min(), 
                    profile_data, 
                    alpha=0.3
                )
                
                # Set title and labels
                if title:
                    ax.set_title(title)
                else:
                    ax.set_title(f"Height Profile (Row {profile_row})")
                
                ax.set_xlabel("Column Index")
                ax.set_ylabel("Height")
                ax.grid(True, linestyle='--', alpha=0.7)
                
                # Show/hide axes
                if not show_axes:
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                
        else:  # Default to 2D
            with ui_module.console.status("Creating 2D visualization..."):
                # Create figure
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Create image plot
                im = ax.imshow(height_map, cmap=colormap)
                
                # Add colorbar
                plt.colorbar(im)
                
                # Set title
                if title:
                    ax.set_title(title)
                    
                # Show/hide axes
                if not show_axes:
                    ax.set_axis_off()
        
        # Save or show the figure
        if output is not None:
            # Ensure directory exists
            output.parent.mkdir(parents=True, exist_ok=True)
            
            # Save figure
            plt.savefig(
                output, 
                bbox_inches='tight', 
                transparent=transparent, 
                dpi=300
            )
            plt.close(fig)
            ui_module.print_success(f"Visualization saved to {output}")
        else:
            plt.tight_layout()
            plt.show()
            ui_module.print_success("Visualization displayed")
            
        return True
        
    except Exception as e:
        ui_module.print_error(f"Error creating matplotlib visualization: {e}")
        logger.error(f"Matplotlib visualization error: {e}", exc_info=True)
        return False

def _create_plotly_visualization(
    height_map: np.ndarray,
    mode: str,
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
    Create visualization with plotly.
    
    Helper function used as fallback when TMDPlotterFactory is not available.
    """
    ui_module = _get_ui_module()
    
    try:
        import plotly.graph_objects as go
        
        if mode == "3d":
            with ui_module.console.status("Creating 3D visualization..."):
                # Create 3D surface
                fig = go.Figure(data=[go.Surface(
                    z=height_map * z_scale,
                    colorscale=colormap,
                )])
                
                # Update layout
                fig.update_layout(
                    title=title,
                    autosize=True,
                    scene=dict(
                        aspectratio=dict(x=1, y=1, z=0.5),
                        xaxis=dict(showticklabels=show_axes, showaxeslabels=show_axes, title=''),
                        yaxis=dict(showticklabels=show_axes, showaxeslabels=show_axes, title=''),
                        zaxis=dict(showticklabels=show_axes, showaxeslabels=show_axes, title='Height')
                    )
                )
                
        elif mode == "profile":
            with ui_module.console.status("Creating profile visualization..."):
                # Get profile data
                if profile_row is None:
                    profile_row = height_map.shape[0] // 2
                
                profile_data = height_map[profile_row, :]
                x_values = list(range(len(profile_data)))
                
                # Create figure
                fig = go.Figure()
                
                # Add line trace
                fig.add_trace(go.Scatter(
                    x=x_values,
                    y=profile_data,
                    mode='lines',
                    line=dict(width=2),
                    name='Height Profile'
                ))
                
                # Add fill
                fig.add_trace(go.Scatter(
                    x=x_values,
                    y=profile_data,
                    fill='tozeroy',
                    fillcolor='rgba(0, 176, 246, 0.2)',
                    line=dict(color='rgba(255, 255, 255, 0)'),
                    showlegend=False
                ))
                
                # Update layout
                fig.update_layout(
                    title=title or f"Height Profile (Row {profile_row})",
                    xaxis=dict(
                        title='Column Index',
                        showticklabels=show_axes
                    ),
                    yaxis=dict(
                        title='Height',
                        showticklabels=show_axes
                    )
                )
                
        else:  # Default to 2D
            with ui_module.console.status("Creating 2D visualization..."):
                # Create heatmap
                fig = go.Figure(data=go.Heatmap(
                    z=height_map,
                    colorscale=colormap,
                ))
                
                # Update layout
                fig.update_layout(
                    title=title,
                    xaxis=dict(
                        showticklabels=show_axes,
                        showgrid=show_axes,
                        zeroline=show_axes
                    ),
                    yaxis=dict(
                        showticklabels=show_axes,
                        showgrid=show_axes,
                        zeroline=show_axes
                    )
                )
        
        # Save or display the figure
        if output is not None:
            # Ensure directory exists
            output.parent.mkdir(parents=True, exist_ok=True)
            
            # Determine file type based on extension
            if output.suffix.lower() in ['.html', '.htm']:
                fig.write_html(str(output))
            else:
                fig.write_image(str(output))
                
            ui_module.print_success(f"Visualization saved to {output}")
        else:
            fig.show()
            ui_module.print_success("Visualization displayed")
            
        return True
        
    except Exception as e:
        ui_module.print_error(f"Error creating plotly visualization: {e}")
        logger.error(f"Plotly visualization error: {e}", exc_info=True)
        return False
