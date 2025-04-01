"""
Polyscope-based visualization for TMD data.

This module provides 3D visualization capabilities using Polyscope.
It implements both BasePlotter and BaseSequencePlotter interfaces for
consistent integration with the TMD plotting framework.
"""

import os
import sys
import logging
import tempfile
import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Union
from pathlib import Path

# Import utility functions from TMD utils
from tmd.utils.utils import TMDUtils
from tmd.utils.files import TMDFileUtilities
from tmd.plotters.base import BasePlotter, BaseSequencePlotter

# Set up logger
logger = logging.getLogger(__name__)

# Check Polyscope dependencies
dependencies = ['polyscope', 'polyscope.imgui']
HAS_POLYSCOPE = all(TMDFileUtilities.import_optional_dependency(dep) is not None for dep in dependencies)

# Lazy-import modules
ps = TMDFileUtilities.import_optional_dependency('polyscope')
psim = TMDFileUtilities.import_optional_dependency('polyscope.imgui')


class PolyscopePlotter(BasePlotter, BaseSequencePlotter):
    """Class for creating interactive 3D visualizations of height maps and sequences using Polyscope."""
    
    NAME = "polyscope"
    DEFAULT_COLORMAP = "viridis"
    SUPPORTED_MODES = ["3d", "point_cloud", "mesh"]
    REQUIRED_DEPENDENCIES = ["polyscope", "polyscope.imgui"]
    
    def __init__(self, is_sequence: bool = False):
        """
        Initialize the Polyscope plotter.
        
        Args:
            is_sequence: Whether the plotter is being initialized for sequence visualization
        """
        super().__init__()
        
        if not HAS_POLYSCOPE:
            raise ImportError("Polyscope is required for this plotter and is not available. "
                            "Install it with: pip install polyscope")
        
        # Initialize polyscope if not already initialized
        try:
            self.is_sequence = is_sequence
            self.current_heightmap = None
            self.current_mesh = None
            self.screenshot_path = None
            
            # Initialize Polyscope with default settings
            if not ps.is_initialized():
                ps.init()
                
                # Set default rendering options
                ps.set_ground_plane_mode("shadow")
                ps.set_program_name("TMD Polyscope Visualizer")
                
                # Set default camera parameters
                ps.set_up_dir("z_up")
                
            # Configure headless rendering if needed
            if os.environ.get('TMD_HEADLESS', '0') == '1':
                ps.set_errors_throw_exceptions(True)
                
        except Exception as e:
            logger.error(f"Failed to initialize Polyscope: {e}")
            raise ImportError(f"Failed to initialize Polyscope: {e}")
    
    def plot(self, height_map: np.ndarray, **kwargs) -> Any:
        """
        Plot a TMD height map using Polyscope.
        
        Args:
            height_map: 2D numpy array representing the height map
            **kwargs: Additional options including:
                - mode: Visualization mode ('3d', 'point_cloud', 'mesh')
                - title: Plot title
                - colormap: Colormap name
                - z_scale: Z-axis scaling factor
                - smooth_shade: Whether to use smooth shading
                - wireframe: Whether to show wireframe
                - edge_width: Width of wireframe edges
                - material: Material name ('wax', 'clay', 'plastic', etc.)
                
        Returns:
            Dictionary with Polyscope visualization state
        """
        # Extract parameters
        mode = kwargs.get("mode", "mesh").lower()
        title = kwargs.get("title", "TMD Height Map")
        colormap = kwargs.get("colormap", self.DEFAULT_COLORMAP)
        z_scale = kwargs.get("z_scale", 1.0)
        smooth_shade = kwargs.get("smooth_shade", True)
        wireframe = kwargs.get("wireframe", False)
        edge_width = kwargs.get("edge_width", 1.0)
        material = kwargs.get("material", "wax")
        show = kwargs.get("show", False)
        
        # Store the current height map
        self.current_heightmap = height_map
        
        # Generate appropriate vertices and faces for the mesh
        if mode == "point_cloud":
            viz_obj = self._create_point_cloud(height_map, z_scale, colormap, title)
        elif mode == "3d" or mode == "mesh":
            viz_obj = self._create_surface_mesh(
                height_map, z_scale, colormap, title, 
                smooth_shade, wireframe, edge_width, material
            )
        else:
            raise ValueError(f"Unsupported mode: {mode}. Use 'mesh', 'point_cloud', or '3d'.")
        
        # Store the current mesh
        self.current_mesh = viz_obj
        
        # Set up the view
        ps.reset_camera_to_home_view()
        
        # Show the visualization if requested
        if show:
            ps.show()
        
        # Return visualization state
        return {
            "mesh": viz_obj,
            "mode": mode,
            "colormap": colormap,
            "title": title,
            "z_scale": z_scale
        }
    
    def plot_3d(self, height_map: np.ndarray, **kwargs) -> Any:
        """
        Create a 3D visualization of the height map.
        
        Args:
            height_map: 2D numpy array representing the height map
            **kwargs: Additional options including:
                - title: Plot title
                - colormap: Colormap name
                - z_scale: Z-axis scaling factor
                - smooth_shade: Whether to use smooth shading
                - wireframe: Whether to show wireframe
                - edge_width: Width of wireframe edges
                
        Returns:
            Dictionary with Polyscope visualization state
        """
        # Set default title for 3D visualization
        if "title" not in kwargs:
            kwargs["title"] = "TMD 3D Surface"
        
        # Set mode to mesh for 3D visualization
        kwargs["mode"] = "mesh"
        
        # Call plot method with updated kwargs
        return self.plot(height_map, **kwargs)
    
    def save(self, plot_obj: Any, filename: str, **kwargs) -> Optional[str]:
        """
        Save a screenshot of the Polyscope visualization.
        
        Args:
            plot_obj: Dictionary with Polyscope visualization state
            filename: Output filename
            **kwargs: Additional options including:
                - width: Screenshot width (default: 1024)
                - height: Screenshot height (default: 768)
                - transparent: Whether to use transparent background
                
        Returns:
            Filename if saved successfully, None otherwise
        """
        try:
            # Create directory if needed
            directory = os.path.dirname(os.path.abspath(filename))
            os.makedirs(directory, exist_ok=True)
            
            # Extract screenshot options
            width = kwargs.get("width", 1024)
            height = kwargs.get("height", 768)
            transparent = kwargs.get("transparent", False)
            
            # Take screenshot
            ps.reset_camera_to_home_view()
            
            if transparent:
                ps.set_transparent_render(True)
            
            # Store the screenshot path
            self.screenshot_path = filename
            
            # Register callback for the next frame
            ps.screenshot(filename, transparent=transparent, 
                        width=width, height=height)
            
            # Show polyscope (which will trigger the screenshot and then exit)
            ps.show()
            
            logger.info(f"Saved Polyscope visualization to {filename}")
            return filename
        except Exception as e:
            logger.error(f"Error saving Polyscope visualization: {e}")
            return None
    
    def visualize_sequence(self, frames: List[np.ndarray], **kwargs) -> Any:
        """
        Visualize a sequence of TMD height maps.
        
        Args:
            frames: List of 2D numpy arrays representing height maps
            **kwargs: Additional options including:
                - colormap: Colormap name
                - z_scale: Z-axis scaling factor
                - smooth_shade: Whether to use smooth shading
                - title: Visualization title
                - current_frame: Initial frame to display
                
        Returns:
            Dictionary with sequence visualization state
        """
        if not frames:
            logger.error("No frames provided for sequence visualization")
            return None
        
        # Extract parameters
        colormap = kwargs.get("colormap", self.DEFAULT_COLORMAP)
        z_scale = kwargs.get("z_scale", 1.0)
        smooth_shade = kwargs.get("smooth_shade", True)
        title = kwargs.get("title", "TMD Sequence Visualization")
        current_frame = kwargs.get("current_frame", 0)
        show = kwargs.get("show", False)
        
        # Ensure current_frame is valid
        current_frame = max(0, min(current_frame, len(frames) - 1))
        
        # Store frames
        self.frames = frames
        self.current_frame = current_frame
        
        # Create a mesh for the first frame
        viz_obj = self._create_surface_mesh(
            frames[current_frame], 
            z_scale, 
            colormap, 
            f"{title} - Frame {current_frame+1}/{len(frames)}",
            smooth_shade
        )
        
        # Define UI callback for frame selection
        def sequence_ui_callback():
            if psim.SliderInt("Frame", self.current_frame, 0, len(self.frames)-1)[1]:
                # Update mesh with new frame
                self._update_surface_mesh(
                    viz_obj,
                    self.frames[self.current_frame],
                    z_scale,
                    f"{title} - Frame {self.current_frame+1}/{len(frames)}"
                )
                
            psim.Text(f"Total Frames: {len(self.frames)}")
        
        # Register UI callback
        ps.set_user_callback(sequence_ui_callback)
        
        # Show the visualization if requested
        if show:
            ps.show()
        
        # Return visualization state
        return {
            "mesh": viz_obj,
            "frames": frames,
            "current_frame": current_frame,
            "colormap": colormap,
            "z_scale": z_scale,
            "title": title
        }
    
    def create_animation(self, frames: List[np.ndarray], **kwargs) -> Any:
        """
        Create an animation from a sequence of TMD height maps.
        
        Args:
            frames: List of 2D numpy arrays representing height maps
            **kwargs: Additional options including:
                - colormap: Colormap name
                - z_scale: Z-axis scaling factor
                - fps: Frames per second
                - output_dir: Directory for output frames
                
        Returns:
            Dictionary with animation state
        """
        if not frames:
            logger.error("No frames provided for animation")
            return None
        
        # Extract parameters
        colormap = kwargs.get("colormap", self.DEFAULT_COLORMAP)
        z_scale = kwargs.get("z_scale", 1.0)
        fps = kwargs.get("fps", 30)
        output_dir = kwargs.get("output_dir", None)
        title = kwargs.get("title", "TMD Animation")
        show_progress = kwargs.get("show_progress", True)
        
        # Create output directory if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        else:
            # Create temporary directory
            output_dir = tempfile.mkdtemp(prefix="tmd_animation_")
        
        # Initialize Polyscope in headless mode
        ps.set_errors_throw_exceptions(True)
        
        # Store frames and initialize animation state
        self.frames = frames
        self.current_frame = 0
        
        # Create visualization for the first frame
        viz_obj = self._create_surface_mesh(
            frames[0], 
            z_scale, 
            colormap, 
            f"{title} - Frame 1/{len(frames)}"
        )
        
        # Render each frame
        frame_files = []
        
        for i, frame in enumerate(frames):
            if show_progress:
                print(f"Rendering frame {i+1}/{len(frames)}...", end="\r")
                
            # Update mesh with current frame
            self._update_surface_mesh(
                viz_obj,
                frame,
                z_scale,
                f"{title} - Frame {i+1}/{len(frames)}"
            )
            
            # Save frame
            frame_file = os.path.join(output_dir, f"frame_{i:04d}.png")
            ps.screenshot(frame_file)
            frame_files.append(frame_file)
        
        if show_progress:
            print("\nAnimation rendering complete.")
            
        # Return animation state
        return {
            "mesh": viz_obj,
            "frame_files": frame_files,
            "fps": fps,
            "frames": len(frames),
            "output_dir": output_dir,
            "title": title
        }
    
    def visualize_statistics(self, stats_data: Dict[str, List[float]], **kwargs) -> Any:
        """
        Visualize statistical data from a sequence using Polyscope ImGui.
        
        Args:
            stats_data: Dictionary with metric names as keys and lists of values
            **kwargs: Additional options
                
        Returns:
            Dictionary with statistics visualization state
        """
        if not stats_data:
            logger.error("No statistics data provided for visualization")
            return None
        
        # Extract parameters
        title = kwargs.get("title", "TMD Statistics")
        metrics = kwargs.get("metrics", [k for k in stats_data.keys() if k != "timestamps"])
        show = kwargs.get("show", False)
        
        # Store stats data
        self.stats_data = stats_data
        self.selected_metric = metrics[0] if metrics else None
        
        # Create a simple point cloud for visualization
        points = np.array([[0, 0, 0]])
        pc = ps.register_point_cloud("stats_visualization", points)
        
        # Define UI callback for statistics display
        def stats_ui_callback():
            psim.Text(title)
            psim.Separator()
            
            # Metric selection
            if self.selected_metric and metrics:
                if psim.BeginCombo("Metric", self.selected_metric):
                    for metric in metrics:
                        is_selected = (metric == self.selected_metric)
                        if psim.Selectable(metric, is_selected)[0]:
                            self.selected_metric = metric
                        if is_selected:
                            psim.SetItemDefaultFocus()
                    psim.EndCombo()
            
            # Display statistics for selected metric
            if self.selected_metric and self.selected_metric in stats_data:
                data = stats_data[self.selected_metric]
                
                psim.Text(f"{self.selected_metric} Statistics:")
                psim.Text(f"Mean: {np.mean(data):.4f}")
                psim.Text(f"Median: {np.median(data):.4f}")
                psim.Text(f"Min: {np.min(data):.4f}")
                psim.Text(f"Max: {np.max(data):.4f}")
                psim.Text(f"Standard deviation: {np.std(data):.4f}")
                
                # Plot if ImPlot is available (would need additional dependency)
                # This is just a stub - actual plotting would require ImPlot
                psim.Text("Graph not available - ImPlot required")
        
        # Register UI callback
        ps.set_user_callback(stats_ui_callback)
        
        # Show the visualization if requested
        if show:
            ps.show()
        
        # Return statistics visualization state
        return {
            "stats_data": stats_data,
            "metrics": metrics,
            "selected_metric": self.selected_metric,
            "title": title
        }
    
    def save_figure(self, fig: Any, filename: str, **kwargs) -> Optional[str]:
        """
        Save a screenshot of the statistics visualization.
        
        Args:
            fig: Statistics visualization state dictionary
            filename: Output filename
            **kwargs: Additional options
                
        Returns:
            Filename if saved successfully, None otherwise
        """
        # This is essentially the same as the save method
        return self.save(fig, filename, **kwargs)
    
    def _create_point_cloud(self, height_map: np.ndarray, z_scale: float, 
                          colormap: str, title: str) -> Any:
        """Create a point cloud visualization from a height map."""
        # Generate points
        h, w = height_map.shape
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Normalize coordinates to [-1, 1]
        x = (x.flatten() / w) * 2 - 1
        y = (y.flatten() / h) * 2 - 1
        z = height_map.flatten() * z_scale
        
        # Create point cloud
        points = np.column_stack((x, y, z))
        
        # Register with Polyscope
        pc = ps.register_point_cloud(title, points)
        
        # Add height as a scalar quantity
        pc.add_scalar_quantity("height", z, enabled=True, cmap=colormap)
        
        return pc
    
    def _create_surface_mesh(self, height_map: np.ndarray, z_scale: float, colormap: str,
                           title: str, smooth_shade: bool = True, wireframe: bool = False,
                           edge_width: float = 1.0, material: str = "wax") -> Any:
        """Create a surface mesh visualization from a height map."""
        # Create vertices and faces for surface mesh
        h, w = height_map.shape
        
        # Create vertex positions
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Normalize coordinates to [-1, 1]
        x = (x.flatten() / w) * 2 - 1
        y = (y.flatten() / h) * 2 - 1
        z = height_map.flatten() * z_scale
        
        vertices = np.column_stack((x, y, z))
        
        # Create faces (triangulation of the grid)
        faces = []
        for i in range(h - 1):
            for j in range(w - 1):
                v0 = i * w + j
                v1 = i * w + (j + 1)
                v2 = (i + 1) * w + j
                v3 = (i + 1) * w + (j + 1)
                
                faces.append([v0, v1, v3])
                faces.append([v0, v3, v2])
        
        faces = np.array(faces)
        
        # Register with Polyscope
        surface = ps.register_surface_mesh(title, vertices, faces, smooth_shade=smooth_shade)
        
        # Add height as a scalar quantity
        surface.add_scalar_quantity("height", z, enabled=True, cmap=colormap)
        
        # Set rendering options
        if wireframe:
            surface.set_edge_width(edge_width)
            surface.set_edge_color((0.8, 0.8, 0.8))
            surface.set_wireframe(True)
        
        if material:
            surface.set_material(material)
        
        return surface
    
    def _update_surface_mesh(self, surface, height_map: np.ndarray, z_scale: float, title: str) -> None:
        """Update an existing surface mesh with new height data."""
        # Update heights
        h, w = height_map.shape
        z = height_map.flatten() * z_scale
        
        # Update positions
        old_positions = np.array(surface.get_vertices())
        new_positions = old_positions.copy()
        new_positions[:, 2] = z  # Update Z coordinates only
        
        # Update the mesh
        surface.update_vertex_positions(new_positions)
        
        # Update the scalar quantity
        surface.add_scalar_quantity("height", z, enabled=True)
        
        # Update the name
        surface.set_name(title)
