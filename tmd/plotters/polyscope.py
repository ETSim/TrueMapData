"""
Polyscope-based visualization for TMD data.

This module provides 3D visualization capabilities using Polyscope.
It implements both BasePlotter and BaseSequencePlotter interfaces for
consistent integration with the TMD plotting framework.
"""
import logging
import os
from typing import Optional, Tuple, List, Dict, Any, Union
from pathlib import Path

import numpy as np

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


class PolyscopePlotter(BasePlotter):
    """Class for creating interactive 3D visualizations of height maps using Polyscope."""
    
    NAME = "polyscope"
    DEFAULT_COLORMAP = "viridis"
    SUPPORTED_MODES = ["3d", "point_cloud", "mesh"]
    REQUIRED_DEPENDENCIES = ["polyscope", "polyscope.imgui"]
    
    def __init__(self):
        """Initialize Polyscope plotter."""
        super().__init__()
        
        if not HAS_POLYSCOPE:
            raise ImportError("Polyscope dependency missing. Install with: pip install polyscope")
        
        self.current_mesh_name = None
        self.mesh_objects = {}
        self.point_cloud_objects = {}
        
        # Initialize Polyscope if it hasn't been already
        if not ps.is_initialized():
            ps.init()
            # Fix: Use string instead of list for up direction
            ps.set_up_dir("z_up")  # Set Z as up direction
            
            # Configure default appearance
            ps.set_ground_plane_mode("shadow_only")  # Changed from "shadow" to "shadow_only"
            ps.set_ground_plane_height_factor(0)
            ps.set_screenshot_extension(".png")
            
            logger.info("Polyscope initialized")
    
    def plot(self, height_map: np.ndarray, **kwargs) -> Any:
        """
        Plot the TMD height map using Polyscope.
        
        Args:
            height_map: 2D numpy array representing the height map
            **kwargs: Additional options including:
                - mode: Plot mode - "3d" (surface), "point_cloud", or "mesh" (default: "3d") 
                - colormap: Colormap name (default: "viridis")
                - title: Plot title (default: "TMD Height Map")
                - z_scale: Scaling factor for Z-axis (default: 1.0)
                - wireframe: Whether to show wireframe in mesh mode (default: False)
                - point_size: Size of points in point_cloud mode (default: 2.0)
                
        Returns:
            A dictionary with metadata about the visualization
        """
        # Extract parameters with defaults
        mode = kwargs.get("mode", "3d").lower()
        colormap = kwargs.get("colormap", self.DEFAULT_COLORMAP)
        title = kwargs.get("title", "TMD Height Map")
        z_scale = kwargs.get("z_scale", 1.0)
        
        # Apply partial range if specified
        partial_range = kwargs.get("partial_range", None)
        if partial_range is not None:
            height_map = height_map[partial_range[0]:partial_range[1], partial_range[2]:partial_range[3]]
            logger.info(f"Partial render applied: rows {partial_range[0]}:{partial_range[1]}, "
                       f"cols {partial_range[2]}:{partial_range[3]}")
        
        # Create a unique name for this visualization
        mesh_name = f"height_map_{id(height_map)}"
        self.current_mesh_name = mesh_name
        
        # Create visualization based on mode
        if mode == "point_cloud":
            # Don't pass z_scale directly, let it be extracted from kwargs
            result = self._create_point_cloud(height_map, **kwargs)
        elif mode == "mesh":
            # Don't pass z_scale directly, let it be extracted from kwargs
            result = self._create_mesh(height_map, **kwargs)
        else:  # Default to 3D surface
            # Don't pass z_scale directly, let it be extracted from kwargs
            result = self._create_surface(height_map, **kwargs)
        
        # Show the scene if requested
        if kwargs.get("show", False):
            ps.show()
        
        return result
    
    def plot_3d(self, height_map: np.ndarray, **kwargs) -> Any:
        """Create a 3D surface visualization with Polyscope."""
        kwargs["mode"] = "3d"
        return self.plot(height_map, **kwargs)
    
    def save(self, plot_obj: Any, filename: str, **kwargs) -> Optional[str]:
        """
        Save a screenshot of the visualization.
        
        Args:
            plot_obj: Plot object returned by plot()
            filename: Output filename or path 
            **kwargs: Additional options including:
                - width: Image width in pixels (default: 1024)
                - height: Image height in pixels (default: 768)
                - transparent: Whether to save with transparent background (default: False)
                
        Returns:
            Path to saved file if successful, None otherwise
        """     
        try:
            # Create directory if needed
            directory = os.path.dirname(os.path.abspath(filename))
            os.makedirs(directory, exist_ok=True)
            
            # Get screenshot options
            width = kwargs.get("width", 1024)
            height = kwargs.get("height", 768)
            transparent = kwargs.get("transparent", False)
            
            # Set the background color for transparency
            if transparent:
                ps.set_transparency_mode('simple')
                ps.set_ground_plane_mode("none")
            
            # Update camera parameters if specified
            camera_params = kwargs.get("camera_params", None)
            if camera_params:
                ps.reset_camera_to_home_view()
                ps.set_camera_view_matrix(camera_params["view_matrix"])
                ps.set_camera_projection_matrix(camera_params["projection_matrix"])
            
            # Take screenshot
            ps.screenshot(filename, transparent=transparent, 
                         resolution=[width, height])
            logger.info(f"Screenshot saved to {filename}")
            return filename
        except Exception as e:
            logger.error(f"Error saving screenshot: {e}")
            return None
    
    def _create_point_cloud(self, height_map: np.ndarray, **kwargs) -> Any:
        """Create a point cloud visualization of the height map."""
        try:
            # Get point cloud parameters
            z_scale = kwargs.get("z_scale", 1.0)  # Extract z_scale from kwargs
            point_size = kwargs.get("point_size", 2.0)
            point_cloud_name = f"point_cloud_{self.current_mesh_name}"
            title = kwargs.get("title", "TMD Height Map")
            colormap = kwargs.get("colormap", self.DEFAULT_COLORMAP)
            
            # Create coordinates and point cloud
            rows, cols = height_map.shape
            points = []
            heights = []
            
            # Sample points from the height map
            sample_rate = kwargs.get("sample_rate", 1)
            for r in range(0, rows, sample_rate):
                for c in range(0, cols, sample_rate):
                    x = c / cols - 0.5  # Center around origin
                    y = r / rows - 0.5
                    z = height_map[r, c] * z_scale
                    points.append([x, y, z])
                    heights.append(height_map[r, c])
            
            # Convert to numpy arrays
            points = np.array(points)
            heights = np.array(heights)
            
            # Register point cloud with Polyscope
            pc = ps.register_point_cloud(point_cloud_name, points)
            pc.set_radius(point_size * 0.01)  # Set point size
            
            # Add height values as a scalar quantity
            pc.add_scalar_quantity("height", heights, 
                                  enabled=True, cmap=colormap)
            
            # Store the point cloud object
            self.point_cloud_objects[point_cloud_name] = pc
            
            # Change the window title
            ps.set_window_title(title)
            
            return {
                "point_cloud_name": point_cloud_name,
                "n_points": len(points),
                "title": title
            }
            
        except Exception as e:
            logger.error(f"Error creating point cloud: {e}")
            # Create a minimal point cloud as fallback
            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
            pc = ps.register_point_cloud("error_cloud", points)
            return {"error": str(e)}
    
    def _create_mesh(self, height_map: np.ndarray, **kwargs) -> Any:
        """Create a triangle mesh visualization of the height map."""
        try:
            # Get mesh parameters
            z_scale = kwargs.get("z_scale", 1.0)  # Extract z_scale from kwargs
            wireframe = kwargs.get("wireframe", False)
            smooth = kwargs.get("smooth", True)
            mesh_name = f"mesh_{self.current_mesh_name}"
            title = kwargs.get("title", "TMD Height Map")
            colormap = kwargs.get("colormap", self.DEFAULT_COLORMAP)
            
            # Create vertices and faces
            rows, cols = height_map.shape
            vertices = []
            faces = []
            
            # Create vertices
            for r in range(rows):
                for c in range(cols):
                    # Scale coordinates to center around origin
                    x = c / (cols - 1) - 0.5 if cols > 1 else 0
                    y = r / (rows - 1) - 0.5 if rows > 1 else 0
                    z = height_map[r, c] * z_scale
                    vertices.append([x, y, z])
            
            # Create faces (triangles)
            for r in range(rows - 1):
                for c in range(cols - 1):
                    # Get vertex indices
                    i00 = r * cols + c
                    i01 = r * cols + (c + 1)
                    i10 = (r + 1) * cols + c
                    i11 = (r + 1) * cols + (c + 1)
                    
                    # Add two triangles for each grid cell
                    faces.append([i00, i01, i11])
                    faces.append([i00, i11, i10])
            
            # Convert to numpy arrays
            vertices = np.array(vertices)
            faces = np.array(faces)
            
            # Register mesh with Polyscope
            mesh = ps.register_surface_mesh(mesh_name, vertices, faces)
            
            # Add height values as a scalar quantity on vertices
            heights = np.array([v[2] for v in vertices])
            mesh.add_scalar_quantity("height", heights, 
                                    enabled=True, cmap=colormap)
            
            # Set mesh display options
            mesh.set_enabled(True)
            mesh.set_wireframe(wireframe)
            mesh.set_smooth_shade(smooth)
            
            # Store the mesh object
            self.mesh_objects[mesh_name] = mesh
            
            # Change the window title
            ps.set_window_title(title)
            
            return {
                "mesh_name": mesh_name,
                "n_vertices": len(vertices),
                "n_faces": len(faces),
                "title": title
            }
            
        except Exception as e:
            logger.error(f"Error creating mesh: {e}")
            # Create a minimal mesh as fallback
            vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
            faces = np.array([[0, 1, 2]])
            ps.register_surface_mesh("error_mesh", vertices, faces)
            return {"error": str(e)}
    
    def _create_surface(self, height_map: np.ndarray, **kwargs) -> Any:
        """Create a surface visualization of the height map (default 3D mode)."""
        # For surfaces, we use the mesh implementation with some different defaults
        return self._create_mesh(
            height_map, 
            wireframe=kwargs.get("wireframe", False),
            smooth=kwargs.get("smooth", True),
            **kwargs
        )


class PolyscopeSequencePlotter(BaseSequencePlotter):
    """Class for creating interactive 3D visualizations of TMD sequences using Polyscope."""
    
    NAME = "polyscope"
    DEFAULT_COLORMAP = "viridis"
    SUPPORTED_MODES = ["3d", "point_cloud", "mesh"]
    REQUIRED_DEPENDENCIES = ["polyscope", "polyscope.imgui"]
    
    def __init__(self):
        """Initialize Polyscope sequence plotter."""
        super().__init__()
        
        if not HAS_POLYSCOPE:
            raise ImportError("Polyscope dependency missing. Install with: pip install polyscope")
        
        # Create a base plotter for rendering individual frames
        self.base_plotter = PolyscopePlotter()
        
        # Sequence-specific properties
        self.sequence_frames = []
        self.current_frame_index = 0
        self.sequence_name = None
        self.sequence_kwargs = {}
        self.frame_time = 0.1
        self.animation_running = False
        self.last_update_time = 0
        self.stats_data = {}
        self.stats_summary = {}
        
    def visualize_sequence(self, frames: List[np.ndarray], **kwargs) -> Any:
        """
        Visualize a sequence of TMD height maps with an interactive slider.
        
        Args:
            frames: List of 2D numpy arrays representing the sequence
            **kwargs: Additional options including:
                - mode: Visualization mode ("3d", "point_cloud", or "mesh") 
                - colormap: Colormap name
                - title: Visualization title
                - z_scale: Z-axis scaling factor
                - show: Whether to show the visualization immediately
                
        Returns:
            A dictionary with visualization metadata
        """
        if not frames:
            logger.error("No frames provided for sequence visualization")
            return None
        
        # Extract parameters with defaults
        mode = kwargs.get("mode", "3d").lower()
        colormap = kwargs.get("colormap", self.DEFAULT_COLORMAP)
        title = kwargs.get("title", "TMD Sequence")
        z_scale = kwargs.get("z_scale", 1.0)
        
        # Create a base name for sequence objects
        sequence_name = f"sequence_{id(frames)}"
        
        # Store frame data for GUI interaction
        self.sequence_frames = frames
        self.sequence_kwargs = kwargs
        self.current_frame_index = 0
        self.sequence_name = sequence_name
        
        # We'll use ImGui to create an interactive UI for the sequence
        ps.set_user_callback(lambda: self._sequence_callback(title))
        
        # Initialize with first frame
        self.base_plotter.plot(frames[0], mode=mode, colormap=colormap, title=title, 
                             z_scale=z_scale, show=False, **kwargs)
        
        # Show the visualization if requested
        if kwargs.get("show", True):
            ps.show()
        
        return {"sequence_name": sequence_name, "n_frames": len(frames)}
    
    def create_animation(self, frames: List[np.ndarray], **kwargs) -> Any:
        """
        Create an animation of a sequence of height maps.
        
        Note: This doesn't create a video file directly, but sets up an interactive
        animation in the Polyscope window that can be recorded.
        
        Args:
            frames: List of 2D numpy arrays representing the sequence
            **kwargs: Additional options
                
        Returns:
            Dictionary with animation metadata
        """
        if not frames:
            logger.error("No frames provided for animation")
            return None
        
        # Extract parameters
        fps = kwargs.get("fps", 10)
        frame_time = 1.0 / fps
        
        # Store animation data
        self.sequence_frames = frames
        self.current_frame_index = 0
        self.frame_time = frame_time
        self.animation_running = True
        self.last_update_time = 0
        
        # Setup animation callback
        ps.set_user_callback(self._animation_callback)
        
        # Initialize with first frame
        result = self.base_plotter.plot(frames[0], **kwargs)
        
        # Start the animation
        if kwargs.get("show", True):
            ps.show()
        
        return {"animation": True, "n_frames": len(frames), "fps": fps}
    
    def visualize_statistics(self, stats_data: Dict[str, List[float]], **kwargs) -> Any:
        """
        Visualize statistical data from the sequence.
        
        Note: Since Polyscope is primarily a 3D visualization tool, this creates
        a basic interface showing statistics with ImGui.
        
        Args:
            stats_data: Dictionary with metric names as keys and lists of values
            **kwargs: Additional options
                
        Returns:
            Dictionary with visualization metadata
        """
        if not stats_data:
            logger.error("No statistical data provided for visualization")
            return None
        
        # Store statistics data for GUI
        self.stats_data = stats_data
        self.stats_kwargs = kwargs
        
        # Calculate some basic statistics for display
        self.stats_summary = {}
        for metric, values in stats_data.items():
            if metric != "timestamps" and len(values) > 0:
                values_array = np.array(values)
                self.stats_summary[metric] = {
                    "mean": np.mean(values_array),
                    "min": np.min(values_array),
                    "max": np.max(values_array),
                    "std": np.std(values_array),
                }
        
        # Setup statistics UI callback
        ps.set_user_callback(self._statistics_callback)
        
        # Create a simple visualization representing the data range
        if "mean_height" in stats_data:
            # Create a simple heightmap to visualize
            size = 100
            heights = np.zeros((size, size))
            mean_heights = stats_data.get("mean_height", [0])
            min_val = np.min(mean_heights)
            max_val = np.max(mean_heights)
            
            # Create a gradient to represent the statistics
            for i in range(size):
                val = min_val + (max_val - min_val) * (i / size)
                heights[i, :] = val
            
            # Plot this representation
            self.base_plotter.plot(heights, title=kwargs.get("title", "Statistics Visualization"))
                
        # Show the visualization
        if kwargs.get("show", True):
            ps.show()
        
        return {"statistics": True, "metrics": list(self.stats_summary.keys())}
    
    def save_figure(self, fig: Any, filename: str, **kwargs) -> Optional[str]:
        """
        Save a visualization to a file.
        
        This uses the base plotter's save method.
        
        Args:
            fig: Output from visualize_sequence, create_animation, or visualize_statistics
            filename: Output filename
            **kwargs: Additional options
                
        Returns:
            Path to saved file if successful, None otherwise
        """
        return self.base_plotter.save(fig, filename, **kwargs)
    
    def _sequence_callback(self, title: str):
        """ImGui callback for sequence visualization controls."""
        if psim is None:
            return
            
        # Create a window for sequence controls
        psim.PushItemWidth(100)
        psim.Begin(f"{title} - Frame Controls", True)
        
        # Current frame selection with slider
        changed, value = psim.SliderInt("Frame", self.current_frame_index, 
                                      0, len(self.sequence_frames) - 1)
        if changed:
            self.current_frame_index = value
            # Update visualization
            self.base_plotter.plot(self.sequence_frames[value], 
                                 **self.sequence_kwargs, show=False)
        
        # Playback controls
        if psim.Button("Previous Frame"):
            self.current_frame_index = max(0, self.current_frame_index - 1)
            self.base_plotter.plot(self.sequence_frames[self.current_frame_index],
                                 **self.sequence_kwargs, show=False)
                     
        psim.SameLine()
        if psim.Button("Next Frame"):
            self.current_frame_index = min(len(self.sequence_frames) - 1,
                                         self.current_frame_index + 1)
            self.base_plotter.plot(self.sequence_frames[self.current_frame_index],
                                 **self.sequence_kwargs, show=False)
        
        # Frame info
        psim.Text(f"Frame {self.current_frame_index + 1} of {len(self.sequence_frames)}")
        
        psim.End()
    
    def _animation_callback(self):
        """Callback to update animation frames."""
        if psim is None:
            return
            
        # Create control window
        psim.Begin("Animation Controls", True)
        
        # Play/pause button
        if self.animation_running:
            if psim.Button("Pause"):
                self.animation_running = False
        else:
            if psim.Button("Play"):
                self.animation_running = True
        
        # Display current frame
        psim.Text(f"Frame: {self.current_frame_index + 1} / {len(self.sequence_frames)}")
        
        # FPS control
        changed, value = psim.SliderFloat("FPS", 1.0 / self.frame_time, 1, 30)
        if changed:
            self.frame_time = 1.0 / value
        
        psim.End()
        
        # Update frame if animation is running
        if self.animation_running:
            # Get current time
            current_time = ps.get_time()
            
            # Update frame if enough time has passed
            if current_time - self.last_update_time > self.frame_time:
                self.current_frame_index = (self.current_frame_index + 1) % len(self.sequence_frames)
                self.base_plotter.plot(self.sequence_frames[self.current_frame_index], 
                                     **self.sequence_kwargs, show=False)
                self.last_update_time = current_time
    
    def _statistics_callback(self):
        """Callback to display statistical information."""
        if psim is None:
            return
            
        # Create a window for statistics
        psim.Begin("TMD Statistics", True)
           
        # Display summary statistics
        for metric, stats in self.stats_summary.items():
            if psim.TreeNode(metric):
                psim.Text(f"Mean: {stats['mean']:.6f}")
                psim.Text(f"Min: {stats['min']:.6f}")
                psim.Text(f"Max: {stats['max']:.6f}")
                psim.Text(f"Std Dev: {stats['std']:.6f}")
                psim.TreePop()
        
        psim.End()