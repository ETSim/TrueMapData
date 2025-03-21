"""
Polyscope visualization for TMD data.
Provides 3D visualizations of topography data using Polyscope.
"""

import numpy as np
import polyscope as ps
import polyscope.imgui as psim
from typing import Optional, Union, Tuple, List, Dict, Any

class PolyscopePlotter:
    """
    Plotter class for visualizing TMD data using Polyscope.
    """
    
    def __init__(self, backend: str = "openGL"):
        """
        Initialize the Polyscope plotter.
        
        Parameters
        ----------
        backend : str
            The rendering backend to use. Options include 'openGL' (default).
            For some systems, you might need to try other options or None.
        """
        self.initialized = False
        self.backend = backend
        self.height_scale = 1.0
        self.z_up = True
        self.height_maps = {}  # Store height maps for callbacks
        self.meshes = {}       # Store meshes for callbacks
        
    def initialize(self):
        """
        Initialize Polyscope.
        """
        if not self.initialized:
            try:
                # Initialize with the specified backend
                if self.backend is not None:
                    ps.init(backend=self.backend)
                else:
                    # If no backend specified, let Polyscope choose
                    ps.init()
                self.initialized = True
                
                # Set the camera to look in the z-up direction
                if self.z_up:
                    ps.set_up_dir("z_up")
                
                # Set up UI options
                ps.set_user_callback(self.ui_callback)
                ps.set_open_imgui_window_for_user_callback(True)  # Auto-open side panel
                
            except RuntimeError as e:
                print(f"Error initializing Polyscope with backend '{self.backend}'. Trying default backend.")
                print(f"Original error: {e}")
                try:
                    # Fallback to default initialization
                    ps.init()
                    self.initialized = True
                    self.backend = None  # Reset to remember we're using default
                    
                    # Set the camera to look in the z-up direction
                    if self.z_up:
                        ps.set_up_dir("z_up")
                        
                except Exception as e2:
                    raise RuntimeError(f"Failed to initialize Polyscope: {e2}")
    
    def ui_callback(self):
        """Main UI callback for Polyscope that handles all interactions."""
        psim.SetNextWindowSize((350, 500), psim.ImGuiCond_FirstUseEver)
        if psim.Begin("TMD Controls", True):
            # General app controls section
            if psim.CollapsingHeader("Display Settings", psim.ImGuiTreeNodeFlags_DefaultOpen):
                # Add global controls
                _, show_wireframe = psim.Checkbox("Show Wireframes", getattr(self, 'show_wireframes', False))
                self.show_wireframes = show_wireframe
                
                # Apply wireframe to all meshes
                for name, data in self.height_maps.items():
                    if 'mesh' in data:
                        data['mesh'].set_edge_width(1.0 if show_wireframe else 0.0)
            
            # Height map controls section
            if self.height_maps and psim.CollapsingHeader("Height Controls", psim.ImGuiTreeNodeFlags_DefaultOpen):
                psim.Text("Adjust height exaggeration:")
                for name, data in self.height_maps.items():
                    height_map = data['height_map']
                    mesh = data['mesh']
                    min_scale = data['min_scale']
                    max_scale = data['max_scale']
                    current_scale = data.get('current_scale', 1.0)
                    
                    # Create a slider for this height map
                    _, new_scale = psim.SliderFloat(f"{name}", 
                                                   current_scale, 
                                                   v_min=min_scale, 
                                                   v_max=max_scale)
                    
                    # Update the height map if scale changed
                    if new_scale != current_scale:
                        data['current_scale'] = new_scale
                        vertices = data['vertices'].copy()
                        vertices[:, 2] = height_map.flatten() * new_scale
                        mesh.update_vertex_positions(vertices)
                        data['vertices'] = vertices
                
                # Reset button
                if psim.Button("Reset All Heights"):
                    for name, data in self.height_maps.items():
                        data['current_scale'] = 1.0
                        vertices = data['vertices'].copy()
                        vertices[:, 2] = data['height_map'].flatten()
                        data['mesh'].update_vertex_positions(vertices)
                        data['vertices'] = vertices
            
            # Mesh-specific controls section
            for name, data in self.meshes.items():
                if psim.CollapsingHeader(f"Mesh: {name}"):
                    pass  # Add mesh-specific controls here if needed
            
            psim.End()
    
    def height_scale_callback(self):
        """
        Callback function to handle height scaling UI.
        """
        # Create UI controls for each height map
        for name, data in self.height_maps.items():
            height_map = data['height_map']
            mesh = data['mesh']
            min_scale = data['min_scale']
            max_scale = data['max_scale']
            current_scale = data.get('current_scale', 1.0)
            
            # Create a slider for this height map
            _, new_scale = psim.SliderFloat(f"Height Scale ({name})", 
                                           current_scale, 
                                           v_min=min_scale, 
                                           v_max=max_scale)
            
            # Update the height map if scale changed
            if new_scale != current_scale:
                data['current_scale'] = new_scale
                # Use stored vertices instead of trying to get them from the mesh
                vertices = data['vertices'].copy()
                vertices[:, 2] = height_map.flatten() * new_scale
                mesh.update_vertex_positions(vertices)
                # Store the updated vertices for future reference
                data['vertices'] = vertices
    
    def plot_point_cloud(self, 
                        points: np.ndarray, 
                        values: Optional[np.ndarray] = None,
                        point_size: float = 1.0,
                        name: str = "points",
                        enabled: bool = True) -> None:
        """
        Plot a point cloud.
        
        Parameters
        ----------
        points : np.ndarray
            The points to plot, shape (n, 3).
        values : np.ndarray, optional
            Values to color the points by, shape (n,).
        point_size : float
            Size of points. Default is 1.0.
        name : str
            Name for the point cloud. Default is "points".
        enabled : bool
            Whether to display the point cloud by default. Default is True.
        """
        self.initialize()
        point_cloud = ps.register_point_cloud(name, points)
        point_cloud.set_point_radius(point_size)
        
        if values is not None:
            point_cloud.add_scalar_quantity("values", values, enabled=enabled)
        
        return point_cloud
    
    def plot_surface(self, 
                     vertices: np.ndarray, 
                     faces: np.ndarray,
                     vertex_values: Optional[np.ndarray] = None,
                     name: str = "surface",
                     enabled: bool = True) -> None:
        """
        Plot a surface mesh.
        
        Parameters
        ----------
        vertices : np.ndarray
            Vertices of the mesh, shape (n, 3).
        faces : np.ndarray
            Faces of the mesh, shape (m, 3).
        vertex_values : np.ndarray, optional
            Values at vertices to color the surface by.
        name : str
            Name for the surface. Default is "surface".
        enabled : bool
            Whether to display the surface values by default. Default is True.
        """
        self.initialize()
        mesh = ps.register_surface_mesh(name, vertices, faces)
        
        if vertex_values is not None:
            mesh.add_scalar_quantity("values", vertex_values, enabled=enabled)
            
        return mesh
    
    def plot_height_map(self, 
                        height_map: np.ndarray, 
                        x_range: Tuple[float, float] = None,
                        y_range: Tuple[float, float] = None,
                        name: str = "height_map",
                        enabled: bool = True,
                        add_height_slider: bool = True,
                        min_scale: float = 0.0,  # Changed from 0.1 to 0.0
                        max_scale: float = 100.0) -> None:  # Changed from 5.0 to 100.0
        """
        Plot a height map as a surface.
        
        Parameters
        ----------
        height_map : np.ndarray
            Height values, shape (n, m).
        x_range : tuple, optional
            Range of x coordinates (min, max).
        y_range : tuple, optional
            Range of y coordinates (min, max).
        name : str
            Name for the height map. Default is "height_map".
        enabled : bool
            Whether to display the height map by default. Default is True.
        add_height_slider : bool
            Whether to add a height scale slider. Default is True.
        min_scale : float
            Minimum scale value for the slider. Default is 0.0.
        max_scale : float
            Maximum scale value for the slider. Default is 100.0.
        """
        self.initialize()
        
        # Store the original height map for scaling
        orig_height_map = height_map.copy()
        
        n, m = height_map.shape
        
        if x_range is None:
            x_range = (0, n-1)
        if y_range is None:
            y_range = (0, m-1)
            
        x = np.linspace(x_range[0], x_range[1], n)
        y = np.linspace(y_range[0], y_range[1], m)
        
        X, Y = np.meshgrid(x, y)
        
        # Apply height scaling
        Z = height_map * self.height_scale
        
        # Create vertices with Z as up direction
        vertices = np.vstack([X.flatten(), Y.flatten(), Z.flatten()]).T
        
        # Create faces (triangles)
        faces = []
        for i in range(n-1):
            for j in range(m-1):
                # Calculate vertex indices
                v0 = i * m + j
                v1 = i * m + (j + 1)
                v2 = (i + 1) * m + j
                v3 = (i + 1) * m + (j + 1)
                
                # Add two triangles per grid cell
                faces.append([v0, v1, v2])
                faces.append([v1, v3, v2])
                
        faces = np.array(faces)
        
        # Register surface mesh
        mesh = ps.register_surface_mesh(name, vertices, faces)
        mesh.add_scalar_quantity("height", vertices[:, 2], enabled=enabled)
        
        # Store the height map data for use in callbacks
        if add_height_slider:
            self.height_maps[name] = {
                'height_map': orig_height_map,
                'mesh': mesh,
                'min_scale': min_scale,
                'max_scale': max_scale,
                'current_scale': self.height_scale,
                'vertices': vertices.copy()  # Store the vertices here
            }
            
            # The UI callback is already set up in initialize()
        
        return mesh
    
    def plot_interactive_height_map(self,
                                    height_map: np.ndarray,
                                    x_range: Tuple[float, float] = None,
                                    y_range: Tuple[float, float] = None,
                                    name: str = "interactive_height_map",
                                    min_height_scale: float = 0.0,  # Changed from 0.1 to 0.0
                                    max_height_scale: float = 100.0,  # Changed from 5.0 to 100.0
                                    initial_scale: float = 1.0) -> None:
        """
        Plot an interactive height map with a height control slider.
        
        This is a convenience method that sets up a scene with a height map and slider controls.
        
        Parameters
        ----------
        height_map : np.ndarray
            Height values, shape (n, m).
        x_range : tuple, optional
            Range of x coordinates (min, max).
        y_range : tuple, optional
            Range of y coordinates (min, max).
        name : str
            Name for the height map. Default is "interactive_height_map".
        min_height_scale : float
            Minimum height scale value. Default is 0.0.
        max_height_scale : float
            Maximum height scale value. Default is 100.0.
        initial_scale : float
            Initial height scale value. Default is 1.0.
        """
        # Use the existing plot_height_map method with the appropriate parameters
        self.height_scale = initial_scale
        return self.plot_height_map(
            height_map,
            x_range=x_range,
            y_range=y_range,
            name=name,
            enabled=True,
            add_height_slider=True,
            min_scale=min_height_scale,
            max_scale=max_height_scale
        )
    
    def show(self) -> None:
        """
        Show the visualization window.
        """
        self.initialize()
        ps.show()
        
    def clear(self) -> None:
        """
        Clear all structures and state.
        """
        self.height_maps = {}
        self.meshes = {}
        ps.remove_all_structures()
        ps.clear_user_callback()
        
    def screenshot(self, filename: str) -> None:
        """
        Take a screenshot of the current view.
        
        Parameters
        ----------
        filename : str
            Filename to save the screenshot to.
        """
        self.initialize()
        ps.screenshot(filename)
