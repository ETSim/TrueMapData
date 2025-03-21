"""
Demonstration of Polyscope visualization capabilities with TMD.

This script demonstrates various ways to visualize topographical data
using the Polyscope integration in TMD.
"""

import numpy as np
import os
import sys
import argparse

# Add parent directory to path to allow imports from tmd package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import polyscope as ps
    import polyscope.imgui as psim
    HAS_POLYSCOPE = True
except ImportError:
    HAS_POLYSCOPE = False
    print("Polyscope not found. Please install with: pip install polyscope")
    print("This demo requires Polyscope to run.")
    sys.exit(1)

try:
    from tmd.plotters.polyscope import PolyscopePlotter
    HAS_TMD_POLYSCOPE = True
except ImportError:
    HAS_TMD_POLYSCOPE = False
    print("TMD Polyscope integration not found.")
    print("Please make sure the tmd.plotters.polyscope module is available.")
    sys.exit(1)

from tmd import TMD
from tmd.utils.utils import create_sample_height_map


def create_crater_surface(size=100, radius=30, depth=5, noise_level=0.5):
    """Create a synthetic crater-like surface."""
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    
    # Base height map with a crater
    D = np.sqrt(X**2 + Y**2)
    Z = np.zeros_like(D)
    
    # Create the crater rim
    crater_mask = D < (radius/size)
    rim_mask = (D > (radius/size)*0.8) & (D < (radius/size))
    
    # Create the crater
    Z[crater_mask] = -depth * (1 - (D[crater_mask]/(radius/size))**2)
    Z[rim_mask] += depth * 0.5  # Add a rim
    
    # Add some random noise for realism
    if noise_level > 0:
        Z += np.random.normal(0, noise_level, Z.shape)
    
    # Add some mountains in the background
    mountain_height = depth * 0.8
    for _ in range(5):
        x0, y0 = np.random.uniform(-0.8, 0.8, 2)
        mountain_radius = np.random.uniform(0.1, 0.3)
        mountain_mask = ((X - x0)**2 + (Y - y0)**2) < mountain_radius**2
        mountain_distance = np.sqrt((X - x0)**2 + (Y - y0)**2)
        mountain_shape = mountain_height * (1 - (mountain_distance[mountain_mask]/mountain_radius)**2)
        Z[mountain_mask] += mountain_shape
    
    return Z


def demonstrate_height_map():
    """Demonstrate visualization of a height map."""
    print("Creating a synthetic crater surface...")
    Z = create_crater_surface(size=200, radius=60, depth=10, noise_level=0.8)
    
    print("Visualizing the crater surface as a height map...")
    plotter = PolyscopePlotter(backend=None)
    
    # Plot the height map with custom ranges
    mesh = plotter.plot_height_map(
        Z, 
        x_range=(-1, 1), 
        y_range=(-1, 1), 
        name="crater_surface",
        enabled=True,
        add_height_slider=True,
        min_scale=0.0,
        max_scale=100.0
    )
    
    print("\nHeight map visualization active.")
    print("- Use the mouse to rotate, pan, and zoom")
    print("- Use the 'Height Scale' slider to adjust vertical exaggeration")
    print("- Press 'q' to exit this visualization")
    
    plotter.show()


def demonstrate_point_cloud():
    """Demonstrate visualization of a point cloud."""
    print("Creating a synthetic point cloud...")
    
    # Create a torus point cloud
    n_points = 10000
    R, r = 1.0, 0.3  # major and minor radius
    
    # Parametric equations for a torus
    theta = np.random.uniform(0, 2*np.pi, n_points)
    phi = np.random.uniform(0, 2*np.pi, n_points)
    
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)
    
    points = np.vstack([x, y, z]).T
    
    # Add some noise
    points += np.random.normal(0, 0.01, points.shape)
    
    # Create scalar values for coloring (distance from center in XY plane)
    values = np.sqrt(x**2 + y**2)
    
    print("Visualizing the torus point cloud...")
    plotter = PolyscopePlotter(backend=None)
    
    # Plot the point cloud
    point_cloud = plotter.plot_point_cloud(
        points, 
        values=values, 
        point_size=0.01, 
        name="torus_point_cloud"
    )
    
    print("\nPoint cloud visualization active.")
    print("- Use the mouse to rotate, pan, and zoom")
    print("- Press 'q' to exit this visualization")
    
    plotter.show()


def demonstrate_surface_comparison():
    """Demonstrate comparison of two surfaces."""
    print("Creating two synthetic surfaces for comparison...")
    
    # Create two slightly different surfaces
    Z1 = create_crater_surface(size=100, radius=30, depth=5, noise_level=0.5)
    Z2 = create_crater_surface(size=100, radius=30, depth=5, noise_level=0.5)
    
    # Calculate difference between surfaces
    Z_diff = Z2 - Z1
    
    print("Visualizing surfaces for comparison...")
    plotter = PolyscopePlotter(backend=None)
    
    # Plot first surface with height slider
    mesh1 = plotter.plot_height_map(
        Z1, 
        x_range=(-1, 1), 
        y_range=(-1, 1), 
        name="surface_1",
        enabled=True,
        add_height_slider=True,
        min_scale=0.0,
        max_scale=100.0
    )
    
    # Offset second surface slightly in z-direction for better visualization
    n, m = Z2.shape
    x = np.linspace(-1, 1, n)
    y = np.linspace(-1, 1, m)
    X, Y = np.meshgrid(x, y)
    
    # Store original vertices for scaling
    orig_vertices2 = np.vstack([X.flatten(), Y.flatten(), Z2.flatten()]).T
    
    # Add offset to separate the surfaces
    vertices2 = orig_vertices2.copy()
    vertices2[:, 2] += 0.5
    
    # Create faces (triangles)
    faces = []
    for i in range(n-1):
        for j in range(m-1):
            v0 = i * m + j
            v1 = i * m + (j + 1)
            v2 = (i + 1) * m + j
            v3 = (i + 1) * m + (j + 1)
            
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])
                
    faces = np.array(faces)
    
    # Register second surface with difference values
    mesh2 = plotter.plot_surface(
        vertices2, 
        faces,
        vertex_values=Z_diff.flatten(),
        name="surface_2_with_diff"
    )
    
    # Store original vertices and surface 2 data for the callback
    surface2_data = {
        'orig_vertices': orig_vertices2,
        'mesh': mesh2,
        'Z2': Z2,
        'scale': 1.0
    }
    
    # Define callback for interactive comparison
    def comparison_callback():
        # Access plotter's height scales
        for name, data in plotter.height_maps.items():
            # This handles the first surface's height
            pass
            
        # Add a separator in the UI
        psim.Separator()
        
        # Surface 2 height control with expanded range
        _, new_scale = psim.SliderFloat("Surface 2 Height", 
                                       surface2_data['scale'], 
                                       v_min=0.0,
                                       v_max=100.0)
        
        # Update surface 2 if scale changed
        if new_scale != surface2_data['scale']:
            surface2_data['scale'] = new_scale
            new_verts = surface2_data['orig_vertices'].copy()
            new_verts[:, 2] = surface2_data['Z2'].flatten() * new_scale + 0.5
            surface2_data['mesh'].update_vertex_positions(new_verts)
            
        # Add option to visualize the difference
        _, show_diff = psim.Checkbox("Show Difference", 
                                    surface2_data['mesh'].get_enabled_quantity() == "values")
        
        if show_diff:
            surface2_data['mesh'].set_enabled_quantity("values")
        else:
            surface2_data['mesh'].set_enabled_quantity("values", False)
    
    # Register this callback in addition to the plotter's internal callback
    ps.set_user_callback(comparison_callback)
    
    print("\nSurface comparison visualization active.")
    print("- Use the mouse to rotate, pan, and zoom")
    print("- Use the sliders to adjust height scaling of each surface")
    print("- Toggle visibility of surface layers using the UI")
    print("- Press 'q' to exit this visualization")
    
    plotter.show()


def demonstrate_interactive_features():
    """Demonstrate interactive features with a complex surface."""
    print("Creating an interactive topographical visualization...")
    
    # Create a more complex terrain with multiple features
    size = 200
    x = np.linspace(-2, 2, size)
    y = np.linspace(-2, 2, size)
    X, Y = np.meshgrid(x, y)
    
    # Base terrain with hills and valleys
    Z = (
        np.sin(X*3) * np.cos(Y*2) +                    # Rolling hills
        np.exp(-((X-0.5)**2 + (Y-0.5)**2) * 2) * 2 +   # Mountain peak
        np.exp(-((X+0.8)**2 + (Y+0.8)**2) * 3) * 1.5 + # Smaller peak
        np.exp(-((X+0.5)**2 + (Y-0.7)**2) * 4) * 1.2 + # Small hill
        np.random.normal(0, 0.05, X.shape)             # Noise
    )
    
    print("Visualizing interactive terrain...")
    plotter = PolyscopePlotter(backend=None)
    
    # Plot interactive height map with slider
    mesh = plotter.plot_height_map(
        Z, 
        x_range=(-2, 2), 
        y_range=(-2, 2), 
        name="interactive_terrain",
        enabled=True,
        add_height_slider=True,
        min_scale=0.0,
        max_scale=100.0
    )
    
    # Add additional slope analysis
    slopes = np.zeros_like(Z)
    for i in range(1, Z.shape[0]-1):
        for j in range(1, Z.shape[1]-1):
            dx = (Z[i+1, j] - Z[i-1, j]) / 2
            dy = (Z[i, j+1] - Z[i, j-1]) / 2
            slopes[i, j] = np.sqrt(dx**2 + dy**2)
    
    # Flatten for the mesh
    slopes_flat = slopes.flatten()
    # Fill in boundary values to match vertices size
    slopes_flat = np.pad(slopes_flat, (0, mesh.n_vertices() - len(slopes_flat)), 'edge')
    
    # Add slope as another scalar quantity
    mesh.add_scalar_quantity("slope", slopes_flat, enabled=False)
    
    # Create visualization options
    viz_options = {
        'current_viz': "Height",
        'show_wireframe': False,
        'height_exaggeration': 1.0
    }
    
    # Create a custom UI callback
    def interactive_callback():
        psim.TextUnformatted("Interactive Terrain Controls")
        psim.Separator()
        
        # Add a dropdown for visualization type
        current_index = 0 if viz_options['current_viz'] == "Height" else 1
        changed = psim.BeginCombo("Visualization Type", viz_options['current_viz'])
        if changed:
            options = ["Height", "Slope"]
            for i, option in enumerate(options):
                _, selected = psim.Selectable(option, i == current_index)
                if selected:
                    viz_options['current_viz'] = option
                    # Update the visualization
                    if option == "Height":
                        mesh.set_enabled_quantity("height")
                        mesh.set_enabled_quantity("slope", False)
                    else:
                        mesh.set_enabled_quantity("height", False)
                        mesh.set_enabled_quantity("slope")
            psim.EndCombo()
        
        # Add a checkbox for wireframe
        _, viz_options['show_wireframe'] = psim.Checkbox("Show Wireframe", 
                                                        viz_options['show_wireframe'])
        
        # Apply wireframe setting
        mesh.set_edge_width(1.0 if viz_options['show_wireframe'] else 0.0)
        
        # Add a slider for height exaggeration with expanded range
        _, height_exag = psim.SliderFloat("Height Exaggeration", 
                                         viz_options['height_exaggeration'],
                                         v_min=0.0,
                                         v_max=100.0)
        
        # Update height if exaggeration changed
        if height_exag != viz_options['height_exaggeration']:
            viz_options['height_exaggeration'] = height_exag
            # Access and update the stored height map
            if 'interactive_terrain' in plotter.height_maps:
                data = plotter.height_maps['interactive_terrain']
                data['current_scale'] = height_exag
                # Use stored vertices instead of get_vertex_positions()
                vertices = data['vertices'].copy()
                vertices[:, 2] = data['height_map'].flatten() * height_exag
                mesh.update_vertex_positions(vertices)
                # Update stored vertices
                data['vertices'] = vertices
    
    # Set the custom callback
    ps.set_user_callback(interactive_callback)
    
    print("\nInteractive terrain visualization active.")
    print("- Use the mouse to rotate, pan, and zoom")
    print("- Use the 'Height Scale' slider to adjust vertical exaggeration")
    print("- Use 'Visualization' dropdown to switch between height and slope views")
    print("- Press 'q' to exit this visualization")
    
    plotter.show()


def demonstrate_real_heightmap(file_path=None):
    """Demonstrate visualization of a real TMD heightmap file."""
    if file_path is None:
        # Default to looking for Dime.tmd in the data directory
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'examples', 'v2')
        file_path = os.path.join(data_dir, 'Dime.tmd')
    
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        print("Please specify a valid TMD file path.")
        return
    
    print(f"Loading TMD heightmap from: {file_path}")
    
    try:
        # Load the TMD file
        tmd_data = TMD(file_path)
        
        # Get the height map data
        height_map = tmd_data.height_map()
        
        # Get metadata
        try:
            metadata = tmd_data.metadata()
            print(f"Metadata: {metadata}")
            
            # Try to get x and y ranges from metadata if available
            x_range = (0, metadata.get('x_length', height_map.shape[1] - 1))
            y_range = (0, metadata.get('y_length', height_map.shape[0] - 1))
        except:
            # If metadata retrieval fails, use array indices
            x_range = (0, height_map.shape[1] - 1)
            y_range = (0, height_map.shape[0] - 1)
        
        print(f"Loaded heightmap with shape: {height_map.shape}")
        print(f"X range: {x_range}, Y range: {y_range}")
        
        # Create a Polyscope plotter
        plotter = PolyscopePlotter(backend=None)
        
        # Plot the height map
        mesh = plotter.plot_height_map(
            height_map,
            x_range=x_range,
            y_range=y_range,
            name=os.path.basename(file_path),
            enabled=True,
            add_height_slider=True,
            min_scale=0.0,
            max_scale=100.0
        )
        
        print("\nTMD heightmap visualization active.")
        print("- Use the mouse to rotate, pan, and zoom")
        print("- Use the side panel to adjust vertical exaggeration")
        print("- Press 'q' to exit this visualization")
        
        plotter.show()
        
    except Exception as e:
        print(f"Error loading or visualizing TMD file: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function to run the demo."""
    parser = argparse.ArgumentParser(description="Polyscope TMD visualization demos")
    parser.add_argument('--demo', type=str, default='all',
                        choices=['all', 'height_map', 'point_cloud', 'comparison', 'interactive', 'real'],
                        help='Specific demo to run')
    parser.add_argument('--file', type=str, default=None,
                        help='Path to TMD file to visualize (for real demo)')
    
    args = parser.parse_args()
    
    print("="*50)
    print("TMD Polyscope Visualization Demo")
    print("="*50)
    print("This demonstration shows various ways to visualize topographical data")
    print("using Polyscope integration with TMD.\n")
    
    if args.demo == 'real' or (args.file is not None):
        print("\n" + "="*20 + " REAL TMD FILE DEMO " + "="*20)
        demonstrate_real_heightmap(args.file)
    elif args.demo == 'all':
        print("\n" + "="*20 + " HEIGHT MAP DEMO " + "="*20)
        demonstrate_height_map()
        
        print("\n" + "="*20 + " POINT CLOUD DEMO " + "="*20)
        demonstrate_point_cloud()
        
        print("\n" + "="*20 + " SURFACE COMPARISON DEMO " + "="*20)
        demonstrate_surface_comparison()
            
        print("\n" + "="*20 + " INTERACTIVE FEATURES DEMO " + "="*20)
        demonstrate_interactive_features()
    else:
        # Run the specific demo selected
        if args.demo == 'height_map':
            demonstrate_height_map()
        elif args.demo == 'point_cloud':
            demonstrate_point_cloud()
        elif args.demo == 'comparison':
            demonstrate_surface_comparison()
        elif args.demo == 'interactive':
            demonstrate_interactive_features()
    
    print("\nAll demonstrations completed!")
    print("="*50)


if __name__ == "__main__":
    # Check if Polyscope is available
    if not HAS_POLYSCOPE:
        print("ERROR: Polyscope not found. Please install with: pip install polyscope")
        sys.exit(1)
    
    # Check if TMD Polyscope integration is available
    if not HAS_TMD_POLYSCOPE:
        print("ERROR: TMD Polyscope integration not found.")
        print("Please ensure the tmd.plotters.polyscope module is available.")
        sys.exit(1)
    
    main()
