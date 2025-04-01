import numpy as np
import polyscope as ps
import polyscope.imgui as psim
from scipy.ndimage import gaussian_filter, sobel
from noise import pnoise2, pnoise3
import logging
import matplotlib.pyplot as plt  # <-- For the 2D UV plots

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('texture_debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def plot_uv_coordinates(uv_coords, method_name="UV Mapping"):
    """
    Display a 2D scatter plot of UV coordinates using matplotlib.
    """
    plt.figure()
    plt.scatter(uv_coords[:, 0], uv_coords[:, 1], c='blue', alpha=0.7)
    plt.title(f"{method_name} UV Coordinates")
    plt.xlabel("U")
    plt.ylabel("V")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

class UVMapper:
    """
    Advanced UV mapping generator with multiple projection techniques
    """
    @staticmethod
    def cube_unwrap(vertices):
        """
        Create a more advanced UV mapping for a cube with minimal distortion
        
        Args:
            vertices (np.ndarray): Vertices of the cube
        
        Returns:
            np.ndarray: UV coordinates for each vertex
        """
        logger.info("Starting cube unwrap UV mapping")
        try:
            # Compute cube dimensions
            min_coords = np.min(vertices, axis=0)
            max_coords = np.max(vertices, axis=0)
            cube_size = max_coords - min_coords
            
            logger.debug(f"Cube dimensions - Min: {min_coords}, Max: {max_coords}")
            
            # Create UV coordinates with consideration of face orientation
            uv_coords = np.zeros((len(vertices), 2), dtype=float)
            
            # Define face mapping logic
            for i, vertex in enumerate(vertices):
                # Determine dominant axis
                abs_vertex = np.abs(vertex - min_coords)
                max_dim_index = np.argmax(abs_vertex)
                
                # Project based on dominant axis
                if max_dim_index == 0:  # X-dominant
                    uv_coords[i, 0] = (vertex[1] - min_coords[1]) / cube_size[1]
                    uv_coords[i, 1] = (vertex[2] - min_coords[2]) / cube_size[2]
                elif max_dim_index == 1:  # Y-dominant
                    uv_coords[i, 0] = (vertex[0] - min_coords[0]) / cube_size[0]
                    uv_coords[i, 1] = (vertex[2] - min_coords[2]) / cube_size[2]
                else:  # Z-dominant
                    uv_coords[i, 0] = (vertex[0] - min_coords[0]) / cube_size[0]
                    uv_coords[i, 1] = (vertex[1] - min_coords[1]) / cube_size[1]
            
            # Normalize and scale UV coordinates
            uv_min = uv_coords.min(axis=0)
            uv_max = uv_coords.max(axis=0)
            uv_coords = (uv_coords - uv_min) / (uv_max - uv_min)
            
            logger.info("Cube unwrap UV mapping completed successfully")
            return uv_coords
        
        except Exception as e:
            logger.error(f"Error in cube unwrap UV mapping: {e}", exc_info=True)
            raise

    @staticmethod
    def conformal_mapping(vertices):
        """
        Create a conformal (angle-preserving) UV mapping
        
        Args:
            vertices (np.ndarray): Vertices of the mesh
        
        Returns:
            np.ndarray: UV coordinates for each vertex
        """
        logger.info("Generating conformal UV mapping")
        try:
            # Convert to float to avoid integer division issues
            vertices_f = vertices.astype(float)
            
            min_coords = np.min(vertices_f, axis=0)
            max_coords = np.max(vertices_f, axis=0)
            
            # Project vertices onto XY plane and normalize
            uv_coords = vertices_f[:, :2] - min_coords[:2]
            denominator = (max_coords[:2] - min_coords[:2])
            
            # Avoid division by zero if min_coords == max_coords
            denominator[denominator == 0] = 1e-9
            
            uv_coords /= denominator
            logger.debug("Conformal mapping completed")
            return uv_coords
        
        except Exception as e:
            logger.error(f"Error in conformal mapping: {e}", exc_info=True)
            raise

    @staticmethod
    def noise_based_mapping(vertices, seed=42):
        """
        Create a noise-based UV mapping for more organic distortion
        
        Args:
            vertices (np.ndarray): Vertices of the mesh
            seed (int): Random seed for reproducibility
        
        Returns:
            np.ndarray: UV coordinates for each vertex
        """
        logger.info("Generating noise-based UV mapping")
        try:
            # Set random seed for reproducibility
            np.random.seed(seed)
            
            # Create base UV coordinates (using the cube unwrap)
            uv_coords = UVMapper.cube_unwrap(vertices)
            
            # Add noise to UV coordinates
            noise_scale = 0.1
            noise_x = np.random.normal(0, noise_scale, uv_coords.shape[0])
            noise_y = np.random.normal(0, noise_scale, uv_coords.shape[0])
            
            uv_coords[:, 0] += noise_x
            uv_coords[:, 1] += noise_y
            
            # Ensure coordinates stay within [0, 1]
            uv_coords = np.clip(uv_coords, 0, 1)
            
            logger.debug("Noise-based mapping completed")
            return uv_coords
        
        except Exception as e:
            logger.error(f"Error in noise-based mapping: {e}", exc_info=True)
            raise

class ProceduralTextureGenerator:
    """
    Advanced procedural texture generator with enhanced noise and control
    """
    def __init__(self, seed=42):
        """
        Initialize with optional seed for reproducibility
        """
        logger.info(f"Initializing ProceduralTextureGenerator with seed {seed}")
        
        self.seed = seed
        np.random.seed(seed)
        
        # Texture generation parameters
        self.octaves = 6
        self.persistence = 0.6
        self.lacunarity = 2.0
        self.base_scale = 1.0
        self.detail_scale = 0.5
    
    def generate_3d_noise(self, width=512, height=512, z=0.0):
        """
        Generate 3D Perlin noise with optional z coordinate
        """
        logger.info(f"Generating 3D noise with dimensions {width}x{height}")
        try:
            noise = np.zeros((height, width))
            for i in range(height):
                for j in range(width):
                    noise[i, j] = pnoise3(
                        i / width * self.lacunarity * self.base_scale, 
                        j / height * self.lacunarity * self.base_scale, 
                        z,
                        octaves=self.octaves, 
                        persistence=self.persistence, 
                        repeatx=width, 
                        repeaty=height,
                        base=self.seed
                    )
            
            # Normalize to [0, 1]
            noise = (noise - noise.min()) / (noise.max() - noise.min())
            
            logger.debug("3D noise generation completed")
            return noise
        
        except Exception as e:
            logger.error(f"Error generating 3D noise: {e}", exc_info=True)
            raise
    
    def create_colored_height_map(self, width=512, height=512):
        """
        Create a colored height map with gradient based on height
        """
        logger.info(f"Creating colored height map with dimensions {width}x{height}")
        try:
            # Generate base height map
            height_map = self.generate_3d_noise(width, height)
            
            # Create color map based on height
            color_map = np.zeros((height, width, 3))
            color_map[:, :, 0] = height_map * 0.7 + 0.3  # Red channel
            color_map[:, :, 1] = height_map * 0.6 + 0.4  # Green channel
            color_map[:, :, 2] = height_map * 0.5 + 0.5  # Blue channel
            
            logger.debug("Colored height map created successfully")
            return np.clip(color_map, 0, 1)
        
        except Exception as e:
            logger.error(f"Error creating colored height map: {e}", exc_info=True)
            raise
    
    def create_bump_map(self, width=512, height=512):
        """
        Generate a bump map from height map
        """
        logger.info(f"Creating bump map with dimensions {width}x{height}")
        try:
            # Generate base height map
            height_map = self.generate_3d_noise(width, height)
            
            # Compute derivatives using Sobel filters
            sx = sobel(height_map, axis=0, mode='constant')
            sy = sobel(height_map, axis=1, mode='constant')
            
            # Convert to bump map (normal map)
            bump_map = np.zeros((height, width, 3))
            bump_map[:, :, 0] = sx  # Red channel (X derivative)
            bump_map[:, :, 1] = sy  # Green channel (Y derivative)
            bump_map[:, :, 2] = 1.0  # Blue channel (height)
            
            # Normalize
            norm = np.sqrt(bump_map[:, :, 0]**2 + 
                           bump_map[:, :, 1]**2 + 
                           bump_map[:, :, 2]**2)
            bump_map /= norm[:, :, np.newaxis]
            
            # Map to [0, 1] range for visualization
            bump_map = (bump_map + 1) / 2
            
            logger.debug("Bump map created successfully")
            return np.clip(bump_map, 0, 1)
        
        except Exception as e:
            logger.error(f"Error creating bump map: {e}", exc_info=True)
            raise
    
    def create_displacement_map(self, width=512, height=512):
        """
        Create an exaggerated displacement map
        """
        logger.info(f"Creating displacement map with dimensions {width}x{height}")
        try:
            # Generate base height map
            displacement = self.generate_3d_noise(width, height)
            
            # Additional non-linear processing for more dramatic displacement
            displacement = np.power(displacement, 2.0)
            
            # Create 3-channel map
            displacement_map = np.zeros((height, width, 3))
            for c in range(3):
                displacement_map[:, :, c] = displacement
            
            logger.debug("Displacement map created successfully")
            return np.clip(displacement_map, 0, 1)
        
        except Exception as e:
            logger.error(f"Error creating displacement map: {e}", exc_info=True)
            raise

def create_cube_mesh():
    """
    Create a cube mesh with detailed vertex positioning
    """
    logger.info("Creating cube mesh")
    # Vertices with more interesting positioning
    vertices = np.array([
        # Bottom face
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        # Top face
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ])

    # Triangulated faces
    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # Bottom
        [4, 5, 6], [4, 6, 7],  # Top
        [0, 4, 7], [0, 7, 3],  # Front
        [1, 5, 6], [1, 6, 2],  # Back
        [0, 1, 5], [0, 5, 4],  # Left
        [3, 2, 6], [3, 6, 7]   # Right
    ])

    logger.debug(f"Cube mesh created with {len(vertices)} vertices and {len(faces)} faces")
    return vertices, faces

def main():
    logger.info("Starting main application")
    try:
        # Initialize Polyscope
        ps.init()
        
        # Create texture generator
        texture_gen = ProceduralTextureGenerator(seed=42)
        
        # UV Mapping options
        uv_mapping_methods = [
            "Cube Unwrap", 
            "Conformal Mapping", 
            "Noise-Based Mapping"
        ]
        current_uv_method_index = 0
        
        # Create cube mesh
        vertices, faces = create_cube_mesh()

        # Global variables for UI
        texture_options = ["color", "bump", "displacement"]
        current_texture_index = 0
        
        # Texture generation
        width, height = 512, 512
        
        logger.info("Generating textures")
        texture_maps = {
            "color": texture_gen.create_colored_height_map(width, height),
            "bump": texture_gen.create_bump_map(width, height),
            "displacement": texture_gen.create_displacement_map(width, height)
        }
        logger.debug("Textures generated successfully")
        
        # Register the mesh with Polyscope
        ps_mesh = ps.register_surface_mesh("procedural_texture_cube", vertices, faces)
        
        # Create initial UV coordinates (Cube Unwrap)
        uv_coords = UVMapper.cube_unwrap(vertices)
        ps_mesh.add_parameterization_quantity(
            "cube_uv", 
            uv_coords, 
            defined_on='vertices', 
            enabled=True,
            viz_style='grid'
        )

        # Add textures to the mesh
        for name, texture in texture_maps.items():
            ps_mesh.add_color_quantity(
                f"{name}_texture", 
                texture, 
                defined_on='texture', 
                param_name="cube_uv", 
                enabled=(name == "color")  # Enable color texture by default
            )

        # Callback to handle Polyscope UI
        def callback():
            nonlocal current_texture_index, current_uv_method_index, uv_coords
            
            # UV Mapping method selection
            psim.TextUnformatted("UV Mapping Method")
            changed, current_uv_method_index = psim.Combo(
                "Select UV Mapping", 
                current_uv_method_index, 
                uv_mapping_methods
            )
            
            # Regenerate UV mapping if method changed
            if changed:
                logger.info(f"Changing UV mapping method to {uv_mapping_methods[current_uv_method_index]}")
                
                # Select appropriate UV mapping method
                if current_uv_method_index == 0:
                    uv_coords = UVMapper.cube_unwrap(vertices)
                elif current_uv_method_index == 1:
                    uv_coords = UVMapper.conformal_mapping(vertices)
                else:
                    uv_coords = UVMapper.noise_based_mapping(vertices)
                
                # Update parameterization
                ps_mesh.add_parameterization_quantity(
                    "cube_uv", 
                    uv_coords, 
                    defined_on='vertices', 
                    enabled=True,
                    viz_style='grid'
                )
                
                # **Plot UV coordinates in Matplotlib**
                plot_uv_coordinates(uv_coords, method_name=uv_mapping_methods[current_uv_method_index])
            
            # Texture selection
            psim.TextUnformatted("Texture Type")
            changed, current_texture_index = psim.Combo(
                "Select Texture", 
                current_texture_index, 
                texture_options
            )
            
            # Update active texture if changed
            if changed:
                logger.info(f"Changing texture to {texture_options[current_texture_index]}")
                # Remove existing texture quantities
                for name in texture_options:
                    try:
                        ps_mesh.remove_quantity(f"{name}_texture")
                    except Exception as e:
                        logger.debug(f"Could not remove texture {name}_texture: {e}")
                
                # Add the selected texture
                current_texture = texture_maps[texture_options[current_texture_index]]
                ps_mesh.add_color_quantity(
                    f"{texture_options[current_texture_index]}_texture", 
                    current_texture, 
                    defined_on='texture', 
                    param_name="cube_uv", 
                    enabled=True
                )
        
        # Set user callback for interactive UI
        ps.set_user_callback(callback)
        
        # Global visualization settings
        ps.set_ground_plane_mode("none")
        ps.set_background_color([0.1, 0.1, 0.2])
        ps_mesh.set_transparency(0.8)
        ps_mesh.set_edge_width(1.5)
        ps.look_at((2, 2, 2), (0.5, 0.5, 0.5))
        
        logger.info("Showing Polyscope visualization")
        ps.show()
    
    except Exception as e:
        logger.error(f"Critical error in main application: {e}", exc_info=True)

if __name__ == "__main__":
    main()
