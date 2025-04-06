import os
import numpy as np
import logging
from typing import Optional, Tuple
from tmd.utils.files import TMDFileUtilities
from tmd.utils.utils import TMDUtils
from noise import snoise2


class TMDTerrain:
    """
    Class for generating terrain and synthetic TMD files.
    """

    @staticmethod
    def create_sample_height_map(
        width: int = 100,
        height: int = 100,
        pattern: str = "waves",
        noise_level: float = 0.05,
        z_value: float = 0.5,  # Add z_value parameter
        wave_height: float = 1.0,  # Add wave height parameter
        seed: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Create a sample height map for testing or demonstration purposes.

        Args:
            width: Width of the height map.
            height: Height of the height map.
            pattern: Type of pattern to generate ("waves", "peak", "dome", "ramp", "combined", "flat").
            noise_level: Level of random noise to add (0.0 - 1.0+).

        Returns:
            2D numpy array with the generated height map.
        """
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)

        # Create coordinate grid
        x = np.linspace(-5, 5, width)
        y = np.linspace(-5, 5, height)
        X, Y = np.meshgrid(x, y)

        # Generate pattern
        if pattern == "waves":
            # Create a more complex wave pattern with adjustable frequency and phase
            frequency = kwargs.get('frequency', 0.5)  # Lower frequency = bigger waves
            phase = kwargs.get('phase', 0.0)
            wave_scale = kwargs.get('wave_scale', 1.0)  # Scale factor for wave pattern
            
            # Primary waves
            primary = np.sin(X * frequency + phase) * np.cos(Y * frequency + phase)
            
            # Secondary waves for more detail
            secondary = np.sin(X * frequency * 2 + phase) * np.cos(Y * frequency * 2 + phase) * 0.3
            
            # Combine waves and adjust amplitude
            Z = (primary + secondary) * wave_height * wave_scale + z_value
            
            # Smooth transitions
            try:
                from scipy.ndimage import gaussian_filter
                Z = gaussian_filter(Z, sigma=kwargs.get('smoothing', 1.0))
            except ImportError:
                pass
            
            # Enhance wave peaks and troughs
            Z = np.tanh(Z * kwargs.get('contrast', 1.5)) * 0.5 + 0.5
        elif pattern == "peak":
            Z = np.exp(-(X**2 + Y**2) / 8) * 2 + z_value
        elif pattern == "dome":
            Z = (1.0 - np.sqrt(X**2 + Y**2) / 5) + z_value
            Z[Z < z_value] = z_value
        elif pattern == "ramp":
            Z = X + Y
        elif pattern == "combined":
            # Create a combination of patterns
            Z = (
                np.sin(X) * np.cos(Y)  # Wave pattern
                + np.exp(-(X**2 + Y**2) / 8) * 2  # Central peak
            )
        elif pattern == "flat":
            # Create a flat terrain with specified value
            Z = np.ones((height, width)) * z_value
        elif pattern == "random":
            # Create a random terrain (uniform distribution)
            Z = np.random.uniform(-1, 1, (height, width))
        elif pattern == "perlin":
            # noise
            Z = np.zeros((height, width))
            scale = kwargs.get('scale', 10.0)
            octaves = kwargs.get('octaves', 4)
            persistence = kwargs.get('persistence', 0.5)
            lacunarity = kwargs.get('lacunarity', 2.0)
            
            if seed is not None:
                base = seed
            else:
                base = np.random.randint(0, 1000000)
                
            for i in range(height):
                for j in range(width):
                    Z[i, j] = snoise2(
                        i / scale, 
                        j / scale, 
                        octaves=octaves, 
                        persistence=persistence, 
                        lacunarity=lacunarity,
                        base=base
                    )
        elif pattern == "fbm":
            # fractal brownian motion
            Z = np.zeros((height, width))
            for i in range(height):
                for j in range(width):
                    Z[i, j] = snoise2(i / 10.0, j / 10.0, octaves=4, persistence=0.5, lacunarity=2.0) * (
                        1 - (i + j) / (height + width)
                    )
        elif pattern == "square":
            # Create a square pattern
            Z = np.zeros((height, width))
            Z[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4] = 1.0
        elif pattern == "sawtooth":
            # Create a sawtooth pattern
            Z = np.zeros((height, width))
            for i in range(height):
                for j in range(width):
                    Z[i, j] = (i + j) % 10 / 10.0
        else:
            Z = np.zeros((height, width))

        # Calculate base amplitude to scale the noise appropriately
        base_amplitude = np.max(np.abs(Z)) if np.max(np.abs(Z)) > 0 else 1.0

        # Add random noise with consistent application
        if noise_level > 0:
            noise = np.random.normal(0, noise_level * base_amplitude, Z.shape)
            Z = Z + noise

        # Normalize to [0, 1] range
        Z_min = Z.min()
        Z_max = Z.max()
        if Z_max > Z_min:  # Avoid division by zero
            Z = (Z - Z_min) / (Z_max - Z_min)

        # Reset random seed
        if seed is not None:
            np.random.seed(None)

        return Z.astype(np.float32)

    @staticmethod
    def generate_synthetic_tmd(
        output_path: str = None,
        width: int = 100,
        height: int = 100,
        pattern: str = "combined",
        x_length: Optional[float] = None,
        y_length: Optional[float] = None,
        mmpp: Optional[float] = None,
        noise_level: float = 0.0,
        wave_height: float = 1.0,  # Add wave height parameter
        z_value: float = 0.5,  # Add z_value parameter
        comment: str = "Created by TrueMap v6",
        version: int = 2,
        **kwargs
    ) -> str:
        """Generate a synthetic TMD file."""
        if output_path is None:
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "synthetic.tmd")

        # Create height map
        height_map = TMDTerrain.create_sample_height_map(
            width=width,
            height=height,
            pattern=pattern,
            z_value=z_value,
            noise_level=noise_level,
            wave_height=wave_height,
            **kwargs
        )

        # Calculate physical dimensions from mmpp if provided
        if mmpp is not None:
            x_length = width * mmpp
            y_length = height * mmpp
        elif x_length is None or y_length is None:
            x_length = x_length or 10.0
            y_length = y_length or (x_length * height / width if width > 0 else 10.0)

        # Write TMD file
        tmd_path = TMDUtils.write_tmd_file(
            height_map=height_map,
            output_path=output_path,
            comment=comment,
            x_length=x_length,
            y_length=y_length,
            x_offset=0.0,
            y_offset=0.0,
            version=version,
            debug=True,
        )

        return tmd_path