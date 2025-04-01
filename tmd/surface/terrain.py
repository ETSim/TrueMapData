import os
import numpy as np
import logging
from typing import Optional, Tuple
from tmd.utils.files import TMDFileUtilities
from tmd.utils.utils import TMDUtils


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
    ) -> np.ndarray:
        """
        Create a sample height map for testing or demonstration purposes.

        Args:
            width: Width of the height map.
            height: Height of the height map.
            pattern: Type of pattern to generate ("waves", "peak", "dome", "ramp", "combined").
            noise_level: Level of random noise to add (0.0 - 1.0+).

        Returns:
            2D numpy array with the generated height map.
        """
        # Create coordinate grid
        x = np.linspace(-5, 5, width)
        y = np.linspace(-5, 5, height)
        X, Y = np.meshgrid(x, y)

        # Generate pattern
        if pattern == "waves":
            Z = np.sin(X) * np.cos(Y)
        elif pattern == "peak":
            Z = np.exp(-(X**2 + Y**2) / 8) * 2
        elif pattern == "dome":
            Z = 1.0 - np.sqrt(X**2 + Y**2) / 5
            Z[Z < 0] = 0
        elif pattern == "ramp":
            Z = X + Y
        elif pattern == "combined":
            # Create a combination of patterns
            Z = (
                np.sin(X) * np.cos(Y)  # Wave pattern
                + np.exp(-(X**2 + Y**2) / 8) * 2  # Central peak
            )
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

        return Z.astype(np.float32)

    @staticmethod
    def generate_synthetic_tmd(
        output_path: str = None,
        width: int = 100,
        height: int = 100,
        pattern: str = "combined",
        comment: str = "Created by TrueMap v6",
        version: int = 2,
    ) -> str:
        """
        Generate a synthetic TMD file for testing or demonstration.

        Args:
            output_path: Path where to save the TMD file (default: "output/synthetic.tmd").
            width: Width of the height map.
            height: Height of the height map.
            pattern: Type of pattern for the height map.
            comment: Comment to include in the file.
            version: TMD version to write (1 or 2).

        Returns:
            Path to the created TMD file.
        """
        if output_path is None:
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "synthetic.tmd")

        # Create a sample height map with named parameters for test compatibility
        height_map = TMDTerrain.create_sample_height_map(width=width, height=height, pattern=pattern)

        # Write the height map to a TMD file using TMDUtils
        tmd_path = TMDUtils.write_tmd_file(
            height_map=height_map,
            output_path=output_path,
            comment=comment,
            x_length=10.0,
            y_length=10.0,
            x_offset=0.0,
            y_offset=0.0,
            version=version,
            debug=True,
        )

        return tmd_path