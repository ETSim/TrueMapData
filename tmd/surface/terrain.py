import logging
import os
from typing import Optional, Union

import numpy as np
from noise import snoise2
from numpy.random import Generator

from tmd.utils.utils import TMDUtils

logger = logging.getLogger(__name__)


# --- Ausloos–Berman Weierstrass–Mandelbrot fractal (NumPy) ---------------------------------


def _wm_sample_phases(rng: Generator, M: int, N_max: int) -> np.ndarray:
    """Uniform random phases φ_mn on [0, 2π), shape (M, N_max)."""
    return rng.uniform(0.0, 2.0 * np.pi, size=(M, N_max))


def validate_wm_params(D_f: float, gamma: float, M: int, N_max: int) -> None:
    if not (2.0 < D_f < 3.0):
        raise ValueError(f"D_f must be strictly between 2 and 3, got {D_f!r}")
    if gamma <= 1.0:
        raise ValueError(f"gamma must be > 1, got {gamma!r}")
    if M < 1:
        raise ValueError(f"M must be >= 1, got {M}")
    if N_max < 1:
        raise ValueError(f"N_max must be >= 1, got {N_max}")


def ausloos_berman_wm_grid(
    x: np.ndarray,
    y: np.ndarray,
    *,
    L: float,
    D_f: float,
    gamma: float,
    M: int,
    N_max: int,
    phases: np.ndarray,
) -> np.ndarray:
    """
    2D Ausloos–Berman W–M fractal on coordinate arrays ``x``, ``y`` (e.g. meshgrid).

    Ridge directions θ_m = 2π(m-1)/M, s_m = x cos θ_m + y sin θ_m, with

    L^{4-2D_f} ln(γ) Σ_m Σ_n γ^{(n-1)2(D_f-2)} cos(2π γ^{n-1} s_m / L - cos(π m/M) + φ_{mn}).
    """
    validate_wm_params(D_f, gamma, M, N_max)
    phases = np.asarray(phases, dtype=np.float64)
    if phases.shape != (M, N_max):
        raise ValueError(
            f"phases must have shape (M, N_max)=({M}, {N_max}), got {phases.shape}"
        )

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ndim = x.ndim
    if y.shape != x.shape:
        raise ValueError(f"x and y must have the same shape, got {x.shape} vs {y.shape}")

    prefactor = (float(L) ** (4.0 - 2.0 * D_f)) * np.log(float(gamma))

    m_idx = np.arange(1, M + 1, dtype=np.float64)
    theta = 2.0 * np.pi * (m_idx - 1.0) / float(M)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    ones_tail = (1,) * ndim
    cos_exp = cos_t.reshape(M, *ones_tail)
    sin_exp = sin_t.reshape(M, *ones_tail)
    s = cos_exp * x + sin_exp * y

    k = np.arange(N_max, dtype=np.float64)
    gamma_pow = gamma**k
    amp = gamma ** (k * 2.0 * (D_f - 2.0))
    cos_pi_m_over_M = np.cos(np.pi * m_idx / float(M))

    sp = s[:, np.newaxis, ...]
    gpow_exp = gamma_pow.reshape(1, N_max, *ones_tail)
    term1 = (2.0 * np.pi / float(L)) * sp * gpow_exp

    m_off = cos_pi_m_over_M.reshape(M, 1, *ones_tail)
    phi_exp = phases.reshape(M, N_max, *ones_tail)
    arg = term1 - m_off + phi_exp

    amp_exp = amp.reshape(1, N_max, *ones_tail)
    contrib = amp_exp * np.cos(arg)
    return prefactor * np.sum(contrib, axis=(0, 1))


def ausloos_berman_wm_scalar(
    x: float,
    y: float,
    *,
    L: float,
    D_f: float,
    gamma: float,
    M: int,
    N_max: int,
    phases: np.ndarray,
) -> float:
    """Single-point height; same model as :func:`ausloos_berman_wm_grid`."""
    z = ausloos_berman_wm_grid(
        np.array(x, dtype=np.float64),
        np.array(y, dtype=np.float64),
        L=L,
        D_f=D_f,
        gamma=gamma,
        M=M,
        N_max=N_max,
        phases=phases,
    )
    return float(z.item())


def make_phases_for_seed(seed: Union[int, np.integer, None], M: int, N_max: int) -> np.ndarray:
    """Phase table φ_mn from integer ``seed``, or non-deterministic if ``seed`` is None."""
    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(int(seed))
    return _wm_sample_phases(rng, M, N_max)


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
            pattern: Type of pattern to generate ("waves", "peak", "dome", "ramp",
                "combined", "flat", "perlin", "fbm", "wm_ab", "ausloos_berman_wm", ...).
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
            # Generate sophisticated Perlin noise with large-scale features
            Z = np.zeros((height, width))
            
            # Adjusted defaults for more interesting terrain
            scale = kwargs.get('scale', 100.0)  # Larger scale for bigger features
            octaves = kwargs.get('octaves', 8)  # Fewer octaves for clearer features
            persistence = kwargs.get('persistence', 0.5)  # Balanced persistence
            lacunarity = kwargs.get('lacunarity', 2.0)  # Standard lacunarity
            ridge_factor = kwargs.get('ridge_factor', 1.2)  # For ridge-like features
            
            if seed is not None:
                base = seed
            else:
                base = np.random.randint(0, 1000000)
            
            try:
                # Generate base terrain with large features
                for i in range(height):
                    for j in range(width):
                        noise_val = 0
                        amplitude = 1.0
                        frequency = 1.0
                        
                        # First pass: large-scale features
                        noise_val = snoise2(
                            i / (scale * 2),
                            j / (scale * 2),
                            octaves=3,
                            persistence=0.7,
                            lacunarity=1.8,
                            base=base
                        )
                        
                        # Second pass: medium details
                        noise_val += 0.5 * snoise2(
                            i / scale,
                            j / scale,
                            octaves=octaves,
                            persistence=persistence,
                            lacunarity=lacunarity,
                            base=base + 1000
                        )
                        
                        # Third pass: fine details (reduced influence)
                        noise_val += 0.25 * snoise2(
                            i / (scale * 0.5),
                            j / (scale * 0.5),
                            octaves=4,
                            persistence=0.3,
                            lacunarity=2.2,
                            base=base + 2000
                        )
                        
                        Z[i, j] = noise_val
                
                # Convert to [0, 1] range
                Z = (Z + 1) * 0.5
                
                # Apply terrain enhancement
                if kwargs.get('enhance', True):
                    # Ridge formation
                    Z = np.abs(Z - 0.5) * 2  # Create ridges
                    Z = np.power(Z, ridge_factor)  # Enhance ridges
                    
                    # Terrain power for more dramatic features
                    power = kwargs.get('power', 1.2)
                    Z = np.power(Z, power)
                    
                    # Optional plateau effect
                    if kwargs.get('plateaus', True):
                        plateau_threshold = kwargs.get('plateau_threshold', 0.7)
                        Z[Z > plateau_threshold] = plateau_threshold + (Z[Z > plateau_threshold] - plateau_threshold) * 0.3
                
                # Apply height scaling and offset
                Z = Z * wave_height + z_value
                
            except Exception as e:
                logger.error(f"Error generating Perlin noise: {e}")
                Z = np.zeros((height, width))
        elif pattern == "fbm":
            # Generate improved Fractal Brownian Motion terrain
            Z = np.zeros((height, width))
            
            # Get parameters with good defaults for fBm
            scale = kwargs.get('scale', 150.0)  # Larger scale for bigger features
            octaves = kwargs.get('octaves', 6)  # Number of octaves
            base_frequency = kwargs.get('base_frequency', 1.0)
            persistence = kwargs.get('persistence', 0.65)
            lacunarity = kwargs.get('lacunarity', 2.0)
            
            if seed is not None:
                base = seed
            else:
                base = np.random.randint(0, 1000000)
            
            try:
                # Generate fBm using optimized numpy operations
                for i in range(height):
                    for j in range(width):
                        amplitude = 1.0
                        frequency = base_frequency
                        noise_value = 0.0
                        
                        for o in range(octaves):
                            noise_value += amplitude * snoise2(
                                i * frequency / scale,
                                j * frequency / scale,
                                octaves=1,
                                persistence=1.0,
                                lacunarity=1.0,
                                base=base + o * 1000
                            )
                            
                            amplitude *= persistence
                            frequency *= lacunarity
                            
                        Z[i, j] = noise_value
                
                # Normalize and enhance
                Z = (Z + 1) * 0.5  # Convert to [0, 1] range
                
                # Apply terrain enhancement if requested
                if kwargs.get('enhance', True):
                    # Ridge formation
                    ridge_weight = kwargs.get('ridge_weight', 0.3)
                    Z = Z * (1 - ridge_weight) + (np.abs(Z - 0.5) * 2) * ridge_weight
                    
                    # Apply power curve for more dramatic features
                    power = kwargs.get('power', 1.2)
                    Z = np.power(Z, power)
                    
                    # River valley formation
                    if kwargs.get('river_valleys', True):
                        valley_threshold = kwargs.get('valley_threshold', 0.3)
                        valley_depth = kwargs.get('valley_depth', 0.4)
                        valleys = Z < valley_threshold
                        Z[valleys] *= valley_depth
                    
                    # Mountain peaks
                    if kwargs.get('mountain_peaks', True):
                        peak_threshold = kwargs.get('peak_threshold', 0.7)
                        peak_factor = kwargs.get('peak_factor', 1.5)
                        peaks = Z > peak_threshold
                        Z[peaks] = peak_threshold + (Z[peaks] - peak_threshold) * peak_factor
                
                # Apply final height scaling and offset
                Z = Z * wave_height + z_value
                
            except Exception as e:
                logger.error(f"Error generating fBm terrain: {e}")
                Z = np.zeros((height, width))
        elif pattern in ("wm_ab", "ausloos_berman_wm"):
            # Ausloos–Berman Weierstrass–Mandelbrot fractal (2D ridge directions)
            D_f = float(kwargs.get("wm_df", 2.5))
            gamma_wm = float(kwargs.get("wm_gamma", 1.5))
            M_wm = int(kwargs.get("wm_M", 16))
            N_max = int(kwargs.get("wm_n_max", 32))
            L_wm = float(kwargs.get("wm_L", 10.0))
            phases = kwargs.get("wm_phases")
            if phases is None:
                phases = make_phases_for_seed(seed, M_wm, N_max)
            else:
                phases = np.asarray(phases, dtype=np.float64)
            try:
                Z = ausloos_berman_wm_grid(
                    X,
                    Y,
                    L=L_wm,
                    D_f=D_f,
                    gamma=gamma_wm,
                    M=M_wm,
                    N_max=N_max,
                    phases=phases,
                )
                Z = Z * wave_height + z_value
            except ValueError as e:
                logger.error("Error generating Ausloos–Berman W–M terrain: %s", e)
                Z = np.zeros((height, width))
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
            if seed is not None:
                np.random.seed(None)
            return np.zeros((height, width), dtype=np.float32)

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