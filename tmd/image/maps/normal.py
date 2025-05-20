"""Normal map generator with enhanced error handling and logging."""
import logging
import sys
import numpy as np
from .base_generator import MapGenerator

# Set up logging
logger = logging.getLogger(__name__)

# Add console handler if not already present
if not logger.handlers:
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)
    logger.setLevel(logging.DEBUG)

class NormalMapGenerator(MapGenerator):
    """Generator for normal maps with enhanced error handling and logging."""
    
    def __init__(self, strength: float = 1.0, normalize: bool = False, debug: bool = False, **kwargs):
        """
        Initialize with strength parameter.
        
        Args:
            strength: Factor to control the strength of normals
            normalize: Whether to normalize the height map before processing
            debug: Whether to print detailed debug information
            **kwargs: Additional parameters
        """
        self.debug = debug
        super().__init__(strength=strength, normalize=normalize, **kwargs)
        logger.info(f"NormalMapGenerator initialized with strength={strength}, normalize={normalize}")
    
    def _log_and_print(self, message, level="info"):
        """Log a message and print to console if debug is enabled."""
        if level == "debug":
            logger.debug(message)
        elif level == "info":
            logger.info(message)
        elif level == "warning":
            logger.warning(message)
        elif level == "error":
            logger.error(message)
            
        if self.debug:
            print(f"NormalMapGenerator: {message}")
    
    def _prepare_height_map(self, height_map, normalize=False):
        """
        Prepare height map by normalizing and handling edge cases.
        
        This is a safer version of the parent class method in case it's not working.
        
        Args:
            height_map: Input height map
            normalize: Whether to normalize to [0,1]
            
        Returns:
            Prepared height map
        """
        try:
            # Ensure the height map is valid
            if height_map is None:
                self._log_and_print("Height map is None, returning zeros", "error")
                return np.zeros((1, 1), dtype=np.float32)
                
            # Convert to float32 if needed
            if height_map.dtype != np.float32:
                height_map = height_map.astype(np.float32)
                self._log_and_print(f"Converted height map from {height_map.dtype} to float32", "debug")
            
            # Normalize if requested
            if normalize:
                h_min = np.min(height_map)
                h_max = np.max(height_map)
                
                # Check if range is too small (flat surface)
                if np.isclose(h_max, h_min):
                    self._log_and_print("Height map is flat, using as is", "warning")
                    return height_map
                
                # Normalize to [0,1]
                normalized = (height_map - h_min) / (h_max - h_min)
                self._log_and_print(f"Height map normalized from range [{h_min}, {h_max}] to [0, 1]", "debug")
                return normalized
            
            # Just use the raw height map if not normalizing
            return height_map
            
        except Exception as e:
            self._log_and_print(f"Error preparing height map: {e}", "error")
            # Return a small empty map as fallback
            return np.zeros((1, 1), dtype=np.float32)
            
    def generate(self, height_map: np.ndarray, **kwargs) -> np.ndarray:
        """
        Generate normal map from height data.
        
        Args:
            height_map: Input height map
            **kwargs: Additional parameters including:
                - strength: Controls the strength of normal map effect
                - metadata: TMD metadata containing physical dimensions
                - debug: Set to True for detailed console output
                
        Returns:
            RGB normal map (0-1 range)
        """
        # Enable debug if requested in kwargs
        if kwargs.get('debug', False):
            self.debug = True
            
        self._log_and_print(f"Starting normal map generation for height map of shape {height_map.shape}", "info")
        
        # 1) Safely extract metadata and ensure it's a dict
        metadata = kwargs.pop('metadata', {}) or {}
        self._log_and_print(f"Using metadata: {metadata}", "debug")
        
        # 2) Pull in algorithmic parameters
        params = self._get_params(**kwargs)
        strength = float(params.get('strength', 1.0))
        normalize = bool(params.get('normalize', False))
        self._log_and_print(f"Using parameters: strength={strength}, normalize={normalize}", "debug")
        
        # 3) Prepare the height map
        try:
            height_map_norm = self._prepare_height_map(height_map, normalize=normalize)
            self._log_and_print("Height map prepared successfully", "debug")
            
            # Verify the height map looks reasonable
            if height_map_norm is not None and height_map_norm.size > 0:
                h_min, h_max = np.min(height_map_norm), np.max(height_map_norm)
                self._log_and_print(f"Height map range: [{h_min}, {h_max}]", "debug")
        except Exception as e:
            self._log_and_print(f"Error during height map preparation: {e}", "error")
            # Create a fallback flat normal map
            fallback = np.zeros((max(1, height_map.shape[0]), max(1, height_map.shape[1]), 3), dtype=np.float32)
            fallback[..., 2] = 1.0
            return (fallback + 1.0) * 0.5
        
        try:
            # Validate height map dimensions
            if height_map_norm is None or height_map_norm.size == 0:
                self._log_and_print("Empty height map provided, returning flat normal map", "warning")
                fallback = np.zeros((1, 1, 3), dtype=np.float32)
                fallback[..., 2] = 1.0
                return (fallback + 1.0) * 0.5
                
            rows, cols = height_map_norm.shape
            self._log_and_print(f"Processing height map with dimensions: {rows}x{cols}", "info")
            
            scaling_applied = False
            normal_map = np.zeros((rows, cols, 3), dtype=np.float32)
            
            # 4a) Physical dimensions if available
            if 'x_length' in metadata and 'y_length' in metadata:
                dx = float(metadata['x_length']) / cols if cols > 0 else 1.0
                dy = float(metadata['y_length']) / rows if rows > 0 else 1.0
                scaling_applied = True
                self._log_and_print(f"Using physical dimensions: dx={dx}, dy={dy}", "debug")
            
            # 4b) Millimeters per pixel override
            if 'mmpp' in metadata:
                mmpp = float(metadata['mmpp'])
                dx = dy = mmpp
                scaling_applied = True
                self._log_and_print(f"Using mmpp: {mmpp}", "debug")
                
                # adjust by magnification if present
                if 'magnification' in metadata:
                    mag = float(metadata['magnification'])
                    mag = max(mag, 0.001)  # Prevent division by zero
                    strength *= (0.5 + 0.5 / mag)
                    self._log_and_print(f"Adjusted strength by magnification: {mag} to {strength}", "debug")
            
            # 4c) Fall back to aspect ratio
            if not scaling_applied:
                aspect = cols / rows if rows > 0 else 1.0
                dx = 1.0
                dy = dx / aspect if aspect > 0 else 1.0  # Prevent division by zero
                self._log_and_print(f"Using aspect ratio: {aspect} with dx={dx}, dy={dy}", "debug")
            
            # 5) Compute gradients
            self._log_and_print("Computing gradients...", "debug")
            dx_array = np.zeros_like(height_map_norm)
            dy_array = np.zeros_like(height_map_norm)
            
            # Handle special cases for very small arrays
            if rows < 3 or cols < 3:
                self._log_and_print("Height map too small for gradient calculation, using flat normal map", "warning")
                normal_map[:, :, 2] = 1.0
                return (normal_map + 1.0) * 0.5
            
            # Central differences for interior points
            dx_array[:, 1:-1] = (height_map_norm[:, 2:] - height_map_norm[:, :-2]) / (2.0 * dx)
            dy_array[1:-1, :] = (height_map_norm[2:, :] - height_map_norm[:-2, :]) / (2.0 * dy)
            
            # Forward/backward differences for edges
            dx_array[:, 0] = (height_map_norm[:, 1] - height_map_norm[:, 0]) / dx
            dx_array[:, -1] = (height_map_norm[:, -1] - height_map_norm[:, -2]) / dx
            dy_array[0, :] = (height_map_norm[1, :] - height_map_norm[0, :]) / dy
            dy_array[-1, :] = (height_map_norm[-1, :] - height_map_norm[-2, :]) / dy
            
            # Handle corners explicitly by averaging neighboring values
            self._log_and_print("Handling corner cases...", "debug")
            # Top-left corner
            dx_array[0, 0] = dx_array[0, 1]
            dy_array[0, 0] = dy_array[1, 0]
            
            # Top-right corner
            dx_array[0, -1] = dx_array[0, -2]
            dy_array[0, -1] = dy_array[1, -1]
            
            # Bottom-left corner
            dx_array[-1, 0] = dx_array[-1, 1]
            dy_array[-1, 0] = dy_array[-2, 0]
            
            # Bottom-right corner
            dx_array[-1, -1] = dx_array[-1, -2]
            dy_array[-1, -1] = dy_array[-2, -1]
            
            # Check for NaNs in gradients
            if np.isnan(dx_array).any() or np.isnan(dy_array).any():
                self._log_and_print("WARNING: NaN values detected in gradients", "error")
                # Replace NaNs with zeros
                dx_array = np.nan_to_num(dx_array)
                dy_array = np.nan_to_num(dy_array)
            
            # 6) Apply strength
            self._log_and_print(f"Applying strength: {strength}", "debug")
            dx_array *= strength
            dy_array *= strength
            
            # 7) Build normal vectors and normalize
            self._log_and_print("Building normal vectors...", "debug")
            normal_map[..., 0] = -dx_array
            normal_map[..., 1] = -dy_array
            normal_map[..., 2] = 1.0
            
            # Check for NaNs in normal vectors
            if np.isnan(normal_map).any():
                self._log_and_print("WARNING: NaN values detected in normal vectors", "error")
                normal_map = np.nan_to_num(normal_map)
            
            # Compute norm ensuring no division by zero
            self._log_and_print("Normalizing vectors...", "debug")
            norm = np.linalg.norm(normal_map, axis=2, keepdims=True)
            
            # Use a minimum norm value to avoid division by zero
            min_norm = 1e-10
            norm = np.maximum(norm, min_norm)
            normal_map = np.divide(normal_map, norm)
            
            # Check for abnormal values
            if not np.all(np.isfinite(normal_map)):
                self._log_and_print("WARNING: Non-finite values detected after normalization", "error")
                normal_map = np.nan_to_num(normal_map)
            
            # 8) Map [-1,1] -> [0,1] for RGB output
            self._log_and_print("Mapping to RGB range...", "debug")
            result = (normal_map + 1.0) * 0.5
            
            self._log_and_print(f"Normal map generation completed successfully", "info")
            return result
        
        except Exception as e:
            self._log_and_print(f"Error generating normal map: {e}", "error")
            import traceback
            traceback.print_exc()
            
            # Fallback: flat normal pointing up
            try:
                # Try to use the dimensions from the input
                default = np.zeros((rows, cols, 3), dtype=np.float32)
                self._log_and_print(f"Using fallback normal map with dimensions: {rows}x{cols}", "debug")
            except:
                # If that fails, create a small default
                default = np.zeros((1, 1, 3), dtype=np.float32)
                self._log_and_print("Using minimal fallback normal map", "debug")
                
            default[..., 2] = 1.0
            return (default + 1.0) * 0.5
    
    def _validate_params(self, params):
        """Validate and adjust parameters."""
        self._log_and_print(f"Validating parameters: {params}", "debug")
        
        if params.get('strength', 0) <= 0:
            self._log_and_print("Invalid strength value, defaulting to 1.0", "warning")
            params['strength'] = 1.0
            
        params['normalize'] = bool(params.get('normalize', False))
        return params