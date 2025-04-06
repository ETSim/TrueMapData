"""
Configuration classes for TMD model exporters.

This module provides configuration classes for model export operations,
with validation, defaults, and serialization capabilities.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple, Union, ClassVar
from dataclasses import dataclass, field, asdict, fields

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """
    Base configuration class for model export operations.
    
    This class provides common configuration parameters for all exporters
    and serves as the base class for format-specific configurations.
    """
    # Spatial parameters
    x_offset: float = 0.0
    y_offset: float = 0.0
    x_length: float = 1.0
    y_length: float = 1.0
    z_scale: float = 1.0
    base_height: float = 0.0
    
    # Mesh parameters
    optimize: bool = True
    calculate_normals: bool = True
    coordinate_system: str = 'right-handed'
    origin_at_zero: bool = True
    normals_inside: bool = False
    
    # Optional parameters (stored as dict)
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """
        Validate configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate numeric parameters
        if self.z_scale <= 0:
            raise ValueError(f"z_scale must be positive, got {self.z_scale}")
        
        if self.base_height < 0:
            raise ValueError(f"base_height cannot be negative, got {self.base_height}")
        
        if self.x_length <= 0:
            raise ValueError(f"x_length must be positive, got {self.x_length}")
        
        if self.y_length <= 0:
            raise ValueError(f"y_length must be positive, got {self.y_length}")
        
        # Validate enum parameters
        valid_coord_systems = ['right-handed', 'left-handed']
        if self.coordinate_system not in valid_coord_systems:
            raise ValueError(
                f"coordinate_system must be one of {valid_coord_systems}, "
                f"got '{self.coordinate_system}'"
            )
    
    def as_dict(self) -> Dict[str, Any]:
        """
        Get configuration as a dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        result = asdict(self)
        # Remove extra if empty
        if not result['extra']:
            del result['extra']
        return result
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """
        Create configuration from a dictionary.
        
        Args:
            config_dict: Dictionary of configuration parameters
            
        Returns:
            New ModelConfig instance
        """
        # Extract known parameters
        known_params = {
            k: v for k, v in config_dict.items() 
            if k in [f.name for f in fields(cls)]
        }
        
        # Store remaining parameters in extra
        extra_params = {
            k: v for k, v in config_dict.items()
            if k not in known_params
        }
        
        # Create instance with known parameters
        config = cls(**known_params)
        
        # Add extra parameters
        config.extra.update(extra_params)
        
        return config


@dataclass
class ExportConfig(ModelConfig):
    """
    Configuration class for model export operations with format-specific parameters.
    """
    # Format parameters
    binary: Optional[bool] = None
    texture: bool = False
    color_map: str = 'terrain'
    texture_resolution: Optional[Tuple[int, int]] = None
    
    # Mesh generation parameters
    triangulation_method: str = "quadtree"  # Changed from adaptive to quadtree for better speed
    method: str = "quadtree"  # Alias for triangulation_method
    error_threshold: float = 0.05  # Increased from 0.01 for faster processing
    min_quad_size: int = 4  # Increased from 2 for fewer subdivisions
    max_quad_size: int = 64  # Increased from 32 for better initial coverage
    curvature_threshold: float = 0.2  # Increased from 0.1 for fewer subdivisions
    max_triangles: Optional[int] = 100000  # Added reasonable default
    simplify_ratio: Optional[float] = 0.25  # Added default simplification
    use_feature_edges: bool = True
    smoothing: float = 0.0
    max_subdivisions: int = 4 # Added default max_subdivisions
    
    def __post_init__(self):
        """Handle parameter aliases and validation."""
        # Sync method and triangulation_method
        if hasattr(self, 'method'):
            self.triangulation_method = str(self.method).replace('MeshMethod.', '').lower()
            
        super().__post_init__()
    
    def validate(self) -> None:
        """
        Validate configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Call parent validation
        super().validate()
        
        # Additional validation for mesh parameters
        if self.triangulation_method not in ['adaptive', 'quadtree']:
            raise ValueError(f"triangulation_method must be 'adaptive' or 'quadtree', got '{self.triangulation_method}'")
            
        if self.error_threshold <= 0:
            raise ValueError(f"error_threshold must be positive, got {self.error_threshold}")
            
        if self.min_quad_size < 1:
            raise ValueError(f"min_quad_size must be at least 1, got {self.min_quad_size}")
            
        if self.max_quad_size < self.min_quad_size:
            raise ValueError(f"max_quad_size ({self.max_quad_size}) must be >= min_quad_size ({self.min_quad_size})")
            
        if self.curvature_threshold <= 0:
            raise ValueError(f"curvature_threshold must be positive, got {self.curvature_threshold}")
            
        if self.max_triangles is not None and self.max_triangles <= 0:
            raise ValueError(f"max_triangles must be positive or None, got {self.max_triangles}")
            
        if self.simplify_ratio is not None and not 0 <= self.simplify_ratio <= 1:
            raise ValueError(f"simplify_ratio must be between 0 and 1, got {self.simplify_ratio}")
            
        if self.smoothing < 0 or self.smoothing > 1:
            raise ValueError(f"smoothing must be between 0 and 1, got {self.smoothing}")
            
        if self.max_subdivisions < 1:
            raise ValueError(f"max_subdivisions must be positive, got {self.max_subdivisions}")


class ConfigManager:
    """
    Manager for storing and retrieving export configurations.
    
    This class provides functionality for reading and writing configuration files,
    maintaining default configurations, and creating configurations for specific formats.
    """
    
    # Default configurations for different formats
    _default_configs: ClassVar[Dict[str, Dict[str, Any]]] = {
        'default': {},  # Base defaults
        'stl': {'binary': True},
        'obj': {'texture': True},
        'ply': {'binary': True, 'calculate_normals': True},
        'gltf': {'texture': True, 'binary': False},
        'glb': {'texture': True, 'binary': True},
        'usd': {'texture': True},
        'nvbd': {'binary': True}
    }
    
    @classmethod
    def get_default_config(cls, format_name: Optional[str] = None) -> ExportConfig:
        """
        Get default configuration for a specific format.
        
        Args:
            format_name: Format name or None for base default
            
        Returns:
            ExportConfig with default values
        """
        # Start with base defaults
        config_dict = cls._default_configs['default'].copy()
        
        # Add format-specific defaults if available
        if format_name and format_name.lower() in cls._default_configs:
            config_dict.update(cls._default_configs[format_name.lower()])
        
        # Create configuration
        return ExportConfig(**config_dict)
    
    @classmethod
    def create_config(
        cls, 
        format_name: Optional[str] = None, 
        **kwargs
    ) -> ExportConfig:
        """
        Create configuration with default values and overrides.
        
        Args:
            format_name: Format name for format-specific defaults
            **kwargs: Configuration overrides
            
        Returns:
            ExportConfig with specified values
        """
        # Get default configuration
        config = cls.get_default_config(format_name)
        
        # Apply overrides
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                config.extra[key] = value
        
        # Validate
        config.validate()
        
        return config
    
    @classmethod
    def load_config(cls, config_file: str) -> ExportConfig:
        """
        Load configuration from a file.
        
        Args:
            config_file: Path to configuration file
            
        Returns:
            ExportConfig loaded from file
            
        Raises:
            IOError: If file cannot be read
            ValueError: If configuration is invalid
        """
        try:
            import json
            
            # Read configuration file
            with open(config_file, 'r') as f:
                config_dict = json.load(f)
            
            # Create configuration
            return ExportConfig.from_dict(config_dict)
            
        except Exception as e:
            raise IOError(f"Failed to load configuration from {config_file}: {e}")
    
    @classmethod
    def save_config(cls, config: ExportConfig, config_file: str) -> None:
        """
        Save configuration to a file.
        
        Args:
            config: ExportConfig to save
            config_file: Path to configuration file
            
        Raises:
            IOError: If file cannot be written
        """
        try:
            import json
            
            # Get configuration as dictionary
            config_dict = config.as_dict()
            
            # Write configuration file
            with open(config_file, 'w') as f:
                json.dump(config_dict, f, indent=2)
                
        except Exception as e:
            raise IOError(f"Failed to save configuration to {config_file}: {e}")
    
    @classmethod
    def merge_configs(cls, base_config: ExportConfig, override_config: Dict[str, Any]) -> ExportConfig:
        """
        Merge base configuration with override values.
        
        Args:
            base_config: Base configuration
            override_config: Dictionary of override values
            
        Returns:
            New ExportConfig with merged values
        """
        # Create dictionary from base config
        config_dict = base_config.as_dict()
        
        # Add extra parameters to main dict
        if 'extra' in config_dict:
            for key, value in config_dict['extra'].items():
                config_dict[key] = value
            del config_dict['extra']
        
        # Apply overrides
        config_dict.update(override_config)
        
        # Create new config
        return ExportConfig.from_dict(config_dict)