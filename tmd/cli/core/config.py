#!/usr/bin/env python3
"""
Configuration management for TMD CLI tools.

This module provides functions for loading, saving, and accessing configuration
settings used by TMD command-line tools.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Default settings
DEFAULT_CONFIG = {
    "output_dir": "tmd_output",
    "default_plotter": "matplotlib",
    "default_colormap": "viridis",
    "auto_open": True,
    "dpi": 300,
    "image_format": "png",
    "debug_mode": False,
    "theme": "default",
    "recent_files": []
}

def get_config_path() -> Path:
    """Get the path to the configuration file."""
    return Path.home() / ".tmd_config.json"

def load_config() -> Dict[str, Any]:
    """
    Load configuration from config file or create default one.
    
    Returns:
        Dictionary containing configuration settings.
    """
    config_path = get_config_path()
    try:
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            # Update with any missing default values
            for key, value in DEFAULT_CONFIG.items():
                if key not in config:
                    config[key] = value
            return config
        else:
            # Create default config
            save_config(DEFAULT_CONFIG)
            return DEFAULT_CONFIG.copy()
    except Exception as e:
        logger.warning(f"Could not load config file: {e}")
        return DEFAULT_CONFIG.copy()

def save_config(config: Dict[str, Any]) -> None:
    """
    Save configuration to config file.
    
    Args:
        config: Configuration dictionary to save.
    """
    config_path = get_config_path()
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not save config file: {e}")

def update_recent_files(filepath: str) -> None:
    """
    Add a file to the recent files list.
    
    Args:
        filepath: Path to the file to add.
    """
    config = load_config()
    recent = config.get("recent_files", [])
    
    # Add to front of list if not already present, otherwise move to front
    if filepath in recent:
        recent.remove(filepath)
    
    # Add to front and limit to 10 entries
    recent.insert(0, filepath)
    config["recent_files"] = recent[:10]
    save_config(config)

def get_config_value(key: str, default: Any = None) -> Any:
    """
    Get a value from the config, falling back to a default if not found.
    
    Args:
        key: Configuration key
        default: Default value to return if key is not found
        
    Returns:
        The configuration value
    """
    config = load_config()
    return config.get(key, default)

def set_config_value(key: str, value: Any) -> None:
    """
    Set a configuration value and save the config.
    
    Args:
        key: Configuration key
        value: Value to set
    """
    config = load_config()
    config[key] = value
    save_config(config)

def reset_config() -> None:
    """Reset configuration to default values."""
    save_config(DEFAULT_CONFIG.copy())