#!/usr/bin/env python3
"""
Base command functionality for TMD CLI tools.

This module provides base classes and utilities for creating commands
that can be used across different TMD command-line tools.
"""

import sys
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, Union

import typer

from tmd.cli.core.config import load_config, save_config
from tmd.cli.core.ui import console, print_rich_table, print_warning, print_error, print_success

class BaseCommand:
    """
    Base class for TMD CLI commands.
    
    This class provides common functionality for TMD command-line tools,
    including configuration access and UI utilities.
    """
    
    def __init__(self, name: str, description: str = ""):
        """
        Initialize the command.
        
        Args:
            name: Command name
            description: Command description
        """
        self.name = name
        self.description = description
        self.config = load_config()
    
    def _convert_value_type(self, value: str) -> Any:
        """Convert string values to appropriate types for config."""
        # Try to interpret boolean values
        if value.lower() in ('true', 'yes', '1', 'y'):
            return True
        if value.lower() in ('false', 'no', '0', 'n'):
            return False
        
        # Try numeric conversion
        try:
            # Try as int first
            return int(value)
        except ValueError:
            try:
                # Then as float
                return float(value)
            except ValueError:
                # Otherwise keep as string
                return value
    
    def display_config(self) -> None:
        """Display the current configuration."""
        table_data = []
        for key, value in sorted(self.config.items()):
            # Skip internal keys that users don't need to see
            if key.startswith('_'):
                continue
                
            # Format special cases
            if key == "recent_files" and isinstance(value, list):
                if value:
                    val_str = f"{len(value)} files (most recent: {value[0]})"
                else:
                    val_str = "No recent files"
                table_data.append({
                    "Setting": key,
                    "Value": val_str,
                    "Type": "list"
                })
            else:
                table_data.append({
                    "Setting": key,
                    "Value": str(value),
                    "Type": type(value).__name__
                })
        
        print_rich_table(
            table_data, 
            "Current Configuration",
            [("Setting", "cyan"), ("Value", "green"), ("Type", "blue")]
        )

    def update_config(self, key: str, value: Any) -> None:
        """
        Update a configuration value and save it.
        
        Args:
            key: Configuration key
            value: New value
        """
        self.config[key] = value
        save_config(self.config)
        print_success(f"Updated '{key}' to '{value}'")

    def reset_config(self) -> None:
        """Reset configuration to defaults with confirmation."""
        if typer.confirm("Reset configuration to defaults?", default=False):
            save_config({})
            self.config = load_config()
            print_success("Configuration reset to defaults")

def check_dependencies_and_install(required_pkgs: List[str] = None) -> bool:
    """
    Check for required dependencies and offer to install them if missing.
    
    Args:
        required_pkgs: List of required package names.
        
    Returns:
        Boolean indicating whether all dependencies are available.
    """
    if required_pkgs is None:
        required_pkgs = ["matplotlib", "plotly", "seaborn", "rich", "typer"]
    
    missing = []
    
    try:
        import pkg_resources
        installed_packages = {pkg.key.lower(): pkg.version for pkg in pkg_resources.working_set}
        missing = [pkg for pkg in required_pkgs if pkg.lower() not in installed_packages]
        
        if missing:
            print_warning(f"Missing required dependencies: {', '.join(missing)}")
            install = typer.confirm("Would you like to install them now?", default=True)
            
            if install:
                import subprocess
                try:
                    console.print("Installing dependencies...")
                    subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
                    print_success("Dependencies installed successfully.")
                    print_warning("Please restart the script to use the new dependencies.")
                    return False
                except Exception as e:
                    print_error(f"Error installing dependencies: {e}")
                    return False
            return False
        return True
    except Exception as e:
        print_warning(f"Could not check dependencies: {e}")
        return False
