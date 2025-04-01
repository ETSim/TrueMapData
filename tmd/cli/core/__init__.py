#!/usr/bin/env python3
"""
Core functionality for TMD CLI tools.

This module provides shared functionality used across different CLI tools,
such as configuration management, file loading, and error handling.
"""

import sys

# Re-export key functionality to make imports easier
from tmd.cli.core.ui import (
    console, 
    print_warning, 
    print_error, 
    print_success,
    print_rich_table,
    display_metadata,
    format_height_map_summary
)

from tmd.cli.core.config import (
    load_config, 
    save_config, 
    get_config_value
)

from tmd.cli.core.io import (
    load_tmd_file, 
    auto_open_file, 
    create_output_dir,
    get_file_extension,
    get_output_filename
)

# Import dependency checking utility
from tmd.utils.files import TMDFileUtilities

# TMD dependencies check
def check_dependencies(auto_install: bool = False, exit_on_failure: bool = True) -> bool:
    """
    Check if all required dependencies are available.
    
    Args:
        auto_install: Whether to attempt installing missing dependencies
        exit_on_failure: Whether to exit if dependencies are missing
        
    Returns:
        True if all required dependencies are available, False otherwise
    """
    required_deps = ['numpy', 'matplotlib']
    optional_deps = ['plotly', 'scipy', 'seaborn']
    
    missing = []
    
    # Check required dependencies
    for dep in required_deps:
        try:
            __import__(dep)
        except ImportError:
            missing.append(dep)
    
    if missing:
        message = f"Required dependencies missing: {', '.join(missing)}"
        console.print(f"[bold red]Error:[/bold red] {message}")
            
        install_cmd = f"pip install {' '.join(missing)}"
        console.print(f"Install with: [bold]{install_cmd}[/bold]")
        
        if auto_install:
            try:
                console.print(f"[yellow]Attempting to install missing dependencies...[/yellow]")
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
                console.print(f"[green]Successfully installed dependencies.[/green]")
                return True
            except Exception as e:
                console.print(f"[bold red]Failed to install dependencies:[/bold red] {str(e)}")
                if exit_on_failure:
                    sys.exit(1)
                return False
        elif exit_on_failure:
            sys.exit(1)
            
        return False
    
    return True
