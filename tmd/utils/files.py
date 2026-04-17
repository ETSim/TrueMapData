#!/usr/bin/env python3
"""
Combined utility class for TMD file processing, environment setup, and lazy imports.

This module provides a single class that encapsulates utilities for working with files,
generating unique filenames, listing files with specific extensions, setting up the
environment, and performing lazy imports.
"""

import importlib
import importlib.util
import logging
import os
import sys
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Set up logger
logger = logging.getLogger(__name__)

# Check for rich text formatting library availability
try:
    from rich import print as rprint
    from rich.console import Console
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    console = None
    rprint = print

# Always ensure numpy is available
HAS_NUMPY = importlib.util.find_spec("numpy") is not None
if not HAS_NUMPY:
    logger.error("NumPy is required for TMD file operations but not installed.")

# Check advanced visualization capabilities
HAS_MATPLOTLIB = False
HAS_SCIPY = False
HAS_PIL = False

def _check_visualization_capabilities():
    """Check for visualization libraries."""
    global HAS_MATPLOTLIB, HAS_SCIPY, HAS_PIL

    HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None
    HAS_SCIPY = importlib.util.find_spec("scipy") is not None
    HAS_PIL = importlib.util.find_spec("PIL") is not None

    return (HAS_MATPLOTLIB, HAS_SCIPY, HAS_PIL)

# Perform initial check
HAS_VIZ = any(_check_visualization_capabilities())


class TMDFileUtilities:
    """Utility class for TMD file operations."""
    
    @staticmethod
    def ensure_directory(directory: Union[str, Path]) -> Path:
        """
        Create directory if it doesn't exist.
        
        Args:
            directory: Path to create
            
        Returns:
            Path object for created directory
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    # Add an alias for backward compatibility
    @staticmethod
    def ensure_directory_exists(directory: Union[str, Path]) -> Path:
        """
        Alias for ensure_directory for backward compatibility.
        
        Args:
            directory: Path to create
            
        Returns:
            Path object for created directory
        """
        return TMDFileUtilities.ensure_directory(directory)
    
    @staticmethod
    def get_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get information about a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with file metadata
        """
        path = Path(file_path)
        return {
            'name': path.name,
            'size': path.stat().st_size,
            'mtime': path.stat().st_mtime,
            'extension': path.suffix,
            'exists': path.exists(),
            'is_file': path.is_file()
        }
    
    @staticmethod
    def get_file_size(file_path: Union[str, Path]) -> int:
        """
        Get file size in bytes.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File size in bytes
        """
        return Path(file_path).stat().st_size
    
    @staticmethod
    def delete_file(file_path: Union[str, Path]) -> bool:
        """
        Delete a file if it exists.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file was deleted, False otherwise
        """
        path = Path(file_path)
        if path.exists():
            path.unlink()
            return True
        return False
    
    @staticmethod
    def delete_files_by_pattern(directory: Union[str, Path], pattern: str) -> int:
        """
        Delete files matching a pattern.
        
        Args:
            directory: Directory to search
            pattern: File pattern to match
            
        Returns:
            Number of files deleted
        """
        count = 0
        for file_path in Path(directory).glob(pattern):
            if file_path.is_file():
                file_path.unlink()
                count += 1
        return count
    
    @staticmethod
    def find_files_by_pattern(directory: Union[str, Path], pattern: str, 
                             recursive: bool = False) -> List[Path]:
        """
        Find files matching a pattern.
        
        Args:
            directory: Directory to search
            pattern: File pattern to match
            recursive: Whether to search subdirectories
            
        Returns:
            List of matching file paths
        """
        directory_path = Path(directory)
        if recursive:
            return list(directory_path.glob(f"**/{pattern}"))
        else:
            return list(directory_path.glob(pattern))
    
    @staticmethod
    def load_json(file_path: Union[str, Path]) -> Dict:
        """
        Load JSON data from a file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Dictionary with JSON data
        """
        import json
        with open(file_path, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def save_json(data: Dict, file_path: Union[str, Path]) -> None:
        """
        Save dictionary as JSON file.
        
        Args:
            data: Dictionary to save
            file_path: Output file path
        """
        import json
        # Ensure directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @staticmethod
    def open_file(file_path: Union[str, Path]) -> None:
        """
        Open a file with the system's default application.
        
        Args:
            file_path: Path to the file to open
        """
        path = Path(file_path)
        if path.suffix.lower() in ['.html', '.htm']:
            webbrowser.open(f"file://{path.absolute()}")
        else:
            # Use platform-specific commands to open files
            if sys.platform == "win32":
                os.startfile(path)
            elif sys.platform == "darwin":  # macOS
                os.system(f"open '{path}'")
            else:  # Linux and other Unix
                os.system(f"xdg-open '{path}'")

    @staticmethod
    def import_optional_dependency(name: str) -> Optional[Any]:
        """
        Import an optional dependency.
        
        Args:
            name: Name of the module to import
            
        Returns:
            Imported module or None if not available
        """
        try:
            return importlib.import_module(name)
        except (ImportError, ModuleNotFoundError, TypeError, SyntaxError) as exc:
            logger.debug("Optional dependency %r not loadable: %s", name, exc)
            return None

    @staticmethod
    def check_tmd_dependencies(auto_install: bool = False, exit_on_failure: bool = False) -> bool:
        """
        Check if required TMD dependencies are installed.
        
        Args:
            auto_install: Whether to attempt installation of missing dependencies
            exit_on_failure: Whether to exit if dependencies are missing
            
        Returns:
            True if all dependencies are available, False otherwise
        """
        required_deps = ['numpy']
        optional_deps = ['matplotlib', 'plotly', 'scipy']

        missing = []

        # Check required dependencies
        for dep in required_deps:
            if importlib.util.find_spec(dep) is None:
                missing.append(dep)

        for dep in optional_deps:
            if importlib.util.find_spec(dep) is None:
                logger.debug("Optional dependency not installed: %s", dep)

        if missing and exit_on_failure:
            logger.error(f"Required dependencies missing: {', '.join(missing)}")
            logger.error(f"Install them with: pip install {' '.join(missing)}")
            sys.exit(1)
            
        return len(missing) == 0