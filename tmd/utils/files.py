"""
File utility functions for TMD.

This module provides utilities for working with files, such as generating unique
filenames, listing files with a specific extension, etc.
"""

import os
import re
import glob
from typing import List, Optional, Tuple, Union

def generate_unique_filename(
    filename: str, 
    max_attempts: int = 1000
) -> str:
    """
    Generate a unique filename by adding a numeric suffix if needed.
    
    Args:
        filename: Original filename
        max_attempts: Maximum number of attempts to generate a unique name
        
    Returns:
        Unique filename that doesn't exist on disk
    """
    if not os.path.exists(filename):
        return filename
        
    # Split the filename into base and extension
    base, ext = os.path.splitext(filename)
    
    # Check if the base already has a numeric suffix
    prefix = base
    pattern = r'(.+)_(\d+)$'
    match = re.match(pattern, base)
    if match:
        prefix = match.group(1)
    
    # Try incrementing numbers until we find an unused filename
    for i in range(1, max_attempts + 1):
        new_filename = f"{prefix}_{i}{ext}"
        if not os.path.exists(new_filename):
            return new_filename
            
    # If we've exhausted all attempts, create a timestamp-based unique name
    import time
    timestamp = int(time.time())
    return f"{prefix}_{timestamp}{ext}"

def list_files_with_extension(
    directory: str,
    extension: str,
    recursive: bool = False
) -> List[str]:
    """
    List all files with a specific extension in a directory.
    
    Args:
        directory: Directory to search
        extension: File extension (with or without dot)
        recursive: Whether to search subdirectories
        
    Returns:
        List of file paths
    """
    # Normalize extension to include dot
    if not extension.startswith('.'):
        extension = '.' + extension
        
    # Create search pattern
    if recursive:
        pattern = os.path.join(directory, '**', f'*{extension}')
        files = glob.glob(pattern, recursive=True)
    else:
        pattern = os.path.join(directory, f'*{extension}')
        files = glob.glob(pattern)
        
    # Sort files for consistent ordering
    return sorted(files)

def ensure_directory_exists(directory: str) -> bool:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Directory path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        os.makedirs(directory, exist_ok=True)
        return True
    except Exception:
        return False

def get_filename_without_extension(filepath: str) -> str:
    """
    Get the filename without extension from a filepath.
    
    Args:
        filepath: Full path including filename
        
    Returns:
        Filename without extension
    """
    return os.path.splitext(os.path.basename(filepath))[0]

def get_directory_from_filepath(filepath: str) -> str:
    """
    Get the directory part from a filepath.
    
    Args:
        filepath: Full path including filename
        
    Returns:
        Directory part of the path
    """
    return os.path.dirname(filepath)

def find_files_by_pattern(
    directory: str,
    pattern: str,
    recursive: bool = False
) -> List[str]:
    """
    Find files matching a pattern in a directory.
    
    Args:
        directory: Directory to search
        pattern: Glob pattern to match
        recursive: Whether to search subdirectories
        
    Returns:
        List of file paths
    """
    search_pattern = os.path.join(directory, pattern)
    return sorted(glob.glob(search_pattern, recursive=recursive))
