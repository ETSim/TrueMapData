#!/usr/bin/env python3
"""
I/O utilities for TMD CLI.

This module provides functions for loading and saving TMD files,
opening files with system applications, and managing output directories.
"""

import os
import sys
import webbrowser
import logging
import time
from pathlib import Path
from typing import Optional, Any, List, Dict, Union, Tuple, Callable

# Terminal interface libraries
from tmd.cli.core.ui import console, print_warning, print_error, print_success, HAS_RICH
from tmd.cli.core.config import load_config
from tmd.cli.exceptions import FileError

# Set up logger
logger = logging.getLogger(__name__)

# Import tqdm for progress bars
from tqdm import tqdm

# Lazy-loaded caching module to avoid circular imports
_caching_module = None

def _get_caching_module():
    """Lazy import the caching module to avoid circular dependencies."""
    global _caching_module
    if _caching_module is None:
        from tmd.cli.utils import caching
        _caching_module = caching
    return _caching_module

def create_output_dir(base_dir: Optional[str] = None, subdir: Optional[str] = None) -> Path:
    """
    Create the output directory if it doesn't exist.
    
    Args:
        base_dir: Optional base directory (uses config if None).
        subdir: Optional subdirectory to create under base_dir.
        
    Returns:
        Path object pointing to the created directory.
    """
    if base_dir is None:
        config = load_config()
        output_path = Path(config["output_dir"])
    else:
        output_path = Path(base_dir)
    
    if subdir:
        output_path = output_path / subdir
    
    # Create directory with Rich feedback
    if not output_path.exists():
        with console.status(f"Creating directory: {output_path}"):
            output_path.mkdir(parents=True, exist_ok=True)
            console.print(f"[green]Created directory:[/green] {output_path}")
    else:
        console.print(f"[blue]Using existing directory:[/blue] {output_path}")
            
    return output_path

def auto_open_file(filepath: Path) -> None:
    """
    Open the file in the default viewer if auto_open is enabled in config.
    
    Args:
        filepath: Path to the file to open.
    """
    config = load_config()
    auto_open = config.get("auto_open", True)
    
    if auto_open:
        try:
            with console.status(f"Opening {filepath.name}..."):
                time.sleep(0.5)  # Small delay to show the status
                _open_file(filepath)
            console.print(f"[green]Opened:[/green] {filepath.name}")
        except Exception as e:
            logger.warning(f"Could not open file automatically: {e}")
            console.print(f"[yellow]Could not open file automatically:[/yellow] {str(e)}")
            
def _open_file(filepath: Path) -> None:
    """Helper function to open a file using the system's default application."""
    if filepath.suffix.lower() in ['.html', '.htm']:
        webbrowser.open(f"file://{filepath.absolute()}")
    else:
        # Use platform-specific commands to open files
        if sys.platform == "win32":
            os.startfile(filepath)
        elif sys.platform == "darwin":
            os.system(f"open '{filepath}'")
        else:
            os.system(f"xdg-open '{filepath}'")

def get_file_extension(plotter: str) -> str:
    """
    Get the appropriate file extension based on the plotter.
    
    Args:
        plotter: Name of the plotter.
        
    Returns:
        File extension (including dot).
    """
    if plotter.lower() == "plotly":
        return ".html"
    
    config = load_config()
    return f".{config.get('image_format', 'png')}"

def get_output_filename(tmd_file: Path, plotter: str, viz_type: str, 
                       output: Optional[Path] = None, subdir: Optional[str] = None) -> Path:
    """
    Generate output filename based on file, plotter and visualization type.
    
    Args:
        tmd_file: Path to the TMD file.
        plotter: Plotter name.
        viz_type: Visualization type.
        output: Optional explicit output path.
        subdir: Optional subdirectory name.
        
    Returns:
        Path object for the output file.
    """
    if output is not None:
        return Path(output)
        
    output_dir = create_output_dir(subdir=subdir)
    file_stem = tmd_file.stem
    ext = get_file_extension(plotter)
    
    result = output_dir / f"{file_stem}_{viz_type}_{plotter}{ext}"
    
    if HAS_RICH:
        console.print(f"[blue]Output will be saved to:[/blue] {result}")
        
    return result

def load_tmd_file(file_path: Path, with_console_status: bool = False, 
                 with_progress: bool = True, use_cache: bool = False) -> Optional[Any]:
    """
    Load a TMD file with progress indicator and error handling.
    
    Uses caching to improve loading times for previously loaded files if use_cache is True.
    
    Args:
        file_path: Path to the TMD file to load
        with_console_status: Whether to show a console status indicator
        with_progress: Whether to show a progress bar
        use_cache: Whether to use cached data (if available)
        
    Returns:
        TMD object or None if loading failed
    """
    try:
        # Try to get cached data first if use_cache is True
        if use_cache:
            caching = _get_caching_module()
            cached_data = caching.get_cached_tmd_data(file_path)
            
            if cached_data:
                height_map, metadata = cached_data
                
                if with_console_status:
                    print_success(f"Loaded {file_path.name} from cache")
                    
                # Create TMD object from cached data
                from tmd import TMD
                tmd_obj = TMD(height_map=height_map, metadata=metadata)
                return tmd_obj
            
        # Fall back to normal loading if not using cache or not in cache
        if with_console_status:
            # Enhanced rich progress display for file loading
            with console.status(f"[bold blue]Loading[/bold blue] {file_path.name}...") as status:
                # Show file size info
                try:
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    console.print(f"File size: {size_mb:.2f} MB")
                except Exception:
                    pass
                    
                from tmd import TMD
                start_time = time.time()
                tmd_obj = TMD.load(file_path)
                elapsed = time.time() - start_time
                console.print(f"Loaded in {elapsed:.2f} seconds")
        else:
            from tmd import TMD
            tmd_obj = TMD.load(file_path)
            
        # Cache the loaded data for future use if use_cache is True
        if use_cache:
            height_map = tmd_obj.height_map
            metadata = tmd_obj.metadata
            
            # Show caching progress
            if with_console_status:
                with console.status("Caching data for faster access next time..."):
                    caching = _get_caching_module()
                    caching.cache_tmd_data(file_path, height_map, metadata)
                    console.print("[green]Data cached successfully[/green]")
            else:
                caching = _get_caching_module()
                caching.cache_tmd_data(file_path, height_map, metadata)
        
        return tmd_obj
    except Exception as e:
        error_msg = f"Failed to load {file_path.name}: {str(e)}"
        if with_console_status:
            print_error(error_msg)
            
            # Provide more detailed error information
            console.print(Panel(str(e), title="Detailed Error Information", 
                               border_style="red"))
            # Check file existence and permissions
            if not file_path.exists():
                console.print("[yellow]File does not exist[/yellow]")
            elif not os.access(file_path, os.R_OK):
                console.print("[yellow]File exists but cannot be read (permission error)[/yellow]")
        else:
            raise FileError(error_msg) from e
        return None

def find_files_by_pattern(directory: Path, pattern: str = "*.tmd", 
                         recursive: bool = False) -> List[Path]:
    """
    Find files matching a pattern in a directory.
    
    Args:
        directory: Directory to search
        pattern: Glob pattern to match
        recursive: Whether to search subdirectories
        
    Returns:
        List of matched file paths
    """
    if recursive:
        files = list(directory.glob(f"**/{pattern}"))
    else:
        files = list(directory.glob(pattern))
        
    if files:
        file_table = Table(title=f"Found {len(files)} files")
        file_table.add_column("Index", style="cyan")
        file_table.add_column("Filename", style="green")
        file_table.add_column("Size (KB)", justify="right")
        
        for i, file in enumerate(files[:10]):  # Show first 10 files only
            try:
                size_kb = file.stat().st_size / 1024
                size_str = f"{size_kb:.1f}"
            except Exception:
                size_str = "N/A"
                
            file_table.add_row(str(i+1), file.name, size_str)
            
        if len(files) > 10:
            file_table.add_row("...", "...", "...")
            
        console.print(file_table)
    else:
        console.print(f"[yellow]No files matching '{pattern}' found in {directory}[/yellow]")
            
    return files