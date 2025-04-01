#!/usr/bin/env python3
"""
Batch processing functionality for TMD CLI.

This module provides tools for processing multiple TMD files in a batch operation.
"""

import time
from pathlib import Path
from typing import Callable, List, Dict, Any, Optional

# Fix imports to avoid circular references
from tmd.cli.core.ui import console, print_warning, print_error, print_success
try:
    from rich import print as rprint
    HAS_RICH = True
except ImportError:
    rprint = print
    HAS_RICH = False

class BatchProcessor:
    """
    Handles batch processing of multiple TMD files.
    
    This class finds and processes multiple files according to a provided
    processing function, with options for recursive search and pattern matching.
    """
    
    def __init__(
        self, 
        directory: Path, 
        recursive: bool = False, 
        pattern: str = "*.tmd"
    ):
        """
        Initialize batch processor.
        
        Args:
            directory: Directory to search for files
            recursive: Whether to search recursively in subdirectories
            pattern: File pattern to match (e.g., "*.tmd")
        """
        self.directory = Path(directory)
        self.recursive = recursive
        self.pattern = pattern
        
    def find_files(self) -> List[Path]:
        """
        Find files matching the pattern in the directory.
        
        Returns:
            List of file paths matching the pattern
        """
        if self.recursive:
            return list(self.directory.glob(f"**/{self.pattern}"))
        else:
            return list(self.directory.glob(self.pattern))
            
    def process_files(
        self, 
        process_func: Callable[[Path], bool], 
        output_dir: Optional[Path] = None,
        description: str = "Processing files"
    ) -> Dict[str, Any]:
        """
        Process all found files using the provided function.
        
        Args:
            process_func: Function that processes a single file
                          Should accept a Path and return a bool indicating success
            output_dir: Output directory (optional)
            description: Description of the processing operation
            
        Returns:
            Dictionary with processing results
        """
        files = self.find_files()
        if not files:
            print_warning(f"No files matching '{self.pattern}' found in {self.directory}")
            return {
                "total": 0,
                "success": 0,
                "failed": 0,
                "output_dir": output_dir,
                "files": []
            }
            
        print_success(f"Found {len(files)} files to process")
        
        success_count = 0
        failed_count = 0
        processed_files = []
        
        # Use Rich progress bar if available
        if HAS_RICH:
            from rich.progress import Progress, TextColumn, BarColumn, SpinnerColumn
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TextColumn("[bold]{task.completed}/{task.total}"),
                console=console
            ) as progress:
                task = progress.add_task(description, total=len(files))
                
                for file_path in files:
                    progress.update(task, description=f"Processing {file_path.name}")
                    try:
                        success = process_func(file_path)
                        if success:
                            success_count += 1
                            processed_files.append(str(file_path))
                        else:
                            failed_count += 1
                    except Exception as e:
                        print_error(f"Error processing {file_path.name}: {e}")
                        failed_count += 1
                        
                    progress.advance(task)
        else:
            # Simple text-based progress for environments without Rich
            print(f"\n{description}...")
            for i, file_path in enumerate(files):
                print(f"[{i+1}/{len(files)}] Processing {file_path.name}...")
                try:
                    success = process_func(file_path)
                    if success:
                        success_count += 1
                        processed_files.append(str(file_path))
                        print(f"  ✓ Success")
                    else:
                        failed_count += 1
                        print(f"  ✗ Failed")
                except Exception as e:
                    print(f"  ✗ Error: {e}")
                    failed_count += 1
        
        return {
            "total": len(files),
            "success": success_count,
            "failed": failed_count,
            "output_dir": output_dir,
            "files": processed_files
        }
