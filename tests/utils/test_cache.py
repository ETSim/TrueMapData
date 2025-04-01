#!/usr/bin/env python3
"""
Test script for TMD caching system.

This script tests the cache functionality by loading a TMD file twice
and verifying that the second load uses the cached data.
"""

import logging
import time
from pathlib import Path
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_caching(file_path):
    """Test the TMD file caching system."""
    from tmd.cli.utils.caching import get_cache_stats, clear_cache, cache_tmd_data, get_cached_tmd_data
    from tmd.cli.core.io import load_tmd_file
    
    # Clear any existing cache first
    print("Clearing cache...")
    clear_cache(expired_only=False)
    
    # Display initial cache stats
    stats = get_cache_stats()
    print(f"Initial cache stats: {stats['entry_count']} entries, {stats['total_size_mb']:.2f} MB")
    
    # First load (should store in cache)
    print(f"\nFirst load of {file_path}...")
    start_time = time.time()
    tmd_obj = load_tmd_file(file_path, with_console_status=True)
    if tmd_obj is None:
        print("Failed to load TMD file.")
        return
    first_load_time = time.time() - start_time
    print(f"First load took {first_load_time:.3f} seconds")
    
    # Check cache stats after first load
    stats = get_cache_stats()
    print(f"Cache stats after first load: {stats['entry_count']} entries, {stats['total_size_mb']:.2f} MB")
    
    # Second load (should use cache)
    print(f"\nSecond load of {file_path}...")
    start_time = time.time()
    tmd_obj = load_tmd_file(file_path, with_console_status=True)
    second_load_time = time.time() - start_time
    print(f"Second load took {second_load_time:.3f} seconds")
    
    # Calculate improvement
    if first_load_time > 0:
        speedup = first_load_time / second_load_time
        print(f"Cache speedup: {speedup:.1f}x faster")
    
    # Show final cache stats
    stats = get_cache_stats()
    print(f"\nFinal cache stats:")
    print(f"- Entries: {stats['entry_count']}")
    print(f"- Size: {stats['total_size_mb']:.2f} MB")
    print(f"- Location: {stats['cache_dir']}")

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python test_cache.py <tmd_file>")
        sys.exit(1)
    
    # Get file path from command line
    file_path = Path(sys.argv[1])
    if not file_path.exists():
        print(f"Error: File '{file_path}' not found")
        sys.exit(1)
    
    # Run the test
    test_caching(file_path)
