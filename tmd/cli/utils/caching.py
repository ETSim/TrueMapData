#!/usr/bin/env python3
"""
Caching utilities for TMD CLI.

This module provides functionality for caching TMD data to improve load times
when working with the same files repeatedly. Uses TMDFileUtilities for file operations.
"""

import time
import pickle
import hashlib
import zlib
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import numpy as np

from tmd.utils.files import TMDFileUtilities

# Try importing progress utilities
try:
    from tmd.cli.utils.progress import spinner_context, process_with_progress
    HAS_PROGRESS = True
except ImportError:
    HAS_PROGRESS = False

# Set up logger
logger = logging.getLogger(__name__)

# Default cache location and settings
DEFAULT_CACHE_DIR = Path.home() / ".tmd" / "cache"
DEFAULT_CACHE_TTL = 60 * 60 * 24 * 7  # 7 days

class TMDCache:
    """Cache manager for TMD files."""
    
    def __init__(self, cache_dir: Optional[Path] = None, ttl: int = DEFAULT_CACHE_TTL):
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.ttl = ttl
        self._ensure_cache_dir()
        self._index = self._load_index()
        
    def _ensure_cache_dir(self) -> None:
        """Create the cache directory."""
        try:
            # Use TMDFileUtilities to ensure directory exists
            TMDFileUtilities.ensure_directory(self.cache_dir)
        except Exception as e:
            logger.warning(f"Failed to create main cache directory: {e}")
            # Fall back to temporary directory
            import tempfile
            self.cache_dir = Path(tempfile.gettempdir()) / "tmd_cache"
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """Load the cache index."""
        index_path = self.cache_dir / "index.json"
        if index_path.exists():
            try:
                # Use TMDFileUtilities for loading JSON
                return TMDFileUtilities.load_json(index_path)
            except Exception as e:
                logger.warning(f"Failed to load cache index: {e}")
                return {}
        return {}
            
    def _save_index(self) -> None:
        """Save the cache index."""
        index_path = self.cache_dir / "index.json"
        try:
            # Use TMDFileUtilities for saving JSON
            TMDFileUtilities.save_json(self._index, index_path)
        except Exception as e:
            logger.warning(f"Failed to save cache index: {e}")
            
    def _get_cache_key(self, file_path: Path) -> str:
        """Generate a unique cache key based on file path and modification time."""
        try:
            # Get file info from TMDFileUtilities
            file_info = TMDFileUtilities.get_file_info(file_path)
            key_data = f"{file_path.absolute()}:{file_info['mtime']}:{file_info['size']}"
            return hashlib.md5(key_data.encode()).hexdigest()
        except Exception as e:
            logger.warning(f"Failed to get file info for cache key: {e}")
            return hashlib.md5(str(file_path.absolute()).encode()).hexdigest()
            
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the file path for a cache entry."""
        return self.cache_dir / f"{cache_key}.tmdcache"
            
    def put(self, file_path: Path, height_map: np.ndarray, metadata: Dict[str, Any]) -> bool:
        """Store TMD data in the cache with compression."""
        try:
            cache_key = self._get_cache_key(file_path)
            cache_path = self._get_cache_path(cache_key)
            
            # Prepare compressed data with progress indicator if available
            if HAS_PROGRESS:
                with spinner_context(f"Caching data for {file_path.name}"):
                    compressed_data = zlib.compress(height_map.tobytes())
                    cache_data = {
                        'shape': height_map.shape,
                        'dtype': str(height_map.dtype),
                        'data': compressed_data,
                        'metadata': metadata
                    }
                    
                    # Save to file
                    with open(cache_path, 'wb') as f:
                        pickle.dump(cache_data, f)
            else:
                # Without progress indicators
                compressed_data = zlib.compress(height_map.tobytes())
                cache_data = {
                    'shape': height_map.shape,
                    'dtype': str(height_map.dtype),
                    'data': compressed_data,
                    'metadata': metadata
                }
                
                # Save to file
                with open(cache_path, 'wb') as f:
                    pickle.dump(cache_data, f)
            
            # Update index
            self._index[cache_key] = {
                'file_path': str(file_path.absolute()),
                'timestamp': time.time(),
                'cache_path': str(cache_path),
                'size': TMDFileUtilities.get_file_size(cache_path)
            }
            self._save_index()
            
            return True
        except Exception as e:
            logger.warning(f"Failed to cache TMD data: {e}")
            return False
            
    def get(self, file_path: Path) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """Retrieve TMD data from the cache if available and not expired."""
        try:
            cache_key = self._get_cache_key(file_path)
            
            # Check if in cache and not expired
            if cache_key not in self._index:
                return None
                
            entry = self._index[cache_key]
            if time.time() - entry['timestamp'] > self.ttl:
                self._remove_cache_entry(cache_key)
                return None
                
            cache_path = Path(entry['cache_path'])
            if not cache_path.exists():
                self._remove_cache_entry(cache_key)
                return None
            
            # Use progress indicator if available
            if HAS_PROGRESS:
                with spinner_context(f"Loading {file_path.name} from cache"):
                    # Load and decompress
                    with open(cache_path, 'rb') as f:
                        cache_data = pickle.load(f)
                        
                    height_map = np.frombuffer(
                        zlib.decompress(cache_data['data']), 
                        dtype=np.dtype(cache_data['dtype'])
                    ).reshape(cache_data['shape'])
            else:
                # Without progress indicator
                # Load and decompress
                with open(cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                    
                height_map = np.frombuffer(
                    zlib.decompress(cache_data['data']), 
                    dtype=np.dtype(cache_data['dtype'])
                ).reshape(cache_data['shape'])
            
            # Update access timestamp
            self._index[cache_key]['timestamp'] = time.time()
            self._save_index()
            
            return height_map, cache_data['metadata']
        except Exception as e:
            logger.warning(f"Failed to retrieve from cache: {e}")
            return None
            
    def _remove_cache_entry(self, cache_key: str) -> None:
        """Remove a cache entry."""
        try:
            if cache_key in self._index:
                cache_path = Path(self._index[cache_key]['cache_path'])
                if cache_path.exists():
                    # Use TMDFileUtilities to delete the file
                    TMDFileUtilities.delete_file(cache_path)
                del self._index[cache_key]
                self._save_index()
        except Exception as e:
            logger.warning(f"Failed to remove cache entry: {e}")
            
    def clear_expired(self) -> int:
        """Remove all expired cache entries."""
        removed_count = 0
        current_time = time.time()
        expired_keys = [
            k for k, v in self._index.items() 
            if current_time - v['timestamp'] > self.ttl
        ]
        
        # Use process_with_progress if available
        if HAS_PROGRESS and expired_keys:
            def remove_key(key):
                self._remove_cache_entry(key)
                return True
                
            results = process_with_progress(
                expired_keys, 
                remove_key, 
                "Removing expired cache entries"
            )
            removed_count = results["success"]
        else:
            # Manual iteration without progress
            for key in expired_keys:
                self._remove_cache_entry(key)
                removed_count += 1
            
        return removed_count
        
    def clear_all(self) -> int:
        """Clear the entire cache."""
        count = len(self._index)
        
        # Clear index
        self._index = {}
        self._save_index()
        
        # Use TMDFileUtilities to find and delete files
        deleted = TMDFileUtilities.delete_files_by_pattern(self.cache_dir, "*.tmdcache")
        logger.info(f"Deleted {deleted} cache files")
                
        return count
        
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_size = sum(entry['size'] for entry in self._index.values())
        current_time = time.time()
        expired_count = sum(1 for v in self._index.values() 
                          if current_time - v['timestamp'] > self.ttl)
        
        return {
            'entry_count': len(self._index),
            'expired_count': expired_count,
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'cache_dir': str(self.cache_dir)
        }

# Global cache instance
_cache_instance = None

def get_cache() -> TMDCache:
    """Get the global TMD cache instance (singleton pattern)."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = TMDCache()
    return _cache_instance

# Simplified public API functions
def cache_tmd_data(file_path: Path, height_map: np.ndarray, metadata: Dict[str, Any]) -> bool:
    """Cache TMD data for faster future loading."""
    return get_cache().put(file_path, height_map, metadata)

def get_cached_tmd_data(file_path: Path) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
    """Get TMD data from cache if available."""
    return get_cache().get(file_path)

def clear_cache(expired_only: bool = True) -> int:
    """Clear the TMD cache."""
    cache = get_cache()
    return cache.clear_expired() if expired_only else cache.clear_all()

def get_cache_stats() -> Dict[str, Any]:
    """Get statistics about the TMD cache."""
    return get_cache().get_stats()
