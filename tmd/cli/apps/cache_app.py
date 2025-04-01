#!/usr/bin/env python3
"""
Cache management app for TMD CLI.
"""

import typer
from rich.panel import Panel

from tmd.cli.core.ui import console, print_error, print_success
from tmd.cli.utils.caching import get_cache_stats, clear_cache

def create_cache_app():
    """Create the cache app with all commands."""
    cache_app = typer.Typer(help="Manage TMD file cache")
    
    cache_app.command(name="info")(cache_info_command)
    cache_app.command(name="clear")(cache_clear_command)
    cache_app.command(name="clear-all")(cache_clear_all_command)
    
    return cache_app

def cache_info_command():
    """Display information about the TMD cache."""
    try:
        stats = get_cache_stats()
        
        console.print(Panel.fit(
            f"[bold]TMD Cache Information[/bold]\n\n"
            f"Location: {stats['cache_dir']}\n"
            f"Total entries: {stats['entry_count']}\n"
            f"Expired entries: {stats['expired_count']}\n"
            f"Total size: {stats['total_size_mb']:.2f} MB\n"
        ))
    except (NameError, ImportError):
        print_error("Cache functionality is not available")
        return 1

def cache_clear_command(
    expired_only: bool = typer.Option(True, help="Clear only expired entries")
):
    """Clear the TMD cache."""
    try:
        with console.status("Clearing cache..."):
            count = clear_cache(expired_only=expired_only)
        
        if expired_only:
            print_success(f"Cleared {count} expired entries from cache")
        else:
            print_success(f"Cleared entire cache ({count} entries)")
    except (NameError, ImportError):
        print_error("Cache functionality is not available")
        return 1

def cache_clear_all_command():
    """Clear the entire TMD cache."""
    try:
        with console.status("Clearing entire cache..."):
            count = clear_cache(expired_only=False)
        
        print_success(f"Cleared entire cache ({count} entries)")
    except (NameError, ImportError):
        print_error("Cache functionality is not available")
        return 1
