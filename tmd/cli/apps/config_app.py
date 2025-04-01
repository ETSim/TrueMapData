#!/usr/bin/env python3
"""
Configuration app for TMD CLI.
"""

import typer

from tmd.cli.core.ui import console, print_success
from tmd.cli.core.config import load_config, save_config
from rich.panel import Panel

def create_config_app():
    """Create the configuration app with all commands."""
    config_app = typer.Typer(help="Manage TMD configuration")
    
    config_app.command(name="show")(config_show)
    config_app.command(name="set")(config_set)
    config_app.command(name="reset")(config_reset)
    
    return config_app

def config_show():
    """Display current configuration settings."""
    config = load_config()
    
    console.print(Panel.fit("[bold]TMD Configuration[/bold]"))
    for key, value in sorted(config.items()):
        console.print(f"{key}: {value}")

def config_set(
    key: str = typer.Argument(..., help="Configuration key"),
    value: str = typer.Argument(..., help="Configuration value")
):
    """Set a configuration value."""
    # Auto-convert value types
    if value.lower() == "true":
        typed_value = True
    elif value.lower() == "false":
        typed_value = False
    elif value.isdigit():
        typed_value = int(value)
    elif "." in value and all(part.isdigit() for part in value.split(".", 1)):
        typed_value = float(value)
    else:
        typed_value = value
    
    # Load config, update and save
    config = load_config()
    config[key] = typed_value
    save_config(config)
    print_success(f"Configuration updated: {key} = {typed_value}")

def config_reset():
    """Reset configuration to default values."""
    default_config = {
        "default_colormap": "viridis",
        "auto_cache": True,
        "cache_ttl_days": 7,
        "default_plotter": "matplotlib",
        "default_compression_level": 9,
        "use_rich_formatting": True
    }
    
    save_config(default_config)
    print_success("Configuration reset to default values")
