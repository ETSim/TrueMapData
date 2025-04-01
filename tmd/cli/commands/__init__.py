"""
TMD CLI Commands Package.

This package contains modular command implementations for the TMD CLI tools.
Commands are organized by functionality (compression, visualization, etc.)
and can be imported and used by different CLI interfaces.
"""

from tmd.cli.commands.base import BaseCommand, check_dependencies_and_install
from tmd.cli.commands.compress import compress_tmd_command, display_file_info_command
from tmd.cli.commands.batch import BatchProcessor
from tmd.cli.commands.model import generate_model_command

__all__ = [
    'BaseCommand',
    'check_dependencies_and_install',
    'compress_tmd_command',
    'display_file_info_command',
    'BatchProcessor',
    'generate_model_command',
]
