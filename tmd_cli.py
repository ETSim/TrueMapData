#!/usr/bin/env python3
"""TMD Command-Line Interface (wrapper around ``tmd.cli.main``)."""

import sys


def _reconfigure_stdio_utf8() -> None:
    """Avoid Rich/Windows cp1252 Unicode errors (spinners, checkmarks) on teardown."""
    if sys.platform != "win32":
        return
    for stream in (getattr(sys, "stdout", None), getattr(sys, "stderr", None)):
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except OSError:
                pass


_reconfigure_stdio_utf8()

try:
    from tmd.cli.main import main
except ImportError as e:
    from rich.console import Console

    Console().print(f"[red]Error importing TMD modules: {e}[/red]")
    Console().print("[yellow]Make sure TMD is properly installed[/yellow]")
    sys.exit(1)

if __name__ == "__main__":
    sys.exit(main())
