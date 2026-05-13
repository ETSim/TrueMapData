#!/usr/bin/env python3
"""
Tests for package version metadata (single source: pyproject / tmd.__version__).
"""

import re

import tmd


def test_version_semver():
    """__version__ follows semantic versioning."""
    assert hasattr(tmd, "__version__")
    assert isinstance(tmd.__version__, str)
    pattern = r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
    assert re.match(pattern, tmd.__version__), f"Invalid semver: {tmd.__version__!r}"


def test_version_matches_release():
    """Keep in sync with pyproject.toml [project].version for releases."""
    assert tmd.__version__ == "1.0.6"


def test_tmd_version_module_has_metadata():
    """``tmd.__version__`` module string matches the package before submodule binds same name."""
    import importlib

    package_version = tmd.__version__
    assert isinstance(package_version, str)
    ver_mod = importlib.import_module("tmd.__version__")
    assert ver_mod.__version__ == package_version
    assert isinstance(ver_mod.__author__, str)
    assert isinstance(ver_mod.__license__, str)
    importlib.reload(tmd)
