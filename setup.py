#!/usr/bin/env python3
"""
Setup configuration for the TMD Processor package.
This setup.py file is maintained for backward compatibility with older tools.
The primary build configuration is now in pyproject.toml.
"""
from setuptools import find_packages, setup

setup(
    use_scm_version={
        "write_to": "tmd/_version.py",
        "version_scheme": "post-release",
        "local_scheme": "no-local-version",
    },
)
