"""
Setup configuration for the TMD Processor package.
This setup.py file is maintained for backward compatibility with older tools.
The primary build configuration is in pyproject.toml.

Version 1.0.5.
"""

from setuptools import find_packages, setup

setup(
    name="truemapdata",
    version="1.0.5",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "matplotlib>=3.3.0",
        "pillow>=8.0.0",
        "opencv-python>=4.5.0",
        "scipy>=1.6.0",
        "psutil>=5.8.0",
        "noise>=1.2.2",
    ],
    extras_require={
        "viz": [
            "plotly>=5.0.0",
            "seaborn>=0.11.0",
        ],
        "full": [
            "plotly>=5.0.0",
            "seaborn>=0.11.0",
            "meshio>=5.0.0",
            "pyvista>=0.34.0",
            "pandas>=1.3.0",
            "nbformat>=5.1.0",
            "ipywidgets>=7.6.0",
        ],
    },
    author="Antoine Boucher",
    author_email="antoine@antoineboucher.info",
    description="A Python library for processing TrueMap Data files",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ETSTribology/TrueMapData",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.8",
    project_urls={
        "Bug Tracker": "https://github.com/ETSTribology/TrueMapData/issues",
        "Documentation": "https://truemapdata.readthedocs.io",
    },
)
