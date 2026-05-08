"""TMD data I/O package.

This package provides exporters and importers for persisting TMD data in
several formats. The public surface mirrors the submodule names so that
callers can either import directly from a submodule or from the package
root, e.g. ``from tmd.compression import MATImporter``.
"""

from .base import TMDDataExporter, TMDDataImporter
from .factory import TMDDataIOFactory
from .mat import (
    ISO_PARAMETER_KEYS,
    TMD_FORMAT_TAG,
    TMD_FORMAT_VERSION,
    MATExporter,
    MATImporter,
)
from .npy import NPYExporter, NPYImporter
from .npz import NPZExporter, NPZImporter
from .pickle import PickleExporter, PickleImporter
from .zip import ZIPExporter, ZIPImporter

__all__ = [
    "TMDDataExporter",
    "TMDDataImporter",
    "TMDDataIOFactory",
    "MATExporter",
    "MATImporter",
    "NPYExporter",
    "NPYImporter",
    "NPZExporter",
    "NPZImporter",
    "PickleExporter",
    "PickleImporter",
    "ZIPExporter",
    "ZIPImporter",
    "ISO_PARAMETER_KEYS",
    "TMD_FORMAT_TAG",
    "TMD_FORMAT_VERSION",
]
