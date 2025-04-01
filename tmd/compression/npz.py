#!/usr/bin/env python3
"""
NPZ Exporter/Importer for TMD Data.

This module provides concrete implementations for exporting and importing
TMD data in the NPZ format (with optional compression).
"""

import os
import logging
import numpy as np
from pathlib import Path
from typing import Any, Dict

from .base import TMDDataExporter, TMDDataImporter

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def _export_npz(data: Dict[str, Any], output_path: str, compress: bool = True) -> str:
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        if compress:
            np.savez_compressed(output_path, **data)
        else:
            np.savez(output_path, **data)
        logger.info(f"Data exported to NPZ file: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error exporting NPZ: {e}")
        raise

def _load_npz(file_path: str) -> Dict[str, Any]:
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {file_path}")
    try:
        npz_data = np.load(file_path, allow_pickle=True)
        data = {key: npz_data[key] for key in npz_data.files}
        logger.info(f"Data loaded from NPZ file: {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading NPZ file: {e}")
        raise

class NPZExporter(TMDDataExporter):
    def __init__(self, compress: bool = True):
        self.compress = compress

    def export(self, data: Dict[str, Any], output_path: str) -> str:
        return _export_npz(data, output_path, compress=self.compress)

class NPZImporter(TMDDataImporter):
    def load(self, file_path: str) -> Dict[str, Any]:
        return _load_npz(file_path)
