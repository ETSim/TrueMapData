#!/usr/bin/env python3
"""
Pickle Exporter/Importer for TMD Data.

This module provides concrete implementations for exporting and importing
TMD data in the Pickle format.
"""

import os
import logging
import pickle
from pathlib import Path
from typing import Any, Dict

from .base import TMDDataExporter, TMDDataImporter

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def _export_pickle(data: Any, output_path: str) -> str:
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        with open(output_path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Data exported to Pickle file: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error exporting Pickle: {e}")
        raise

def _load_pickle(file_path: str) -> Any:
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Pickle file not found: {file_path}")
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        logger.info(f"Data loaded from Pickle file: {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading Pickle file: {e}")
        raise

class PickleExporter(TMDDataExporter):
    def export(self, data: Dict[str, Any], output_path: str) -> str:
        return _export_pickle(data, output_path)

class PickleImporter(TMDDataImporter):
    def load(self, file_path: str) -> Dict[str, Any]:
        return _load_pickle(file_path)
