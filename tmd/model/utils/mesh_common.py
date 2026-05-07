"""
Mesh preparation for OBJ/PLY/NVBD exporters.

Provides ``prepare_mesh_for_export`` used after ``ModelExporter.create_mesh_from_heightmap``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from ..base import MeshData

logger = logging.getLogger(__name__)


def prepare_mesh_for_export(mesh: MeshData, config_dict: Dict[str, Any]) -> Optional[MeshData]:
    """
    Apply normals, UVs, and optional optimization based on export configuration.

    Args:
        mesh: Mesh produced by triangulation
        config_dict: Typically ``config.__dict__`` from :class:`~tmd.model.base.ExportConfig`

    Returns:
        The same mesh instance with derived attributes updated, or ``None`` if ``mesh`` is None.
    """
    if mesh is None:
        return None
    try:
        if config_dict.get("calculate_normals", True):
            mesh.ensure_normals(force_recalculate=False)
        gen_uv = config_dict.get("generate_uvs") or config_dict.get("texture")
        if gen_uv:
            uv_method = config_dict.get("uv_method", "planar")
            mesh.ensure_uvs(method=str(uv_method))
        if config_dict.get("optimize"):
            mesh.optimize()
        return mesh
    except Exception as exc:
        logger.warning("prepare_mesh_for_export: continuing with partially processed mesh: %s", exc)
        return mesh
