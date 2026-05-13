# AI assistants and project documentation

Use **TrueMapData-owned documentation** before generic model memory or third-party MCP snippets. This page complements the repository root **`llms.txt`** file (a compact link index for crawlers and editors that support it).

## Start here (canonical)

| Source | Role |
|--------|------|
| **[`llms.txt`](https://github.com/ETSTribology/TrueMapData/blob/main/llms.txt)** (repo root) | Single-page map: published docs URLs, key `tmd/` paths, contributing and security links. Prefer the raw GitHub URL when a tool needs a stable fetch target. |
| **[Documentation home](../index.md)** | User guide, diagrams, and navigation for this site. |
| **[CLI reference](../reference/cli.md)** | Command tables and flags; `tmd-process --help` remains the runtime ground truth. |
| **[Contributing](contributing.md)** / **[`CONTRIBUTING.md`](https://github.com/ETSTribology/TrueMapData/blob/main/CONTRIBUTING.md)** | Environment, `pytest`, Ruff, pre-commit, PR expectations. |
| **[Building the docs](building-docs.md)** | `mkdocs serve`, `mkdocs build --strict`, docs dependencies. |

PyPI and GitHub skimming: **[`README.md`](https://github.com/ETSTribology/TrueMapData/blob/main/README.md)** stays intentionally short and points into this site for depth.

## Domain facts (avoid inventing behavior)

- **TMD** — Binary height maps (TrueMap v6, GelSight-class). Layout, metadata, and NumPy-facing usage: [Working with TMD files](../user-guide/working-with-tmd-files.md).
- **Apply-on-mesh tiling** — Derived from template OBJ span, `obj_units_to_mm` (default **1000** for meter-based OBJ), TMD `mm_per_pixel` (metadata or `--tmd-mm-per-pixel`), and `x_length` / `y_length`. UV path expects UVs on the template. [Exporting data](../user-guide/exporting-data.md) and the CLI reference above.
- **Roughness / Surfalize** — Optional, GPL-gated integration; see user guide and CLI notes before changing that path.
- **Sequential wear** — Translation alignment (`phase_fft` / OpenCV), volume vs reference or incremental, slip-axis and scratch series: [`tmd.sequence.wear_analysis`](https://github.com/ETSTribology/TrueMapData/blob/main/tmd/sequence/wear_analysis.py) and [`TMDSequence`](https://github.com/ETSTribology/TrueMapData/blob/main/tmd/core/sequence.py). User-facing guide: [Sequential wear analysis](../user-guide/sequential-wear-analysis.md). CLI: `tmd-wear` / `tmd-process wear`.
- **Tribology dashboard metrics** — Core helpers under `tmd.surface.metrics`; user guide: [Tribology metrics](../user-guide/tribology-metrics.md).
- **Fixtures** — Sample `.tmd` files are **not** shipped in the repository (size and licensing). Use local captures, published attachments, or synthetic surfaces (for example `TMDTerrain` / `terrain` CLI) for tests and examples.

## Where implementation usually lives

| Topic | Starting files |
|-------|----------------|
| Load / core model | `tmd/core/tmd.py` |
| CLI entry (`tmd-process`) | `tmd/cli/main.py`, `tmd/cli/commands/` |
| Mesh formats | `tmd/model/`, `tmd/cli/apps/export_mesh_app.py` |
| Map generators / PNG export | `tmd/image/maps/`, `tmd/image/export/` |
| Plotting | `tmd/plotters/` |
| Sequences / batch | `tmd/sequence/`, `tmd/cli/apps/sequence_app.py` |
| Phase-correlation alignment | `tmd/sequence/alignment.py` (`TMDSequence.align_height_maps_phase_fft`) |
| Sequential wear metrics | `tmd/sequence/wear_analysis.py`, `tmd/core/sequence.py` |
| Tribology / surface metrics | `tmd/surface/metrics/` |
| Wear CLI (`tmd-wear`) | `tmd/cli/apps/wear_app.py` |

Match existing patterns in the touched package; CI runs **Ruff** and **pytest** as in `CONTRIBUTING.md`.

## Verify before you claim “done”

- `pytest`
- `ruff check .`
- `python -m mkdocs build --strict` (with docs extras or `requirements-docs.txt`)

Local environments sometimes break on **NumPy 2.x** wheels mixed with packages built for NumPy 1.x; align pins or use a clean venv if collection fails.

## Optional editor MCP (non-project-specific)

**Model Context Protocol** integrations in editors (for example Cursor) are optional. Use vendor or registry-backed **library-doc** tools only when you change **third-party** APIs (Typer, Matplotlib, Plotly, mesh stacks). For **TrueMapData** semantics, this site, `llms.txt`, tests, and `--help` output override generic snippets.

## Security

Do **not** commit secrets, tokens, or personal data in documentation, tracked editor config, or MCP setup. Keep credentials local-only.
