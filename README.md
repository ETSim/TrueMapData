# TMD Library: TrueMap Data Processing & Visualization

[![PyPI version](https://img.shields.io/pypi/v/truemapdata)](https://pypi.org/project/truemapdata/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/truemapdata)](https://pypi.org/project/truemapdata/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/truemapdata)](https://pypi.org/project/truemapdata/)
[![License](https://img.shields.io/github/license/ETSTribology/TrueMapData)](https://github.com/ETSTribology/TrueMapData/blob/main/LICENSE)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://etstribology.github.io/TrueMapData/)

The documentation site builds from **GitHub Actions**. In the repository **Settings → Pages**, set **Build and deployment** to **GitHub Actions** once per repo or fork.

Python library and **`tmd-process` CLI** for **TrueMap v6** and **GelSight** `.tmd` height maps: binary I/O, filters and transforms, visualization, derived maps (normal, displacement, bump, hillshade, ambient occlusion, tribology proxy maps, and related generators), defect detection, multi-frame sequences (GIF, video, PowerPoint), compression helpers, synthetic terrain for fixtures, and mesh export including apply-on-mesh OBJ/MTL bundles with tiling from template bounds and TMD metadata.

Areal ISO 25178 roughness uses the optional `truemapdata[roughness]` extra, which installs **Surfalize** under **GPL-3.0** (see `pyproject.toml`; it is intentionally not part of the permissive `full` extra).

Typical users: tribology and surface metrology, lab pipelines that need NumPy height arrays, and graphics workflows that want measured displacement or normal maps.

<p align="center">
  <img src="https://raw.githubusercontent.com/ETSTribology/TrueMapData/refs/heads/main/image.svg" width="300" alt="TMD Processor Logo">
</p>

---

## Table of contents

- [Features](#features)
- [Documentation](#documentation)
- [Installation](#installation)
- [Usage](#usage)
- [CLI](#cli)
- [TMD file format](#tmd-file-format)
- [Sample data](#sample-data)
- [Visual examples](#visual-examples)
- [GelSight sequence visualization](#gelsight-sequence-visualization)
- [Contributing](#contributing)
- [License](#license)

---

## Features

| Feature | Description |
|---------|-------------|
| **Visualizations** | 3D surfaces, 2D heatmaps, profiles via matplotlib, plotly, and seaborn; optional Polyscope in the `advanced` extra |
| **Map export** | `maps` CLI and library generators: normal, displacement, bump, hillshade, ambient occlusion, parallax-style stacks, roughness-style textures, and related outputs |
| **Meshes** | STL, OBJ, PLY, glTF, USD; optional `mesh` extra for trimesh, pygltflib, and numpy-stl |
| **Apply-on-mesh** | Bundle height-derived textures onto a template OBJ/MTL with physically motivated tiling from template span, `mm_per_pixel`, and capture size metadata (see the CLI reference) |
| **Sequences** | `sequence` commands align and crop stacks, with exporters for GIF, video, and PowerPoint (OpenCV-backed) |
| **Defects** | `defect` commands for pits, peaks, scratches, cracks, and directionality-style anomalies |
| **Roughness** | `roughness` subcommands for ISO 25178 areal metrics via optional Surfalize (`truemapdata[roughness]`; GPL-3.0, not bundled in `full`) |
| **Tribology** | `tribology` CLI and `tmd.surface.metrics` helpers: preferred texture axis, bearing-style contact curve, **`tribology plot`** PNG dashboard, shear/debris/summit map types; lubrication ISO volumes via Surfalize subcommand |
| **Wear toolkit** | `tmd-process wear` (also installed as **`tmd-wear`**) for Abbott / bearing curves, wear volume series, hazard maps, debris-pocket scores, scratch evolution, slip axis, and roughness trajectories on aligned sequences |
| **Compression** | `compress` subcommands for npy, npz, zip, mat, and pickle-oriented workflows |
| **Terrain** | `terrain` CLI and `TMDTerrain` helpers for synthetic heightmaps used in tests and demos |
| **Formats** | TrueMap v6-style TMD (v2) and legacy v1; GelSight-compatible layouts |
| **Python** | **3.8+** (`requires-python` in [`pyproject.toml`](pyproject.toml)); optional groups `viz`, `mesh`, `advanced`, `dev`, and `docs` are listed there |

---

## Documentation

Full guides, developer notes, and the **CLI reference** live on **[GitHub Pages](https://etstribology.github.io/TrueMapData/)** (same URL as [`pyproject.toml` `[project.urls] Documentation`](pyproject.toml)).

| Section | What you will find |
|---------|-------------------|
| [User guide](https://etstribology.github.io/TrueMapData/user-guide/installation/) | Install, getting started, TMD binary layout, visualization, export, [tribology metrics](https://etstribology.github.io/TrueMapData/user-guide/tribology-metrics/), [sequential wear analysis](https://etstribology.github.io/TrueMapData/user-guide/sequential-wear-analysis/) |
| [CLI reference](https://etstribology.github.io/TrueMapData/reference/cli/) | Command tables, mesh and apply-on-mesh notes, defect defaults |
| [Developers](https://etstribology.github.io/TrueMapData/developers/contributing/) | Contributing, building docs, doc style, optional MCP tooling |

---

## Installation

From [PyPI](https://pypi.org/project/truemapdata/):

```bash
pip install truemapdata
```

Optional dependency groups (see [`pyproject.toml`](pyproject.toml) for pins and Python version gates):

```bash
pip install "truemapdata[mesh,viz,advanced]"
pip install "truemapdata[roughness]"   # Surfalize (GPL-3.0); Python 3.10+ only
```

From a git checkout:

```bash
pip install -r requirements.txt
pip install -e .
```

The console entry point is **`tmd-process`**. From the repository you can also run **`python tmd_cli.py`**.

---

## Usage

```python
from tmd import TMD

data = TMD.load("path/to/your/file.tmd")
height_map = data.height_map
metadata = data.metadata
```

Use a real `.tmd` from your instrument or pipeline; **no sample `.tmd` files are committed** to this repository (size, licensing). You can generate synthetic terrain for tests via `TMDTerrain` / the `terrain` CLI.

---

## CLI

Entry points: **`tmd-process`** (full toolkit) and **`tmd-wear`** (wear-oriented subset; same flags as `tmd-process wear`). From a repository checkout you can also run **`python tmd_cli.py`**. Global help: `tmd-process --help`. Embedded examples: `tmd-process visualize examples`.

Top-level groups include **`config`**, **`cache`**, **`compress`**, **`maps`**, **`mesh`**, **`sequence`**, **`roughness`**, **`tribology`**, **`wear`**, **`defect`**, **`visualize`**, and **`terrain`**, plus **`info`**, **`version`**, and **`check`**.

The **[CLI reference](https://etstribology.github.io/TrueMapData/reference/cli/)** lists flags, `maps` / `mesh` details, apply-on-mesh tiling notes, defect defaults, and the full **`tribology`** and **`wear`** subcommand tables. For **aligned stacks**, volume series, slip axis, and scratch evolution, see **[Sequential wear analysis](https://etstribology.github.io/TrueMapData/user-guide/sequential-wear-analysis/)**. Use **`--help` on each subcommand** for the live flag list.

---

## TMD file format

The canonical byte layout, version 1 vs 2, GelSight quirks, and endianness are documented under **[Working with TMD files](https://etstribology.github.io/TrueMapData/user-guide/working-with-tmd-files/)** on the doc site (tables and diagrams). At a glance: v2 uses a fixed header plus **`float32`** heights in row-major order **`(height, width)`**; v1 is a shorter header with the same raster layout.

---

## Sample data

Examples in docs and tests use **paths like `path/to/file.tmd`** or **synthetic heightmaps**. Committing large proprietary `.tmd` fixtures is avoided; use your own captures or published attachments.

To populate the canonical **example paths** used by notebooks and `tests/cli/test_example_tmds_smoke.py` (`examples/gelsight/…`, `examples/v1/Dime.tmd`, `examples/v2/Dime.tmd`), run from the repo root:

```bash
python examples/generate_example_tmds.py
```

Then try wear metrics, for example:

```bash
tmd-wear bearing curve examples/gelsight/circle_0mm_100g_heightmap_linear_detrend.tmd --json
tmd-wear volume-series examples/gelsight/circle_0mm_100g_heightmap_linear_detrend.tmd examples/gelsight/circle_worn_0mm_100g_heightmap_linear_detrend.tmd --json
```

---

## Visual examples

### Height map statistics (illustrative)

| Metric | Value |
|--------|-------|
| **Shape** | (200, 200) |
| **Min** | 0.0 |
| **Max** | 1.0 |
| **Mean** | 0.41206 |
| **Std Dev** | 0.18863 |

### Image gallery

| **3D surface** | **X profile** |
|----------------|----------------|
| <img src="https://github.com/user-attachments/assets/faa4db7d-62ee-47e9-8883-4b8d4af13eb9" width="400"> | <img src="https://github.com/user-attachments/assets/fcf95e3c-5810-4dfd-93f0-06e98297490b" width="400"> |

| **Displacement** | **Y profile** |
|--------------------|---------------|
| <img src="https://github.com/user-attachments/assets/0a89659c-5af0-4a53-969b-9a96f04dac0a" width="400"> | <img src="https://github.com/user-attachments/assets/d48e4158-76ba-42fa-8399-7ceb60241925" width="400"> |

| **STL** | **3D crop** |
|---------|-------------|
| <img src="https://github.com/user-attachments/assets/885b363a-10da-44b8-a574-bcd4848c2837" width="400"> | <img src="https://github.com/user-attachments/assets/f00dcec9-6a2c-4080-b643-cb42ee5f3193" width="400"> |

| **Plotly** | **HeightMap** |
|------------|---------------|
| <img src="https://github.com/user-attachments/assets/2cb7d052-6b63-4435-af7b-04becaf1a594" width="400"> | <img src="https://github.com/user-attachments/assets/f4a4b855-cf83-4971-ad15-f393fb52e03b" width="400"> |

| **Bump maps** | **Smoothing** |
|---------------|---------------|
| <img src="https://github.com/user-attachments/assets/cf1cc89a-0a35-4f6c-966c-22ea193e0a70" width="400"> | <img src="https://github.com/user-attachments/assets/d227f561-6d39-45fb-94eb-0a94d66fc948" width="400"> |

---

## GelSight sequence visualization

```bash
python visualize_gelsight_sequence.py --show
python visualize_gelsight_sequence.py --files path/to/a.tmd path/to/b.tmd --format 3d --z-scale 1.5 2.0 --show
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for local setup, tests, Ruff, docs, and how to use the GitHub issue and PR templates. Report security issues privately as described in [SECURITY.md](SECURITY.md). For a reproducible editor environment, use the [Dev Container](https://containers.dev/) definition in [`.devcontainer/devcontainer.json`](.devcontainer/devcontainer.json).

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
