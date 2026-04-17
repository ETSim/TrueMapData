# TMD Library: TrueMap Data Processing & Visualization

[![PyPI version](https://img.shields.io/pypi/v/truemapdata)](https://pypi.org/project/truemapdata/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/truemapdata)](https://pypi.org/project/truemapdata/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/truemapdata)](https://pypi.org/project/truemapdata/)
[![License](https://img.shields.io/github/license/ETSTribology/TrueMapData)](https://github.com/ETSTribology/TrueMapData/blob/main/LICENSE)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://etstribology.github.io/TrueMapData/)

The documentation workflow publishes with **GitHub Actions**; in the repository **Settings → Pages**, set **Build and deployment** source to **GitHub Actions** (once per repo or fork).

A Python library and CLI for **TrueMap v6** and **GelSight** TMD height maps: I/O, processing, visualization, map export, and 3D mesh generation.

<p align="center">
  <img src="https://raw.githubusercontent.com/ETSTribology/TrueMapData/refs/heads/main/image.svg" width="300" alt="TMD Processor Logo">
</p>

---

## Table of contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [CLI and mesh export](#cli-and-mesh-export)
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
| **Rich visualizations** | 3D surfaces, 2D heatmaps, profiles (matplotlib, plotly, seaborn, optional Polyscope) |
| **Export** | Displacement/normal maps, STL/OBJ/PLY/glTF/USD meshes, NumPy-friendly workflows |
| **Formats** | TrueMap v6-style TMD (v2) and legacy v1; GelSight-compatible layouts |
| **Python** | **3.8+** (see `requires-python` in [`pyproject.toml`](pyproject.toml)) |

---

## Installation

From [PyPI](https://pypi.org/project/truemapdata/):

```bash
pip install truemapdata
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

## CLI and mesh export

- **Help:** `tmd-process --help` or `python tmd_cli.py --help`
- **CLI examples (markdown):** `tmd-process visualize examples` or `python tmd_cli.py visualize examples`

**Maps** (normal, height, AO, etc.):

```bash
tmd-process maps list
tmd-process maps height path/to/file.tmd --output-file height.png
```

**Meshes** (height field → 3D surface):

```bash
tmd-process mesh formats
tmd-process mesh stl path/to/file.tmd --quality high
tmd-process mesh generate path/to/file.tmd --format stl --method adaptive --quality high
```

- **`adaptive`** — error-driven refinement; lower **`error_threshold`** and higher **`max_triangles`** yield finer meshes.
- **`quadtree`** — hierarchical grid refinement; **`max_subdivisions`** caps depth. Python API also exposes **`detail_boost`** on `ExportConfig`.

Documentation: [GitHub Pages](https://etstribology.github.io/TrueMapData/) (same URL as [`pyproject.toml` `[project.urls] Documentation`](pyproject.toml)).

---

## TMD file format

### TrueMap v6 (v2 layout)

| Field | Description |
|-------|-------------|
| **Header (32 bytes)** | ASCII prefix, e.g. `Binary TrueMap Data File v2.0` (null-padded) |
| **Comment (24 bytes in reader path)** | ASCII comment / padding |
| **Dimensions (8 bytes)** | Little-endian `uint32` **width**, **height** |
| **Spatial info (16 bytes)** | `float32` **x_length**, **y_length**, **x_offset**, **y_offset** |
| **Height data** | `float32` row-major block, logical shape **(height, width)** |

### GelSight

Same overall binary layout as v2; comments and naming may differ. The reader uses header cues and metadata to interpret the grid.

### Legacy v1

Shorter header (28-byte offset to dimensions), then **width**, **height**, **x_length**, **y_length** (two floats only), then height samples.

---

## Sample data

Examples in docs and tests use **paths like `path/to/file.tmd`** or **synthetic heightmaps**. Committing large proprietary `.tmd` fixtures is avoided; use your own captures or published attachments.

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
