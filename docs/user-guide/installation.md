# Installation

Requires **Python 3.8+** (see `requires-python` in `pyproject.toml`).

```bash
pip install truemapdata
```

For a development checkout:

```bash
pip install -r requirements.txt
pip install -e .
```

The CLI is available as **`tmd-process`** after install, or run **`python tmd_cli.py`** from a clone.

### Optional: roughness analysis (Surfalize)

To use **`tmd-process roughness`** (ISO 25178 areal parameters on `.tmd` files), install the optional extra. Surfalize is **GPL-3.0**—see [Working with TMD files](working-with-tmd-files.md#roughness-and-topography-surfalize).

```bash
pip install "truemapdata[roughness]"
```

Equivalent: `pip install "surfalize>=0.16.0"` on **Python 3.10+** so TrueMap **`.tmd`** I/O matches what this library writes (older Surfalize releases may reject some `.tmd` files; current upstream Surfalize is not importable on Python 3.8–3.9).
