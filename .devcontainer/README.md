# Dev Container notes

This configuration uses **Python 3.12** on Debian Bookworm with runtime packages commonly needed for **OpenCV** (`libglib2.0-0`, `libgl1`). After the container is created, dependencies are installed from `requirements.txt`, the package is installed editable, `requirements-dev.txt` is applied, **Ruff** is installed for CI parity, and **pre-commit** hooks are installed.

## Working with `.tmd` files

The repository does not ship large proprietary `.tmd` binaries. For **small synthetic fixtures** at the canonical paths (`examples/gelsight/`, `examples/v1/`, `examples/v2/`), run **`python examples/generate_example_tmds.py`** from the repo root after cloning. Otherwise mount or copy your own captures (or use `TMDTerrain` / the `terrain` CLI).

## Useful commands

- **Tests:** `pytest`
- **Lint (matches CI):** `ruff check .`
- **CLI:** `tmd-process --help`, `tmd-wear --help`, or `python tmd_cli.py --help`
- **Docs:** `mkdocs serve -a 0.0.0.0:8000` (port **8000** is forwarded by default)

Python **3.8–3.12** is validated in GitHub Actions; the container uses a single modern interpreter for day-to-day work.
