# Contributing to TrueMapData

Thank you for your interest in improving TrueMapData. This document is the main entry point for local development, checks, and how we use GitHub.

## Issues and pull requests

- **Bug reports and feature requests:** use the templates under [`.github/ISSUE_TEMPLATE/`](.github/ISSUE_TEMPLATE/).
- **Pull requests:** follow [`.github/PULL_REQUEST_TEMPLATE.md`](.github/PULL_REQUEST_TEMPLATE.md) and link related issues when applicable.
- **Security:** do not open public issues for vulnerabilities. See [`SECURITY.md`](SECURITY.md).

## Development environment

### Option A: requirements files (matches README)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

For notebooks, docs tooling, and extra dev dependencies:

```bash
pip install -r requirements-dev.txt
```

### Option B: editable install with extras

```bash
pip install --upgrade pip
pip install -e ".[dev,docs]"
```

Optional groups from [`pyproject.toml`](pyproject.toml) include `viz`, `mesh`, `advanced`, and `full` (Polyscope, glTF/Open3D-style workflows, etc.). Install what you need for your change; CI does not require every optional extra locally.

### Dev Container

For a reproducible environment (Python 3.12 image, dependencies, and suggested editor extensions), open the repository in a Dev Container using [`.devcontainer/devcontainer.json`](.devcontainer/devcontainer.json). Full **Python 3.8–3.12** compatibility is validated in GitHub Actions, not necessarily inside the container image.

See [`.devcontainer/README.md`](.devcontainer/README.md) for notes on `.tmd` files and common commands.

## Before you open a PR

### Tests

```bash
pytest
```

Options, logging, markers, and default `addopts` (including coverage reports) are defined in **[`pytest.ini`](pytest.ini)** at the repository root.

### Linting

Continuous integration runs **Ruff** (see [`.github/workflows/ruff.yml`](.github/workflows/ruff.yml)). Run it locally before pushing:

```bash
python -m ruff check .
```

Optional: `python -m ruff format .` to apply the formatter locally (CI currently runs **`ruff check`** only; see [`.github/workflows/ruff.yml`](.github/workflows/ruff.yml)).

The `[project.optional-dependencies] dev` group in `pyproject.toml` still lists tools such as Black, isort, Flake8, and mypy for contributors who use them; **Ruff is the linter aligned with CI**.

### Pre-commit

If you use Git hooks, install them after cloning:

```bash
pre-commit install
```

Configuration lives in [`.pre-commit-config.yaml`](.pre-commit-config.yaml).

### Documentation

With docs dependencies installed:

```bash
mkdocs serve
```

For binding to all interfaces inside a container, use `mkdocs serve -a 0.0.0.0:8000` and forward port **8000** if needed.

For **AI-assisted editing**, the repository root [`llms.txt`](https://github.com/ETSTribology/TrueMapData/blob/main/llms.txt) lists canonical TrueMapData links; the published site expands that with **[AI assistants (llms.txt)](https://etstribology.github.io/TrueMapData/developers/ai-tooling/)** (optional editor tooling, verification, and security notes).

## Sample and proprietary `.tmd` data

Large proprietary captures are not committed as a rule. **Whitelisted** `.tmd` paths under `examples/` and the wear-volume second-frame binary under `tests/fixtures/` used by notebooks and `tests/cli/test_example_tmds_smoke.py` are **tracked in git** (see `.gitignore` negation rules for `examples/`). CI runs `pytest` on a clean checkout without a generator step.

Do **not** replace the canonical bytes of `examples/gelsight/circle_0mm_100g_heightmap_linear_detrend.tmd`, `examples/v1/Dime.tmd`, or `examples/v2/Dime.tmd` when refreshing fixtures. The GelSight `circle_0mm` file may be a **full-resolution** capture (large on disk); shrinking it for convenience requires an explicit project decision (for example Git LFS), not an automated rewrite.

To refresh the wear-volume smoke **second frame** only, regenerate `tests/fixtures/gelsight_wear_volume_second_frame.tmd` locally by loading `examples/gelsight/circle_0mm_100g_heightmap_linear_detrend.tmd` (read-only) and writing that fixture path with the same grid and metadata-derived `x_length` / `y_length`. Do not write under `examples/gelsight/` except the frozen `circle_0mm` asset you maintain yourself.

To work with your own data, add files under `examples/` locally; only the whitelisted fixture filenames are tracked—other `examples/**/*.tmd` paths stay ignored.

## Code style

Match existing patterns in the `tmd` package: naming, imports, and typing conventions. When in doubt, run Ruff and the test suite and mirror surrounding modules.

## Questions

Open a [discussion or issue](https://github.com/ETSTribology/TrueMapData/issues) for design questions before large refactors so effort stays aligned with project goals.
