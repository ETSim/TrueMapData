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

Options and coverage defaults are defined under `[tool.pytest.ini_options]` in [`pyproject.toml`](pyproject.toml).

### Linting

Continuous integration runs **Ruff** (see [`.github/workflows/ruff.yml`](.github/workflows/ruff.yml)). Run it locally before pushing:

```bash
ruff check .
```

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

## Sample and proprietary `.tmd` data

**Sample `.tmd` files are not committed** to this repository (size and licensing). Use your own captures, published attachments, or synthetic heightmaps (for example `TMDTerrain` / the `terrain` CLI) when developing or adding tests. Do not commit large binary fixtures without maintainer agreement.

## Code style

Match existing patterns in the `tmd` package: naming, imports, and typing conventions. When in doubt, run Ruff and the test suite and mirror surrounding modules.

## Questions

Open a [discussion or issue](https://github.com/ETSTribology/TrueMapData/issues) for design questions before large refactors so effort stays aligned with project goals.
