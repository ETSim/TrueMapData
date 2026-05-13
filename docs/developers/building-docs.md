# Building the docs

The public site is built with **[MkDocs](https://www.mkdocs.org/)** and the **[Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)** theme. Configuration lives in **[`mkdocs.yml`](https://github.com/ETSTribology/TrueMapData/blob/main/mkdocs.yml)** at the repository root; Markdown sources live under **`docs/`**.

## Install tooling

**CI / pinned set** (matches [.github/workflows/docs.yml](https://github.com/ETSTribology/TrueMapData/blob/main/.github/workflows/docs.yml)):

```bash
pip install -r requirements-docs.txt
pip install -e .
```

**Editable install with the `docs` extra** (minimum versions from [`pyproject.toml`](https://github.com/ETSTribology/TrueMapData/blob/main/pyproject.toml)):

```bash
pip install -e ".[docs]"
```

## Local preview

```bash
mkdocs serve
```

Inside a Dev Container or remote environment, bind all interfaces if needed:

```bash
mkdocs serve -a 0.0.0.0:8000
```

Then open the printed URL (default port **8000**).

## Strict build (same as CI)

```bash
python -m mkdocs build --strict
```

`--strict` turns warnings (for example broken internal links) into failures. Fix links before merging doc changes.

## Publishing

On push to **`main`**, [.github/workflows/docs.yml](https://github.com/ETSTribology/TrueMapData/blob/main/.github/workflows/docs.yml) builds the site and deploys the `site/` artifact to **GitHub Pages**. Repository **Settings → Pages → Build and deployment** should use **GitHub Actions** as the source.

## Tests and lint (before large doc edits)

Doc-only changes rarely need the full suite, but broken examples or cross-links are easier to catch when the tree is healthy. From the repository root, after installing dev dependencies (see [Contributing](contributing.md) in this site, or the root **`CONTRIBUTING.md`** in the repo):

- **`python -m pytest tests/ -q`** — full test run (coverage thresholds match CI when using the same `pytest.ini` defaults).
- **`python -m ruff check .`** — same static analysis as the [Ruff workflow](https://github.com/ETSTribology/TrueMapData/blob/main/.github/workflows/ruff.yml) on GitHub.

Use **`python -m mkdocs build --strict`** (above) as the final gate for documentation merges.
