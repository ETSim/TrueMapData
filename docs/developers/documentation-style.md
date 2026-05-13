# Documentation style

This site favors **clear structure**, **copy-pasteable commands**, and **diagrams** where they replace long prose. The layout follows common scientific-library practice: **user guide**, **reference**, and **developers** sections (similar in spirit to projects that use Sphinx, without switching stacks).

## Conventions

- **Admonitions** — Use `!!! note`, `!!! tip`, and `!!! warning` for licensing caveats (for example Surfalize GPL), Python version constraints, and operational warnings.
- **Mermaid** — Use `flowchart` for navigation or data-flow overviews; use `sequenceDiagram` when step order matters (for example load path from file to `height_map`).
- **Tables** — Prefer tables for CLI command summaries; keep examples on one line when reasonable.
- **Links** — Prefer **internal** links between MkDocs pages so `mkdocs build --strict` validates them. Use absolute GitHub URLs only when pointing at repository files outside `docs/` (for example `CONTRIBUTING.md` in the repo root).

## Material for MkDocs

Behavior of tabs, search, navigation, and extensions is controlled in **`mkdocs.yml`**. For features not documented here, see the upstream reference:

- [Material for MkDocs reference](https://squidfunk.github.io/mkdocs-material/reference/)

## README vs site

**PyPI** renders the root **`README.md`**. Keep it welcoming and short: feature summary, install, minimal Python example, links to this site for **CLI reference** and **binary format** details. Avoid duplicating large tables in both places.
