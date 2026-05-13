## Learned User Preferences

- Prefer the agent run install, test, build, and documentation commands locally and report concrete pass or fail output instead of only suggesting commands.
- For large coverage pushes or broad refactors, prefer phased scope and confirm before defaulting to an everything-at-once change set.
- Prefer the published MkDocs Material site as the place for deep reference and diagrams, with a slimmer README that links into the site; when reorganizing docs, favor a clear split between user guide, reference, and developer-oriented pages (similar in spirit to IPC Toolkit–style manuals) without unnecessary duplication.
- Treat mesh, export, and CLI plans as scope-locked unless the user explicitly widens the scope.
- For refactors, prefer consolidating into existing package boundaries (for example sequential wear under `tmd.sequence`, tribology metrics under `tmd.surface.metrics`) and keeping modules reasonably sized rather than many overlapping files.
- For notebooks, prefer minimal shared helpers and removing thin notebook-only shims when the user asks for a cleaner layout.
- Keep in-repo developer AI hints (for example `llms.txt`) focused on TrueMapData workflows rather than generic MCP-only filler.

## Learned Workspace Facts

- This repository is TrueMapData: TrueMap and GelSight TMD height-map I/O and processing, optional ISO roughness via Surfalize (GPL-gated path), sequences, defects, map generators, and mesh generation or apply-on-mesh OBJ and MTL bundling; some fixtures and notebooks reference TextureFriction under `E:/master/TextureFriction/` on this machine.
- Apply-on-mesh physical tiling derives target atlas and capture tile sizes from the template OBJ X and Z span converted with `obj_units_to_mm` (default 1000 for meter-based OBJ), combined with `mm_per_pixel` from TMD metadata or `--tmd-mm-per-pixel`, and TMD `x_length` and `y_length` metadata for tile sizing; UV mode expects UV coordinates in the template OBJ.
- On this development machine, `python -m pytest` can fail during collection when NumPy 2.x is paired with SciPy (or other wheels) built for NumPy 1.x; fixing requires aligned pins or a clean virtual environment with compatible versions.
- Documentation builds use MkDocs; `requirements-docs.txt` pins `mkdocs-material`, and GitHub Actions workflows in `.github/workflows/` include docs and release automation for this repo.
- Sequential wear tables, incremental wear series, and related helpers live in `tmd.sequence.wear_analysis`; translation-only phase-correlation alignment is in `tmd.sequence.alignment`, with `TMDSequence.align_height_maps_phase_fft`.
- Tribology-oriented height-map metrics (core, auxiliary, dashboard, optional Surfalize bridge) live under `tmd.surface.metrics`.
