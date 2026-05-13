# CLI reference

Console entry point: **`tmd-process`** (or **`python tmd_cli.py`** from a repository checkout).

| Command | Purpose |
|---------|---------|
| `tmd-process --help` | Top-level commands and global options |
| `tmd-process visualize examples` | Embedded markdown examples for visualization |

!!! tip "Ground truth"
    Subcommands and flags can change between releases. When in doubt, use `--help` on the command you are running.

## Top-level commands

| Command | Feature | Example |
|---------|---------|---------|
| `info` | Inspect one `.tmd` (metadata, optional height sample) | `tmd-process info path/to/file.tmd` |
| `version` | Print CLI and core `__version__` | `tmd-process version` |
| `check` | Verify optional dependencies / environment | `tmd-process check` |
| `maps` | Export image maps (normal, height, AO, …) | `tmd-process maps list` |
| `mesh` | Height field → STL / OBJ / PLY / glTF / USD | `tmd-process mesh formats` |
| `visualize` | 2D/3D plots, profiles, Polyscope, backends | `tmd-process visualize basic path/to/file.tmd` |
| `sequence` | Align frames and export maps/meshes from a run | `tmd-process sequence align a.tmd b.tmd c.tmd -o ./aligned` |
| `roughness` | ISO 25178 areal roughness (needs **Surfalize**) | `tmd-process roughness file path/to/file.tmd --quick` |
| `tribology` | Texture axis, bearing curve, **PNG dashboard**, lubrication ISO volumes (latter needs **Surfalize**) | `tmd-process tribology plot path/to/file.tmd -o tribo.png --plane-removal mean` |
| `wear` | Wear-oriented metrics: Abbott / bearing curve, wear volume series, hazard maps, scratch evolution, slip axis (also reachable as **`tmd-wear`**) | `tmd-process wear bearing curve path/to/file.tmd --json` |
| `terrain` | Synthetic heightmaps and texture exports | `tmd-process terrain generate perlin --width 512 --height 512 -o ./synthetic` |
| `compress` | Downsample / quantize / combined / batch TMDs | `tmd-process compress downsample path/to/file.tmd --scale 0.5` |
| `cache` | Inspect or clear visualization cache | `tmd-process cache info` |
| `config` | Show or change CLI-related settings | `tmd-process config show` |
| `mesh apply` | Apply TMD maps onto an existing template OBJ mesh | `tmd-process mesh apply path/to/file.tmd -o ./bundle --template-mesh ./plane.obj --mode uv` |

## `maps` subcommands

| Subcommand | Feature | Example |
|------------|---------|---------|
| `list` | List supported map type names | `tmd-process maps list` |
| `batch` | Export many `.tmd` files from a directory | `tmd-process maps batch ./data -o ./textures --pattern "*.tmd"` |
| `all` | Export the default map set for one file | `tmd-process maps all path/to/file.tmd -o ./maps_out` |
| `height` | Grayscale height map PNG | `tmd-process maps height path/to/file.tmd --output-file height.png` |
| `normal` | Tangent-space normal map | `tmd-process maps normal path/to/file.tmd -o normal.png` |
| `ao` | Ambient occlusion map | `tmd-process maps ao path/to/file.tmd -o ao.png` |
| `bump` | Bump / relief map | `tmd-process maps bump path/to/file.tmd -o bump.png` |
| `hillshade` | Hillshade rendering | `tmd-process maps hillshade path/to/file.tmd --output-file hill.png` |
| `curvature` | Curvature visualization | `tmd-process maps curvature path/to/file.tmd --output-file curv.png` |
| `displacement` | Displacement map | `tmd-process maps displacement path/to/file.tmd --output-file disp.png` |

Run `tmd-process maps --help` for additional map types (`roughness`, `metallic`, `parallax_ao`, `angle`, `depth`, `shear_hazard`, `debris_pocket`, `summit_curvature`, …).

## `mesh` subcommands

| Subcommand | Feature | Example |
|------------|---------|---------|
| `formats` | List registered mesh exporters | `tmd-process mesh formats` |
| `generate` | Unified mesh export (`--format`, `--method`, `--quality`) | `tmd-process mesh generate path/to/file.tmd --format stl --method adaptive --quality high` |
| `stl` / `obj` / `ply` / `gltf` / `usd` | Shorthand export to one format | `tmd-process mesh stl path/to/file.tmd --quality high` |
| `batch` | Mesh many files from a directory | `tmd-process mesh batch ./data --output-dir ./meshes --pattern "*.tmd"` |
| `apply` | Apply-on-mesh flow (template mesh + maps → OBJ/MTL bundle) | `tmd-process mesh apply path/to/file.tmd -o ./bundle --template-mesh ./plane.obj --mode uv` |

### Mesh generation notes

- **`adaptive`** — error-driven refinement; lower **`error_threshold`** and higher **`max_triangles`** yield finer meshes.
- **`quadtree`** — hierarchical grid refinement; **`max_subdivisions`** caps depth. The Python API also exposes **`detail_boost`** on `ExportConfig`.

### Apply-on-mesh notes

- `mesh apply` is separate from `mesh generate`; it does not regenerate topology by default.
- Built-in templates are available with `--template-kind plane|sphere|cube`.
- `--mode uv` (default) keeps template geometry unchanged and binds maps in MTL.
- `--mode displace` is opt-in and emits displacement-ready material binding (`map_disp`) for downstream tools.
- `--uv-alignment-mode preserve` is default and keeps template UVs unchanged.
- Template OBJ units are treated as meters by default and converted with `--obj-units-to-mm` (default `1000`).
- Physical tiling uses measured scale:
    - `target_w_px = round(template_x_mm / tmd_mm_per_pixel)`
    - `target_h_px = round(template_z_mm / tmd_mm_per_pixel)`
    - `tile_w_px = round(tmd_x_length_mm / tmd_mm_per_pixel)`
    - `tile_h_px = round(tmd_y_length_mm / tmd_mm_per_pixel)`
- Use `--tmd-mm-per-pixel` to override metadata `mm_per_pixel` when needed.
- Generated MTL uses standard keys only: `map_Kd`, `map_Bump`, `map_disp`, `map_Pr`.

More context on apply-on-mesh and exporters: [Exporting data](../user-guide/exporting-data.md).

## Other useful `visualize` / `sequence` commands

| Command | Feature | Example |
|---------|---------|---------|
| `visualize 3d` | 3D surface (matplotlib / plotly / …) | `tmd-process visualize 3d path/to/file.tmd -o surf.html` |
| `visualize profile` | Row or column height profile | `tmd-process visualize profile path/to/file.tmd` |
| `visualize backends` | Which backends are installed | `tmd-process visualize backends` |
| `sequence export` | After `sequence align`, export maps + meshes for `*_aligned.tmd` | `tmd-process sequence export ./aligned_out` |

### `sequence align` (ORB vs TextureFriction SIFT)

| `--method` | Behavior |
|------------|----------|
| `auto`, `affine_ransac`, `phase_correlation` | ORB / RANSAC affine and optional phase refine (`tmd.surface.transformations`) |
| `sift` (alias `texture-friction`) | Two-pass SIFT/ECC in `tmd.sequence.alignment`; use **`--register-from height`** (scalar height warp) or **`normals`** (normals from height → align → warp heights) |
| `roughness batch` | CSV / stdout roughness over a folder | `tmd-process roughness batch ./tmds --pattern "*.tmd"` |
| `defect batch` | CSV / stdout defect counts and confidence over a folder | `tmd-process defect batch ./tmds -o ./defects.csv` |
| `compress quantize` | Reduce unique height levels | `tmd-process compress quantize path/to/file.tmd` |
| `cache clear` | Drop cached visualization artifacts | `tmd-process cache clear` |
| `config set` | Persist a config key | `tmd-process config set --help` |

## Defect command speed defaults

- `tmd-process defect file --json` and `tmd-process defect batch` use a fast summary mode by default.
- Opt into heavier computations only when needed:
    - `--include-mask`
    - `--include-overlay`
    - `--include-responses`
    - `--mask-output <file.png>`
    - `--overlay-output <file.png>`

## Roughness CLI

Roughness depends on optional **Surfalize** (GPL-3.0). Install and behavior details, including `roughness file`, `roughness batch`, and `roughness sequence`, are documented in [Working with TMD files — Roughness and topography (Surfalize)](../user-guide/working-with-tmd-files.md#roughness-and-topography-surfalize).

## `tribology` subcommands

Tribology metrics from `.tmd` height maps. The core helpers (axis, bearing curve, dashboard PNG, and proxy maps) are NumPy-only and MIT-licensed; **`lubrication`** requires optional **Surfalize** (GPL-3.0). See the user-guide page [Tribology metrics](../user-guide/tribology-metrics.md) for sign conventions and the Python API.

| Subcommand | Feature | Example |
|------------|---------|---------|
| `axis` | Preferred slip / texture axis from gradients, PSD lay, and directionality anomalies | `tmd-process tribology axis path/to/file.tmd --json --plane-removal mean` |
| `contact-curve` | Bearing / geometric contact fraction vs separation (CSV or JSON) | `tmd-process tribology contact-curve path/to/file.tmd --n 64 --plane-removal mean -o contact.csv` |
| `plot` | Multi-panel PNG dashboard: leveled height, bearing curve, slip axis, proxy maps | `tmd-process tribology plot path/to/file.tmd -o tribo.png --plane-removal mean --dpi 150` |
| `lubrication` | ISO 25178 functional / lubrication volumes (`Vvv`, `Vvc`, …) — requires **Surfalize** (GPL) | `tmd-process tribology lubrication path/to/file.tmd --no-json` |

### `tribology` flag highlights

- `--plane-removal {none,mean,median,surfalize}` — large-scale plane removal before the metric. `surfalize` is GPL-3.0 and requires the `[roughness]` extra; it falls back to `mean` with a warning if Surfalize is unavailable.
- `axis` adds `--gaussian-sigma FLOAT` to pre-smooth heights before the gradient tensor.
- `contact-curve` adds `--n INT` (separation samples), `--z-reference {mean,median}`, and `--output / -o PATH` for CSV.
- `plot` adds `--z-reference`, `--n`, `--dpi`, `--gaussian-sigma`, `--no-maps` (omit shear / debris / summit panels), and `--include-anomaly-angle` (fuse directionality-anomaly cue; slower, matches the default `axis` JSON output).
- `lubrication` defaults to `--json` on; pass `--no-json` for a Rich table. `--level/--no-level` toggles plane leveling before the ISO volumes.

## `wear` subcommands (also `tmd-wear`)

Wear-oriented surface metrics. The same Typer app is mounted under **`tmd-process wear`** and exposed as a standalone console entry point **`tmd-wear`** (`[project.scripts]` in `pyproject.toml`); the two forms accept identical flags. For raw stacks, prefer **`--align phase-fft`** for fast translation-only alignment, **`--align sift-normals`** (OpenCV) for TextureFriction-style normal-driven registration, or pre-align with `tmd-process sequence align`. User guide: [Sequential wear analysis](../user-guide/sequential-wear-analysis.md).

| Subcommand | Feature | Example |
|------------|---------|---------|
| `bearing curve` | Abbott–Firestone material ratio (`rmr_percent` vs depth from the peak) | `tmd-wear bearing curve path/to/file.tmd --samples 256 --rmr-at-depth 0.01,0.02,0.05 --json` |
| `roughness-track` | Surfalize roughness trajectory + Sp/Sv ratio, valley share, and `ssk_trend_hint` | `tmd-wear roughness-track --from-dir ./aligned --sort-by name -o track.csv` |
| `hazard-map` | Relative shear-hazard proxy PNG (`|grad| · |Laplace| · local RMS`) | `tmd-wear hazard-map path/to/file.tmd -o hazard.png --window 7 --normalize p98` |
| `debris-risk` | Heuristic debris-pocket score from pits + valleys + low gradient | `tmd-wear debris-risk path/to/file.tmd --score-png pockets.png --json` |
| `volume-series` | Wear volume and localization; optional `--align {none,phase-fft,opencv,sift,sift-normals}` before metrics | `tmd-wear volume-series a.tmd b.tmd --align phase-fft --reference 0 --json` |
| `scratch-evolve` | Scratch masks + pairwise `pairs` and full-frame `series`; optional `--align` / `-r` | `tmd-wear scratch-evolve --from-dir ./caps --json` |
| `slip-axis-series` | Per-frame slip-axis metrics on a stack; optional `--align` / `-r` | `tmd-wear slip-axis-series f0.tmd f1.tmd --json` |
| `slip-axis` | Preferred slip axis heuristics (gradient tensor + PSD wedges) | `tmd-wear slip-axis path/to/file.tmd --use-directionality-mask --json` |

### `wear` flag highlights

- Sequence-style commands (`roughness-track`, `volume-series`, `scratch-evolve`, `slip-axis-series`) accept either positional `PATHS…` **or** `--from-dir DIR` with `--pattern "*.tmd"` and `--sort-by {name,mtime}`; mixing both raises an error.
- `bearing curve`: `--samples INT` (depth samples), `--rmr-at-depth "d1,d2,…"` (query depths in the same units as the height map, typically mm), `--output-csv / -o PATH`, `--json`.
- `roughness-track` requires **Surfalize**; reports Sa, Sq, Sp, Sv, Sz, Ssk, Sku trajectories, Sp/Sv ratio, valley share, and an `ssk_trend_hint` heuristic. `--no-level` skips plane leveling per frame.
- `hazard-map`: `--window` must be an odd integer ≥ 3; `--normalize` accepts `none`, `minmax`, or `p98` (98th-percentile clip).
- `volume-series`: loss convention is `z_ref − z_i` (positive = current frame lower than reference). `--incremental` switches to consecutive-frame loss `z_{i-1} − z_i`. `--signed` adds signed volume vs reference. `--cycles "c1,c2,…"` and `--times "t1,t2,…"` (lengths must match the number of frames) populate `wear_rate_dV_dcycle` and `wear_rate_dV_dt`. **`--align phase-fft`**, **`opencv`**, **`sift`**, or **`sift-normals`** runs stack alignment before volume rows (`sift` / `sift-normals` require OpenCV and follow the TextureFriction `align.ipynb` SIFT pipeline in `tmd.sequence.alignment`); JSON may include an `alignment` block.
- `scratch-evolve`: `--json` returns **`pairs`** (consecutive pairs) and **`series`** (reference row plus per-frame evolution). Optional **`--align`** / **`--reference`** (`-r`) match `volume-series`.
- `slip-axis-series`: JSON **`frames`** list with `frame_index`, `file`, and slip-axis fields; optional **`--align`** / **`--reference`** and **`--use-directionality-mask`**.
- `slip-axis` adds `--use-directionality-mask` to mask the gradient tensor with the defect detector's `directionality_anomalies` mask before computing the preferred axis.

The `tmd-wear` entry point is convenient for shell history filtering and shorter invocations; under the hood it calls `tmd.cli.main:wear_main`, which forwards to the same `create_wear_app()` factory in [`tmd/cli/apps/wear_app.py`](https://github.com/ETSTribology/TrueMapData/blob/main/tmd/cli/apps/wear_app.py).
