# Tribology metrics from height maps

This library adds **NumPy-only** helpers in `tmd.surface.metrics` (MIT-licensed core) plus a **`tmd-process tribology`** CLI group. Metrics are **proxies** for research and visualization unless you validate them for your materials and instruments.

## Sign convention (bearing / contact curve)

Heights are leveled in two steps: optional **large-scale plane removal** (`plane_removal` on the tribology helpers), then the bearing reference (`z_reference`: **mean** or **median**). Let `z'` be the height after both steps, in the same units as the TMD raster (for example millimetres if your capture is in mm).

For a grid of **separation** values `d`, the exported **area fraction** is

`mean(z' >= d)` — the fraction of pixels whose leveled height lies **on or above** the cutting plane at height `d`.

When `d` runs from the minimum to the maximum of `z'`, the curve starts near **1** (almost all material above the lowest plane) and ends near **0** (almost none above the highest plane). The discrete derivative `dA/dd` is a geometric **stiffness proxy**, not an elastic Hertz solution.

## Python API

```python
from tmd import TMD
from tmd.surface.metrics import (
    preferred_slip_axis,
    interfacial_shear_proxy_map,
    bearing_area_curve,
    summit_curvature_map,
    debris_pocket_map,
    slide_azimuth_rad_from_direction,
    texture_modulated_pair_friction_maps,
)

data = TMD.load("path/to/file.tmd")
z = data.height_map
meta = data.metadata or {}

axis = preferred_slip_axis(z, meta, plane_removal="none")
curve = bearing_area_curve(z, n=64, z_reference="mean", metadata=meta, plane_removal="mean")
hazard = interfacial_shear_proxy_map(z, meta, plane_removal="none")
summits = summit_curvature_map(z, meta, plane_removal="none")
pockets = debris_pocket_map(z, meta, plane_removal="none")
```

### Texture-modulated pair friction maps (two surfaces)

For a **master–slave** height pair (same grid as the height-matrix wear notebook), you can build four direction-dependent maps that start from `tmd.sequence.wear_simulation.combined_mu` and apply a **phenomenological** spatial modulation from **local** texture direction on both surfaces:

- **`slide_azimuth_rad_from_direction(direction)`** — image-plane angle (radians) for the lab slide codes `1..4` (`+x`, `-x`, `+y`, `-y`), consistent with `sliding_vector` in `wear_simulation`.
- **`texture_modulated_pair_friction_maps(ms, ss, wear_params, …)`** — for each direction, multiplies baseline `combined_mu` by a weight in `[0, 1]` derived from doubled-angle alignment between the slide azimuth and locally averaged gradient directions on MS and SS (`local_texture_angle_and_coherence` with an odd `window`, default `9`). Weights from the two bodies are fused with `fusion="rms"` (geometric mean of alignments) or `"mean"`.

Parameters:

| Parameter | Role |
|-----------|------|
| `strength` in `[0, 1]` | `0` recovers baseline `combined_mu` exactly; larger values increase texture-driven contrast between directions. |
| `return_aux=True` | Also returns per-direction **modulation-only** maps plus smoothed angle/coherence fields for debugging plots. |

**Scope:** these maps are for **analysis and visualization** (e.g. comparing anisotropic effective μ before/after wear). They are **not** a substitute for measured friction and are **not** applied inside `run_simulation` unless you wire them in yourself at the contact step.

Example (aligned grids from your own loader or from `select_pair` in a notebook):

```python
from tmd.sequence.wear_simulation import WearParameters

wear_p = WearParameters(friction_clip=0.3)
mu_by_dir = texture_modulated_pair_friction_maps(
    ms_height, ss_height, wear_p, metadata=meta, strength=0.35, fusion="rms", window=9
)
# Keys 1..4 match DIRECTION_NAMES in wear_simulation (+x, -x, +y, -y).
slide_phi_rad = slide_azimuth_rad_from_direction(1)
```

### Large-scale plane removal (`plane_removal`)

All tribology helpers that take a height map accept `plane_removal`:

| Value | Behaviour |
|-------|-----------|
| `none` | Use heights as-is (default; matches earlier releases). |
| `mean` | Subtract global mean after any copy. |
| `median` | Subtract global median. |
| `surfalize` | Use **Surfalize** `Surface.level()` when installed (pass **`metadata`** so x/y pixel size matches your capture). If Surfalize is missing or leveling fails, a **warning** is emitted and **mean** removal is used instead. |

Map generators (`shear_hazard`, `debris_pocket`, `summit_curvature`) forward the same keyword from their options dict.

- **`preferred_slip_axis`** combines gradient-based global lay, PSD dominant direction from `calculate_surface_isotropy`, and optional directionality-anomaly weighting (reuses `detect_surface_defects`).
- **`interfacial_shear_proxy_map`** is `|∇z| × |H| × Sq_local` with local RMS roughness from a high-pass residual (not a calibrated μ).
- **`summit_curvature_map`** labels local maxima on a smoothed surface and reports `|mean curvature|` on summits. ``min_mean_curvature`` applies to **|H|** (peaks have negative `H` in the same sign convention as the mean-curvature map generator).
- **`debris_pocket_map`** is a heuristic `(pits ∪ deep valleys) ∩ low slope` mask for third-body pocket discussion.

## CLI

| Subcommand | Purpose |
|------------|---------|
| `tmd-process tribology axis FILE.tmd [--plane-removal {none,mean,median,surfalize}] [--json]` | Preferred texture / slip axis and asymmetry |
| `tmd-process tribology contact-curve FILE.tmd [--n 50] [--plane-removal {none,mean,median,surfalize}] [--json \| -o out.csv]` | Bearing-style area vs separation (metadata from TMD for Surfalize steps) |
| `tmd-process tribology plot FILE.tmd -o OUT.png [--plane-removal mean] [--no-maps] [--dpi 150]` | One PNG: leveled height, bearing curve + \|dA/dd\|, slip-axis arrow, optional proxy maps |
| `tmd-process tribology lubrication FILE.tmd` | ISO functional volumes (`Vvv`, `Vvc`, …) via **Surfalize** (GPL-3.0); use `--no-json` for a table |

Programmatic export uses **`tmd.surface.metrics.save_tribology_dashboard_png`** (matplotlib **Agg** backend; no display required).

## Map export types

Registered with `tmd-process maps` (same pipeline as other generators):

| Map type | Meaning |
|----------|---------|
| `shear_hazard` | Normalized interfacial shear proxy |
| `debris_pocket` | Normalized debris-pocket score |
| `summit_curvature` | Normalized summit `|H|` |

Example:

```bash
tmd-process maps batch ./data -o ./out --types shear_hazard debris_pocket summit_curvature
```

## Surfalize / GPL

`tribology lubrication` and ISO volume parameters are computed with **Surfalize** when installed (`pip install "truemapdata[roughness]"`). Optional **`plane_removal="surfalize"`** on the MIT tribology helpers uses `tmd.surface.metrics.surfalize.level_height_map_surfalize` for least-squares plane removal that respects **metadata** pixel spacing (same µm/mm convention as roughness tooling). Surfalize is **GPL-3.0**; check license obligations before redistributing or combining with proprietary stacks.

## See also

For sequence-oriented wear analysis (Abbott / bearing curve, wear volume series, hazard maps, scratch evolution, slip axis, and Surfalize-backed roughness trajectories), see the **[`wear` subcommands](../reference/cli.md#wear-subcommands-also-tmd-wear)** section in the CLI reference. The same Typer app is available as the standalone **`tmd-wear`** console script and shares `--plane-removal` conventions with the `tribology` group documented above.
