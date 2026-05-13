# Sequential wear analysis

This page covers **stack-level** wear-style metrics on sequences of height maps: translation alignment, positive-loss volume vs a reference, incremental volume between frames, localization of wear, per-frame slip-axis heuristics, and scratch-mask evolution. Implementations live in [`tmd.sequence.wear_analysis`](https://github.com/ETSTribology/TrueMapData/blob/main/tmd/sequence/wear_analysis.py) and on [`TMDSequence`](https://github.com/ETSTribology/TrueMapData/blob/main/tmd/core/sequence.py).

## Alignment first

Volumes, defect-based scratch masks, and slip-axis metrics assume frames share the **same grid size** and are **comparable in-plane** (same region of the surface). For raw captures with small shifts, align **before** wear tables:

- **Python**: `TMDSequence.align_height_maps_phase_fft(reference_index=0)`, or pass `align_before` into `sequential_wear_metrics` / `to_dict` (see below). Values: `phase_fft`, `opencv` (ORB + affine / phase in `tmd.surface.transformations`), **`sift`** (TextureFriction-style SIFT on height), **`sift_normals`** (normals from height → SIFT on normals → warp heights).
- **CLI**: `tmd-wear volume-series`, `scratch-evolve`, and `slip-axis-series` accept `--align` `phase-fft`, `opencv`, **`sift`**, or **`sift-normals`**. Default is `none` for backward compatibility with already-aligned stacks.

**TextureFriction `align.ipynb` parity:** `tmd.sequence.alignment` implements the notebook’s two-pass SIFT/ECC pipeline. On stacks, use **`align_before="sift_normals"`** (or CLI `sift-normals`) for the primary path (normals drive registration), or **`sift`** for registration on normalized height rasters. **`tmd-process sequence align --method sift --register-from normals|height`** writes aligned `.tmd` files the same way.

OpenCV alignment (`opencv`) and SIFT paths can change crop and resolution; phase FFT keeps shape and uses integer rolls only.

### Pre-align and save `.tmd` files

When you want **persistent** aligned captures on disk (for inspection, roughness tracks, or repeated wear runs), align once with the sequence CLI, then point `tmd-wear` at the `*_aligned.tmd` outputs with **`--align none`**:

```bash
tmd-process sequence align a.tmd b.tmd c.tmd -o ./aligned
# TextureFriction-style SIFT (OpenCV): normals drive registration, then heights are warped
tmd-process sequence align a.tmd b.tmd c.tmd -o ./aligned --method sift --register-from normals
tmd-wear volume-series ./aligned/*_aligned.tmd --align none --reference 0 --json
```

Default `sequence align` uses **ORB / affine** (`--method auto`, …). **`--method sift`** selects the notebook-style pipeline; pair with **`--register-from normals`** (default alternative: `height`). Command tables: [CLI reference — `visualize` / `sequence`](../reference/cli.md#other-useful-visualize-sequence-commands).

## Loss convention and volume

For reference height `z_ref` and frame `z_i`, the library uses **loss = z_ref − z_i**. **Positive** loss at a pixel means the current frame is **lower** than the reference (material removal relative to reference when heights increase toward the sensor in your convention).

- **Vs reference**: `wear_series_vs_reference` integrates the positive part of that loss (optionally with signed volume when requested).
- **Incremental**: `wear_incremental_series` uses consecutive pairs `z_{i-1} − z_i` and accumulates positive incremental volume.

**Localization index**: fraction of total positive loss contributed by the top `--top-fraction` of positive-loss pixels (default 0.10), measuring how concentrated wear is.

## Python API

```python
from tmd.core.sequence import TMDSequence
import numpy as np

seq = TMDSequence("demo")
# ... add_frame(...) or add_tmd_file(...) for each capture ...

wear = seq.sequential_wear_metrics(
    dx_mm=0.01,
    dy_mm=0.01,
    reference_index=0,
    align_before="phase_fft",  # optional: "phase_fft" | "opencv" | "sift" | "sift_normals"
    align_sift_kwargs={"two_pass": True, "crop": True},  # optional, for sift / sift_normals
    include_slip_axis_series=True,
    include_scratch_series=True,
)
# wear may include: alignment, vs_reference, incremental, slip_axis_series, scratch_series
```

Volume keys require **both** `dx_mm` and `dy_mm` (pixel pitch in mm). Slip-axis and scratch series do not need pitch. You can request slip or scratch alone with `dx_mm=None`, `dy_mm=None`.

For export-style dicts with statistics:

```python
d = seq.to_dict(
    include_derived=True,
    wear_dx_mm=0.01,
    wear_dy_mm=0.01,
    wear_align_before="phase_fft",
    wear_include_slip_axis_series=True,
    wear_include_scratch_series=True,
)
# d["derived"]["wear"] holds the same structure as sequential_wear_metrics
```

`scratch_series` uses [`detect_surface_defects`](https://github.com/ETSTribology/TrueMapData/blob/main/tmd/surface/defects.py) with standard scratch masks; override with `scratch_defect_config` if needed.

## CLI (`tmd-wear` / `tmd-process wear`)

| Command | Role |
|---------|------|
| `volume-series` | CSV/JSON volume + localization; `--align` `phase-fft` \| `opencv` \| `sift` \| `sift-normals` |
| `slip-axis-series` | JSON `frames` array with per-file slip-axis metrics; same `--align` / `-r` |
| `scratch-evolve` | Pairwise evolution plus full `series` from `scratch_series_metrics`; `--align` / `-r` |
| `slip-axis` | Single-file slip-axis metrics |
| `roughness-track` | Surfalize trajectory + trajectory derivatives (requires Surfalize) |

Examples:

```bash
tmd-wear volume-series a.tmd b.tmd c.tmd --align phase-fft --reference 0 --json
tmd-wear volume-series a.tmd b.tmd c.tmd --align sift-normals --reference 0 --json
tmd-wear slip-axis-series --from-dir ./captures --pattern "*.tmd" --align phase-fft --json
tmd-wear scratch-evolve f0.tmd f1.tmd --json
```

JSON from `scratch-evolve` includes **`pairs`** (consecutive file pairs) and **`series`** (one row per frame: reference row then per-frame evolution metrics).

## Related: height-matrix wear simulation

The notebook [`notebooks/surface_wear_height_matrix.ipynb`](https://github.com/ETSTribology/TrueMapData/blob/main/notebooks/surface_wear_height_matrix.ipynb) is a **separate** workflow: two height matrices in contact, friction, and cumulative wear simulation (CPU reference for texture-space wear). It is not the same as `wear_series_vs_reference`, but it complements sequential analysis when you model wear from first principles.

See also: [Tribology metrics](tribology-metrics.md), [Working with TMD files](working-with-tmd-files.md), and the CLI reference [Wear / `tmd-wear`](../reference/cli.md#wear-subcommands-also-tmd-wear).
