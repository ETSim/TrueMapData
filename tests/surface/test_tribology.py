"""Tests for tribology metrics and optional plane removal."""

from __future__ import annotations

import numpy as np
import pytest

from tmd.surface import metrics as tr
from tmd.surface.metrics import (
    bearing_area_curve,
    cell_sizes_from_metadata,
    debris_pocket_map,
    interfacial_shear_proxy_map,
    preferred_slip_axis,
    summit_curvature_map,
)


def test_bearing_area_curve_endpoints() -> None:
    rng = np.random.default_rng(0)
    z = rng.normal(size=(32, 32)).astype(np.float64)
    curve = bearing_area_curve(z, n=20, z_reference="mean")
    a = curve["area_fraction"]
    assert a[0] >= a[-1] - 1e-6
    assert curve.get("plane_removal") == "none"


def test_bearing_area_curve_median_reference() -> None:
    z = np.arange(12, dtype=np.float64).reshape(3, 4)
    c = bearing_area_curve(z, n=5, z_reference="median")
    assert c["z_reference"] == "median"


def test_bearing_area_curve_explicit_separations() -> None:
    z = np.zeros((4, 4), dtype=np.float64)
    z[1, 1] = 1.0
    s = np.array([-0.5, 0.0, 0.5, 1.5])
    c = bearing_area_curve(z, separations=s, z_reference="mean")
    np.testing.assert_array_equal(c["separations"], s)


def test_preferred_slip_axis_flat() -> None:
    z = np.zeros((24, 24), dtype=np.float32)
    out = preferred_slip_axis(z, {"x_length": 1.0, "y_length": 1.0}, include_anomaly_angle=False)
    assert "axis_rad" in out
    assert out["asymmetry"] >= 0.0


def test_preferred_slip_plane_removal_mean() -> None:
    z = np.ones((16, 16), dtype=np.float32) * 3.0
    out = preferred_slip_axis(z, {"x_length": 1.0, "y_length": 1.0}, include_anomaly_angle=False, plane_removal="mean")
    assert out["axis_rad"] == pytest.approx(0.0, abs=1e-5)


def test_shear_proxy_positive() -> None:
    x = np.linspace(-1, 1, 48)
    xx, yy = np.meshgrid(x, x)
    z = (xx**2 + yy**2).astype(np.float32)
    m = interfacial_shear_proxy_map(z, {"x_length": 1.0, "y_length": 1.0}, roughness_sigma=3.0)
    assert m.shape == z.shape
    assert float(np.nanmax(m)) >= 0.0


def test_summit_curvature_non_negative_inv_radius() -> None:
    rng = np.random.default_rng(1)
    z = rng.uniform(-0.1, 0.1, size=(40, 40)).astype(np.float32)
    out = summit_curvature_map(z, {"x_length": 1.0, "y_length": 1.0})
    assert out["inv_radius"].shape == z.shape
    assert float(np.max(out["inv_radius"])) >= 0.0


def test_summit_high_min_curvature_empty_summits() -> None:
    z = np.zeros((20, 20), dtype=np.float32)
    out = summit_curvature_map(z, {"x_length": 1.0, "y_length": 1.0}, min_mean_curvature=1e9)
    assert not np.any(out["summit_mask"])


def test_summit_curvature_peak_has_positive_inv_radius() -> None:
    """Graph peaks have negative H in this convention; summits still get |H| painted."""
    n = 48
    x = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    xx, yy = np.meshgrid(x, x)
    z = np.exp(-4.0 * (xx * xx + yy * yy)).astype(np.float32)
    out = summit_curvature_map(z, {"x_length": 1.0, "y_length": 1.0}, smooth_sigma=0.5, min_mean_curvature=0.0)
    assert float(np.max(out["inv_radius"])) > 1e-6
    assert int(out["summit_mask"].sum()) >= 1


def test_debris_pocket_mask_shape() -> None:
    z = np.zeros((30, 30), dtype=np.float32)
    z[10:12, 10:12] = -0.5
    out = debris_pocket_map(z, {"x_length": 1.0, "y_length": 1.0})
    assert out["pocket_mask"].shape == z.shape


    dx, dy = cell_sizes_from_metadata({"x_length": 2.0, "y_length": 1.0}, (10, 20))
    assert abs(dx - 0.1) < 1e-6
    assert abs(dy - 0.1) < 1e-6


def test_cell_sizes_mmpp() -> None:
    dx, dy = cell_sizes_from_metadata({"mmpp": 0.05}, (10, 10))
    assert abs(dx - 0.05) < 1e-9
    assert abs(dy - 0.05) < 1e-9


def test_cell_sizes_defaults() -> None:
    dx, dy = cell_sizes_from_metadata(None, (5, 5))
    assert dx == 1.0 and dy == 1.0


def test_non_2d_raises() -> None:
    with pytest.raises(ValueError, match="2D"):
        preferred_slip_axis(np.zeros(5), None)
    with pytest.raises(ValueError, match="2D"):
        interfacial_shear_proxy_map(np.zeros((2, 2, 2)), None)
    with pytest.raises(ValueError, match="2D"):
        bearing_area_curve(np.zeros(3), n=2)
    with pytest.raises(ValueError, match="2D"):
        summit_curvature_map(np.zeros(4), None)
    with pytest.raises(ValueError, match="2D"):
        debris_pocket_map(np.zeros((2, 3, 1)), None)


def test_circular_mean_angle_weighted_uniform_weights() -> None:
    # Avoid 4-fold symmetry in doubled-angle space (e.g. 0, ±π/2, π sums to c=s=0 → None).
    a = np.array([0.1, 0.2, 0.35, 0.15], dtype=np.float64)
    w = np.ones(4)
    ang = tr._circular_mean_angle_weighted(a, w)
    assert ang is not None
    assert -np.pi <= ang <= np.pi
    assert abs(ang - float(np.mean(a))) < 0.2


def test_circular_mean_angle_weighted_symmetric_returns_none() -> None:
    """Four directions at 90° cancel in doubled-angle (2θ) space."""
    a = np.array([0.0, np.pi / 2, np.pi, -np.pi / 2], dtype=np.float64)
    assert tr._circular_mean_angle_weighted(a, np.ones(4)) is None


def test_circular_mean_zero_weight_returns_none() -> None:
    assert tr._circular_mean_angle_weighted(np.zeros(3), np.zeros(3)) is None


def test_asymmetry_three_aligned() -> None:
    t = 0.3
    assert tr._asymmetry_from_three_angles([t, t, t]) < 0.05


def test_asymmetry_all_none() -> None:
    assert tr._asymmetry_from_three_angles([None, None, None]) == 1.0


def test_preprocess_plane_invalid() -> None:
    with pytest.raises(ValueError, match="plane_removal"):
        tr._preprocess_plane(np.zeros((3, 3), np.float32), {}, "not_a_mode")


def test_preprocess_plane_none_is_copy() -> None:
    z = np.arange(9, dtype=np.float32).reshape(3, 3)
    out = tr._preprocess_plane(z, {}, "none")
    assert out.shape == z.shape
    out[0, 0] = 999.0
    assert z[0, 0] != 999.0


def test_surfalize_fallback_warns_when_level_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    from tmd.surface.metrics import surfalize as ts

    monkeypatch.setattr(ts, "level_height_map_surfalize", lambda h, m: None)
    z = np.ones((6, 6), dtype=np.float32) * 2.0
    with pytest.warns(UserWarning, match="Surfalize"):
        out = tr._preprocess_plane(z, {"x_length": 1.0, "y_length": 1.0}, "surfalize")
    np.testing.assert_allclose(out, np.zeros_like(z), atol=1e-5)


def test_tribology_surfalize_module_available_flag() -> None:
    from tmd.surface.metrics import surfalize as ts

    assert isinstance(ts.surfalize_available(), bool)
