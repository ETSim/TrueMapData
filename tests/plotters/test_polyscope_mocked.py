"""Polyscope plotters with a fake ``polyscope`` module (no GUI)."""

from __future__ import annotations

from typing import Any, List

import numpy as np
import pytest

pytestmark = pytest.mark.optional


class _Stub:
    def __getattr__(self, name: str) -> Any:
        return self

    def __call__(self, *a: Any, **k: Any) -> Any:
        return self


class FakePS(_Stub):
    def __init__(self) -> None:
        self.calls: List[Any] = []
        self._t = 0.0

    def is_initialized(self) -> bool:
        return True

    def init(self) -> None:
        self.calls.append("init")

    def set_up_dir(self, *_a: Any, **_k: Any) -> None:
        self.calls.append("set_up_dir")

    def set_ground_plane_mode(self, *_a: Any, **_k: Any) -> None:
        pass

    def set_ground_plane_height_factor(self, *_a: Any, **_k: Any) -> None:
        pass

    def set_screenshot_extension(self, *_a: Any, **_k: Any) -> None:
        pass

    def set_transparency_mode(self, *_a: Any, **_k: Any) -> None:
        pass

    def reset_camera_to_home_view(self) -> None:
        pass

    def set_camera_view_matrix(self, *_a: Any, **_k: Any) -> None:
        pass

    def set_camera_projection_matrix(self, *_a: Any, **_k: Any) -> None:
        pass

    def register_point_cloud(self, name: str, points: np.ndarray) -> _Stub:
        self.calls.append(("pc", name, points.shape))
        return _Stub()

    def register_surface_mesh(self, name: str, vertices: np.ndarray, faces: np.ndarray) -> _Stub:
        self.calls.append(("mesh", name, vertices.shape, faces.shape))
        return _Stub()

    def screenshot(self, path: str, **k: Any) -> None:
        self.calls.append(("screenshot", path, k))

    def show(self) -> None:
        self.calls.append("show")

    def set_user_callback(self, cb: Any) -> None:
        self.calls.append(("cb", cb))

    def set_window_title(self, *_a: Any, **_k: Any) -> None:
        pass

    def get_time(self) -> float:
        self._t += 0.5
        return self._t


class FakePSIM:
    def PushItemWidth(self, *_a: Any, **_k: Any) -> None:
        pass

    def Begin(self, *_a: Any, **_k: Any) -> None:
        pass

    def SliderInt(self, *_a: Any, **_k: Any) -> tuple:
        return (False, 0)

    def SliderFloat(self, *_a: Any, **_k: Any) -> tuple:
        return (False, 10.0)

    def Button(self, *_a: Any, **_k: Any) -> bool:
        return False

    def Text(self, *_a: Any, **_k: Any) -> None:
        pass

    def End(self) -> None:
        pass

    def SameLine(self) -> None:
        pass

    def TreeNode(self, *_a: Any, **_k: Any) -> bool:
        return False

    def TreePop(self) -> None:
        pass


@pytest.fixture
def polyscope_module(monkeypatch: pytest.MonkeyPatch):
    import tmd.plotters.polyscope as pm

    fake = FakePS()
    monkeypatch.setattr(pm, "HAS_POLYSCOPE", True)
    monkeypatch.setattr(pm, "ps", fake)
    monkeypatch.setattr(pm, "psim", FakePSIM())
    return pm, fake


def test_polyscope_plotter_modes(polyscope_module, tmp_path) -> None:
    pm, fake = polyscope_module
    from tmd.plotters.polyscope import PolyscopePlotter

    plotter = PolyscopePlotter()
    hm = np.array([[0.0, 1.0], [0.5, 0.25]], dtype=np.float32)
    plotter.plot(hm, mode="point_cloud", show=False)
    plotter.plot(hm, mode="mesh", show=False)
    plotter.plot(hm, mode="3d", show=False)
    plotter.plot(hm, partial_range=(0, 2, 0, 2), show=False)
    assert any(c[0] == "pc" or c[0] == "mesh" for c in fake.calls)

    png = tmp_path / "cap.png"
    out = plotter.save({"x": 1}, str(png), transparent=True)
    assert out == str(png)


def test_polyscope_sequence_plotter(polyscope_module) -> None:
    pm, fake = polyscope_module
    from tmd.plotters.polyscope import PolyscopeSequencePlotter

    sp = PolyscopeSequencePlotter()
    frames = [
        np.zeros((4, 4), dtype=np.float32),
        np.ones((4, 4), dtype=np.float32),
    ]
    # Library passes explicit plot kwargs and also ``**kwargs``; extra keys must be empty.
    sp.visualize_sequence(frames)
    sp.create_animation(frames, show=False, fps=2)
    for c in fake.calls:
        if isinstance(c, tuple) and c[0] == "cb":
            c[1]()  # invoke imgui callback
            break

    stats = sp.visualize_statistics({"mean_height": [0.1, 0.2], "x": [1.0]}, show=False)
    assert stats is not None
