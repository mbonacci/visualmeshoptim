"""Polyscope integration for interactive optimization."""
from __future__ import annotations

import time
from typing import Any, Dict, Mapping, Optional

import numpy as np
from .utils import TunableParameter


def _import_backend(backend: Optional[tuple[Any, Any]]) -> tuple[Any, Any]:
    if backend is not None:
        return backend
    try:  # pragma: no cover - exercised in runtime env with Polyscope
        import polyscope as ps
        import polyscope.imgui as psim
    except Exception as exc:  # pragma: no cover - exercised when Polyscope missing
        raise RuntimeError("Polyscope is required for interactive visualization") from exc
    return ps, psim


class MeshVisualizer:
    """Encapsulate Polyscope viewer management."""

    def __init__(
        self,
        *,
        title: str = "Visual Mesh Optimizer",
        mesh_name: str = "Mesh",
        backend: Optional[tuple[Any, Any]] = None,
    ) -> None:
        self.title = title
        self.mesh_name = mesh_name
        self._ps, self._psim = _import_backend(backend)
        self._mesh = None
        self._last_update = 0.0

    # ------------------------------------------------------------------
    def run(
        self,
        *,
        initial_vertices: np.ndarray,
        faces: np.ndarray,
        quantities: Mapping[str, np.ndarray],
        vertex_queue,
        stats: Dict[str, float],
        problem_parameters: Mapping[str, TunableParameter],
        optimizer_parameters: Mapping[str, TunableParameter],
        on_start,
        on_stop,
        is_running,
        refresh_quantities,
        on_reset=None,
    ) -> None:
        ps, psim = self._ps, self._psim
        ps.set_max_fps(60)
        ps.set_enable_vsync(False)
        ps.init()
        mesh = ps.register_surface_mesh(self.mesh_name, initial_vertices, faces, smooth_shade=True)
        mesh.set_edge_width(0.3)
        mesh.set_edge_color((0.2, 0.2, 0.2))
        for name, values in quantities.items():
            mesh.add_scalar_quantity(name, values, defined_on="vertices", enabled=True)
        self._mesh = mesh

        def ui_callback() -> None:
            self._update_mesh(vertex_queue)
            self._render_stats(psim, stats)
            self._render_parameters(psim, "Problem Parameters", problem_parameters, refresh_quantities)
            self._render_optimizer_panel(psim, optimizer_parameters, on_start, on_stop, is_running, on_reset)

        ps.set_user_callback(ui_callback)

        try:
            while not ps.window_requests_close():
                time.sleep(0.01)
                ps.frame_tick()
        finally:
            on_stop()
            ps.shutdown()

    # ------------------------------------------------------------------
    def _update_mesh(self, vertex_queue) -> None:
        if self._mesh is None:
            return
        now = time.perf_counter()
        if (now - self._last_update) < (1.0 / 30.0):
            return
        verts = None
        try:
            verts = vertex_queue.get_nowait()
        except Exception:
            verts = None
        if verts is not None:
            try:
                self._mesh.update_vertex_positions(verts)
            except Exception:
                pass
        self._last_update = now

    def _update_quantities(self, values: Mapping[str, np.ndarray]) -> None:
        if self._mesh is None:
            return
        for name, array in values.items():
            try:
                self._mesh.add_scalar_quantity(name, array, defined_on="vertices", enabled=True)
            except Exception:
                pass

    def _render_stats(self, psim, stats: Mapping[str, float]) -> None:
        try:
            shown, _ = psim.Begin("Statistics", True)
        except Exception:
            return
        if not shown:
            psim.End()
            return
        psim.TextUnformatted(
            f"iter {int(stats.get('iteration', 0))} | obj={stats.get('objective', 0.0):.3e} | grad={stats.get('grad_norm', 0.0):.3e}"
        )
        psim.End()

    def _render_parameters(self, psim, title: str, params: Mapping[str, TunableParameter], refresh_callback=None) -> None:
        if not params:
            return
        try:
            shown, _ = psim.Begin(title, True)
        except Exception:
            return
        if not shown:
            psim.End()
            return
        for name, param in params.items():
            self._render_parameter(psim, name, param)
        if refresh_callback is not None:
            if psim.Button("Refresh Visuals"):
                updated = refresh_callback()
                if isinstance(updated, Mapping):
                    self._update_quantities(updated)
        psim.End()

    def _render_parameter(self, psim, name: str, param: TunableParameter) -> None:
        dtype = param.dtype.lower()
        try:
            if dtype == "bool":
                changed, value = psim.Checkbox(name, bool(param.value))
                if changed:
                    param.update(value)
            elif dtype == "int":
                changed, value = psim.InputInt(name, int(param.value))
                if changed:
                    param.update(value)
            elif dtype == "float":
                changed, value = psim.InputFloat(name, float(param.value))
                if changed:
                    param.update(value)
            elif dtype in {"choice", "enum"}:
                psim.TextUnformatted(name + ":")
                for option in param.options or []:
                    label = f"{option} " + ("[x]" if option == param.value else "[ ]")
                    if psim.Button(label):
                        param.update(option)
                    psim.SameLine()
                psim.NewLine()
            else:
                psim.TextUnformatted(f"{name}: {param.value}")
        except Exception:
            pass

    def _render_optimizer_panel(
        self,
        psim,
        optimizer_parameters: Mapping[str, TunableParameter],
        on_start,
        on_stop,
        is_running,
        on_reset,
    ) -> None:
        try:
            shown, _ = psim.Begin("Optimization", True)
        except Exception:
            return
        if not shown:
            psim.End()
            return

        for name, param in optimizer_parameters.items():
            if name in {"optimizer", "learning_rate", "max_iterations", "tolerance", "fix_centroid", "lbfgs_history"}:
                self._render_parameter(psim, name, param)
        psim.Separator()
        if not is_running():
            if psim.Button("Start"):
                on_start()
            psim.SameLine()
            if psim.Button("Reset") and on_reset is not None:
                on_reset()
        else:
            psim.TextUnformatted("Status: Running…")
            if psim.Button("Stop"):
                on_stop()
        psim.End()
