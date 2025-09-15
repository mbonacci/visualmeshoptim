from math import e
from pathlib import Path
import threading
import queue

import numpy as np
import time
import torch
from torch.utils import cmake_prefix_path

from PollenShell import PollenShell

# Visualization via Polyscope (pip install polyscope)
import polyscope as ps
import polyscope.imgui as psim


def main():
    # 1) Load shell
    device = "cuda" if torch.cuda.is_available() else "cpu"
    shell = PollenShell.from_npz(
        "pollen.npz",
        device,
        edge_k_scale=1.0,
        bend_k_scale=0.1,
        rest_bend_from_mesh=False,
    )

    shell.quick_diag(-20.0)

    # 2) Init Polyscope and register mesh
    V0 = shell.verts.detach().cpu().numpy()
    F = shell.faces.detach().cpu().numpy()
    v_stiff = shell.vertex_stiffness.detach().cpu().numpy()

    ps.set_max_fps(60)           # set FPS cap
    ps.set_enable_vsync(False)   # don't block on vblank
    ps.set_SSAA_factor(3)

    ps.init()

    m = ps.register_surface_mesh("Pollen Shell", V0, F, smooth_shade=True)

    # Add per-vertex scalar quantity for stiffness and enable it
    q = m.add_scalar_quantity("Stiffness", v_stiff, defined_on="vertices", enabled=True, cmap="viridis")

    # Optional: light edge overlay for readability
    m.set_edge_width(0.3)
    m.set_edge_color((0.2, 0.2, 0.2))

    # Shared state for UI
    ui_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=1)
    stats = {
        "iter": 0,
        "E": 0.0,
        "V": float(shell.volume(shell.verts)),
        "g": 0.0,
    }
    run_state = {"started": False, "thread": None}

    # on_step runs in the solver thread; store latest verts and stats
    def on_step(iter_idx: int, energy: float, verts: np.ndarray, volume: float, grad_norm: float):
        stats["iter"] = iter_idx
        stats["E"] = energy
        stats["V"] = volume
        stats["g"] = grad_norm
        try:
            if ui_queue.full():
                ui_queue.get_nowait()
            # Cast to float32 to lighten payload
            ui_queue.put_nowait(verts.astype(np.float32, copy=False))
        except queue.Full:
            pass
        # Do not call request_redraw excessively; viewer runs at its own FPS


    def run_relax_thread():
        def run_relax():
            shell.optimize_pressure(
                pressure=-2.0,
                fix_centroid=False,
                maxiter=5000,
                tol=1e-14,
                on_step=on_step,
                verbose=True
            )
        t = threading.Thread(target=run_relax, daemon=True)

        t.start()
        run_state["thread"] = t
        run_state["started"] = True

    # Per-frame UI callback in the viewer thread
    def ui_callback():
        # throttle mesh updates to ~30 FPS, but always draw UI
        if not hasattr(ui_callback, "_last"):  # type: ignore[attr-defined]
            ui_callback._last = 0.0  # type: ignore[attr-defined]
        now = time.perf_counter()
        should_update = (now - ui_callback._last) >= (1.0 / 30.0)  # type: ignore[attr-defined]
        if should_update:
            # Apply the most recent vertices if any
            try:
                V = ui_queue.get_nowait()
            except queue.Empty:
                V = None
            if V is not None:
                try:
                    m.update_vertex_positions(V)
                except Exception:
                    pass
            ui_callback._last = now  # type: ignore[attr-defined]

        # HUD panel with iteration stats
        try:
            open_state = True
            shown, open_state = psim.Begin("PollenSim", open_state)
            if shown:
                psim.TextUnformatted(
                    f"iter {stats['iter']}  E={stats['E']:.3e}  V={stats['V']:.3e}  |g|={stats['g']:.3e}"
                )
                # Button to start optimization (spawns solver thread)
                if not run_state["started"]:
                    if psim.Button("Start Optimization"):
                        run_relax_thread()
                else:
                    psim.TextUnformatted("Status: Running...")
            psim.End()
        except Exception:
            # If imgui API differs, skip HUD
            pass

    ps.set_user_callback(ui_callback)

    # 4) Enter the viewer loop (blocks until closed)
    while not ps.window_requests_close():
        # sleep to yield to cuda thread (this is crucial for optimization performance!)
        time.sleep(0.01)
        ps.frame_tick() 

if __name__ == "__main__":
    main()
