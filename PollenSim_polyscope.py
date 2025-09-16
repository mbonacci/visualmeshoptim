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
    run_state = {"started": False, "thread": None, "stop_event": threading.Event()}

    # UI state for optimization parameters
    ui = {
        "objective": "Pressure",    # "Pressure" or "Volume"
        "optimizer": "LBFGS",       # "LBFGS" or "Adam"
        "maxiter": 2000,
        "tol": 1e-10,
        "pressure": -2.0,
        "V_target": float(shell.volume(shell.verts)),
        "beta": 1.0,
        "fix_centroid": False,
    }

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


    def run_opt_thread():
        # construct objective function based on UI selection
        objective = ui["objective"].lower()
        optimizer = ui["optimizer"]
        maxiter = int(max(1, ui["maxiter"]))
        tol = float(max(1e-20, ui["tol"]))
        pressure = float(ui["pressure"])  # used if objective == pressure
        V_target = float(ui["V_target"])  # used if objective == volume
        beta = float(max(0.0, ui["beta"]))
        fix_centroid = bool(ui["fix_centroid"])

        run_state["stop_event"].clear()

        def obj_fn(V):
            if objective == "pressure":
                return shell.total_energy(V, pressure)
            else:
                return shell.strech_energy(V) + shell.bend_energy(V) + 0.5 * beta * (shell.volume(V) - V_target) ** 2

        def should_stop():
            return run_state["stop_event"].is_set()

        def run():
            try:
                shell._optimize(
                    obj=obj_fn,
                    pinned_idx=None,
                    fix_centroid=fix_centroid,
                    maxiter=maxiter,
                    tol=tol,
                    verbose=True,
                    callback=on_step,
                    optimizer=optimizer,
                    should_stop=should_stop,
                )
            finally:
                # mark as stopped regardless of outcome
                run_state["started"] = False

        t = threading.Thread(target=run, daemon=True)
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

        # Update run state if thread finished
        if run_state["thread"] is not None and not run_state["thread"].is_alive():
            run_state["started"] = False
            run_state["thread"] = None

        # HUD panel with iteration stats
        try:
            open_state = True
            shown, open_state = psim.Begin("PollenSim", open_state)
            if shown:
                psim.TextUnformatted(
                    f"iter {stats['iter']}  E={stats['E']:.3e}  V={stats['V']:.3e}  |g|={stats['g']:.3e}"
                )
            psim.End()
        except Exception:
            # If imgui API differs, skip HUD
            pass

        # Optimization control panel
        try:
            open_state2 = True
            shown2, open_state2 = psim.Begin("Optimization", open_state2)
            if shown2:
                # Objective selection
                psim.TextUnformatted("Objective:")
                if psim.Button(("Pressure " + ("[x]" if ui["objective"] == "Pressure" else "[ ]"))):
                    ui["objective"] = "Pressure"
                psim.SameLine()
                if psim.Button(("Volume " + ("[x]" if ui["objective"] == "Volume" else "[ ]"))):
                    ui["objective"] = "Volume"

                # Optimizer selection
                psim.Separator()
                psim.TextUnformatted("Optimizer:")
                if psim.Button(("LBFGS " + ("[x]" if ui["optimizer"] == "LBFGS" else "[ ]"))):
                    ui["optimizer"] = "LBFGS"
                psim.SameLine()
                if psim.Button(("Adam " + ("[x]" if ui["optimizer"] == "Adam" else "[ ]"))):
                    ui["optimizer"] = "Adam"

                # Numeric inputs
                psim.Separator()
                try:
                    changed, val = psim.InputInt("Max Iters", int(ui["maxiter"]))
                    if changed:
                        ui["maxiter"] = max(1, int(val))
                except Exception:
                    pass
                try:
                    changed, val = psim.InputFloat("Tolerance", float(ui["tol"]))
                    if changed:
                        ui["tol"] = float(max(1e-20, val))
                except Exception:
                    pass
                try:
                    changed, val = psim.Checkbox("Fix centroid", bool(ui["fix_centroid"]))
                    if changed:
                        ui["fix_centroid"] = bool(val)
                except Exception:
                    pass

                # Objective-specific params
                if ui["objective"] == "Pressure":
                    try:
                        changed, val = psim.InputFloat("Pressure", float(ui["pressure"]))
                        if changed:
                            ui["pressure"] = float(val)
                    except Exception:
                        pass
                else:
                    try:
                        changed, val = psim.InputFloat("Target Volume", float(ui["V_target"]))
                        if changed:
                            ui["V_target"] = float(val)
                    except Exception:
                        pass
                    try:
                        changed, val = psim.InputFloat("Beta", float(ui["beta"]))
                        if changed:
                            ui["beta"] = float(max(0.0, val))
                    except Exception:
                        pass

                psim.Separator()

                # Control buttons
                if not run_state["started"]:
                    if psim.Button("Start Optimization"):
                        run_opt_thread()
                    psim.SameLine()
                    if psim.Button("Reset Vertices"):
                        # Reset shell and mesh to original positions
                        shell.update_vertices(V0)
                        try:
                            m.update_vertex_positions(V0)
                        except Exception:
                            pass
                        # Clear queue and reset stats
                        try:
                            while not ui_queue.empty():
                                ui_queue.get_nowait()
                        except Exception:
                            pass
                        stats["iter"] = 0
                        stats["E"] = 0.0
                        stats["V"] = float(shell.volume(shell.verts))
                        stats["g"] = 0.0
                else:
                    psim.TextUnformatted("Status: Running…")
                    if psim.Button("Stop Optimization"):
                        run_state["stop_event"].set()

            psim.End()
        except Exception:
            pass

    ps.set_user_callback(ui_callback)

    # 4) Enter the viewer loop (blocks until closed)
    while not ps.window_requests_close():
        # sleep to yield to cuda thread (this is crucial for optimization performance!)
        time.sleep(0.01)
        ps.frame_tick() 

if __name__ == "__main__":
    main()
