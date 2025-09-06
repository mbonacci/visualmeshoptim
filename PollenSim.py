from pathlib import Path
import os
from re import S
import threading
import queue
from turtle import pos
os.environ["QT_API"] = "pyqt5"
import pyvista as pv
from pyvistaqt import BackgroundPlotter
import numpy as np

from PollenShell import PollenShell

# 1) Load your shell
shell = PollenShell.from_npz(
    "pollen.npz",
    "cuda",             # "cpu" or "cuda" if you have a GPU
    edge_k_scale=1.0,   # scales stretching stiffness
    bend_k_scale=0.1,   # scales bending stiffness
    rest_bend_from_mesh=False
)

shell.quick_diag(-20.0)

# 2) Optional: set up a live plotter and step-callback for visual feedback
p = BackgroundPlotter()
actor = p.add_mesh(shell.mesh, scalars="Stiffness", cmap="viridis", show_edges=True)

uiQueue = queue.Queue(maxsize=1)

iter_idx_text = [0]
energy_text = [0.0]
volume_text = [shell.volume(shell.verts)]
grad_text = [0.0]


def on_step(iter_idx, energy, verts, volume, grad_norm):
    iter_idx_text[0] = iter_idx
    energy_text[0] = energy
    volume_text[0] = volume
    grad_text[0] = grad_norm
    try:
        if uiQueue.full():
            uiQueue.get_nowait()
        uiQueue.put_nowait(verts.copy())   # copy to avoid races
    except queue.Full:
        pass

def ui_tick():
    try:
        V = uiQueue.get_nowait()
    except queue.Empty:
        return
    # Apply in GUI thread only:
    shell.mesh.points = V
    p.add_text(
        f"iter {iter_idx_text[0]}  E={energy_text[0]:.3e}  V={volume_text[0]:.3e}  |g|={grad_text[0]:.3e}",
        name="hud",
        font_size=10,
    )
    p.render()

p.add_callback(ui_tick, interval=40)  # ~25 FPS poll

# 3) Run relaxation (keep centroid fixed)
def run_relax():
    # res = shell.optimize_pressure(
    #     pressure=2.0,
    #     #pinned_idx=[0],     # keep at least one vertex pinned
    #     fix_centroid=True,
    #     maxiter=50000,
    #     tol=1e-14,
    #     on_step=on_step,
    #     verbose=True
    # )
    res = shell.optimize_volume(
        V_target=1.0,
        beta=25.0,
        #pinned_idx=[0],     # keep at least one vertex pinned
        fix_centroid=True,
        maxiter=50000,
        tol=1e-15,
        callback=on_step,
        verbose=True
    )

# 3) Start relaxation in a separate thread
relax_thread = threading.Thread(target=run_relax, daemon=True)
relax_thread.start()

# 4) Keep the window open
p.app.exec()


