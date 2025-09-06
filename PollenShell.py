from re import M, S
import re
from tkinter import SE
import numpy as np
from sympy import E
import torch
from typing import Iterable, Optional, Callable
import pyvista as pv
import MeshUtils as mu

class PollenShell:
    """
    PollenShell
    -----------
    - Loads a triangulated shell (.npz) with:
        vertices: (N,3)
        faces:    (T,3) int
        edges:    (M,2) int  (optional; will be built if missing)
        v_*:      exactly one vertex layer -> elasticity factor per vertex (scalar or RGB)
    - Builds:
        edges (M,2), L0 (rest lengths), k_edge (edge stiffness),
        bend_edges (B,4), theta0 (rest dihedrals), k_bend (bending stiffness)
    - Maintains a PyVista mesh for visualization.
    - Provides vectorized energies: stretch, bend, pressure, total.
    """

    def __init__(self, device, edge_k_scale: float = 1.0, bend_k_scale: float = 0.1,
                 rest_bend_from_mesh: bool = True):
        self.edge_k_scale = float(edge_k_scale)
        self.bend_k_scale = float(bend_k_scale)
        self.rest_bend_from_mesh = bool(rest_bend_from_mesh)
        self.device = device

        # Geometry / topology
        self.verts = None          # (N,3)
        self.faces = None          # (T,3) int
        self.edges = None          # (M,2) int
        self.bend_edges = None     # (B,4) int (i,j,k,l)

        # Rest-state / parameters
        self.L0 = None             # (M,)   rest edge lengths
        self.theta0 = None         # (B,)   rest dihedrals (if any)
        self.k_edge = None         # (M,)   stretching stiffness per edge
        self.k_bend = None         # (B,)   bending stiffness per bend-edge
        self.vertex_stiffness = None  # (N,) scalar stiffness factor from v_* layer

        # PyVista mesh
        self.mesh = None           # pv.PolyData

    # ----------------------
    # Loading & preparation
    # ----------------------

    @classmethod
    def from_npz(cls, path, device, edge_k_scale: float = 1.0, bend_k_scale: float = 0.1,
                 rest_bend_from_mesh: bool = True):
        """
        Load a shell from a .npz file exported by Blender script Export_shell.py
        """
        obj = cls(device, edge_k_scale=edge_k_scale, bend_k_scale=bend_k_scale,
                  rest_bend_from_mesh=rest_bend_from_mesh)
        data = np.load(path, allow_pickle=False)

        # Required arrays
        verts = data["vertices"].astype(np.float64)
        faces = data["faces"].astype(np.int32)
        edges = data["edges"].astype(np.int32) 

        # Find exactly one vertex data layer (prefixed 'v_')
        v_keys = [k for k in data.files if k.startswith("v_")]
        if len(v_keys) < 1:
            raise ValueError(f"Expected at least one 'v_*' vertex layer, found {len(v_keys)}: {v_keys}")
        vertex_layer = data[v_keys[0]]
        v_stiffness = mu.to_scalar_layer_np(vertex_layer)

        obj._ingest(verts, faces, edges, v_stiffness)

        # Build PyVista mesh (optional but requested)
        pv_faces = mu.faces_to_pv_faces(obj.faces.cpu().numpy())
        obj.mesh = pv.PolyData(obj.verts.detach().cpu().numpy(), pv_faces)
        # Store the vertex layer as point data for coloring
        obj.mesh.point_data["Stiffness"] = obj.vertex_stiffness.detach().cpu().numpy()
        return obj

    def _ingest(self, verts, faces, edges, v_stiffness):
        self.verts = torch.as_tensor(verts, device=self.device, dtype=torch.double)
        self.faces = torch.as_tensor(faces, device=self.device, dtype=torch.long)
        self.edges = torch.as_tensor(edges, device=self.device, dtype=torch.long)
        self.vertex_stiffness = torch.as_tensor(v_stiffness, device=self.device, dtype=torch.double)

        # Edge stiffness: average of vertex factors * scale
        vfac = self.vertex_stiffness
        e = self.edges
        k_edge_factor = 0.5 * (vfac[e[:, 0]] + vfac[e[:, 1]])
        self.k_edge = self.edge_k_scale * k_edge_factor

        # Rest edge lengths
        self.L0 = mu.edge_lengths(self.verts, self.edges)

        # Bending topology and rest dihedrals
        self.bend_edges = torch.as_tensor(mu.build_bending_adjacency(self.faces.cpu().numpy()), device=self.device, dtype=torch.long) # (B, 2)
        if self.rest_bend_from_mesh:
            self.theta0 = mu.signed_dihedrals(self.verts, self.bend_edges)
            #self.theta0 = torch.as_tensor(mu.compute_rest_dihedrals(self.verts.cpu().numpy(), self.bend_edges.cpu().numpy()), device=self.device, dtype=torch.double)
        else:
            self.theta0 = torch.zeros((self.bend_edges.shape[0],), device=self.device, dtype=torch.double)

        # Bending stiffness: average vertex factors on edge endpoints * scale
        be = self.bend_edges
        kb_factor = 0.5 * (vfac[be[:, 0]] + vfac[be[:, 1]])
        self.k_bend = self.bend_k_scale * kb_factor


    # -----------------
    # Utility / helpers
    # -----------------

    def update_vertices(self, new_verts):
        self.verts = torch.as_tensor(new_verts, device=self.device, dtype=torch.double)

    # -----------------
    # Energy functions
    # -----------------

    def strech_energy(self, V: torch.Tensor) -> float:
        e = V[self.edges[:, 1]] - V[self.edges[:, 0]]
        L = torch.linalg.vector_norm(e, dim=1)
        dL = L - self.L0
        return 0.5 * torch.sum(self.k_edge * (dL ** 2))

    def bend_energy(self, V: torch.Tensor) -> float:
        theta = mu.signed_dihedrals(V, self.bend_edges)
        dtheta = mu.wrap_angle(theta - self.theta0)
        return 0.5 * torch.sum(self.k_bend * (dtheta ** 2))

    def pressure_energy(self, V: torch.Tensor, pressure: float) -> float:
        return -float(pressure) * mu.signed_volume(V, self.faces)

    def total_energy(self, V: torch.Tensor, pressure: float) -> float:
        return self.strech_energy(V) + self.bend_energy(V) + self.pressure_energy(V, pressure)

    def volume(self, V: torch.Tensor = None) -> float:
        return mu.signed_volume(V if V is not None else self.verts, self.faces).float()


    # -----------------
    # Optimization helper methods
    # -----------------

    def _optimize(
        self,
        obj: Callable[[torch.Tensor], torch.Tensor],
        pinned_idx: Optional[Iterable[int]] = None,
        fix_centroid: bool = True,
        maxiter: int = 200,
        tol: float = 1e-6,
        verbose: bool = True,
        callback: Optional[Callable[[int, float, np.ndarray, float], None]] = None,
        optimizer: str = "LBFGS",
    ):
        """Generic optimizer using torch.optim.

        Parameters mirror the previous SciPy-based helper.  ``obj`` should take a
        ``(N,3)`` tensor and return a scalar energy.  ``pinned_idx`` contains
        vertex indices that should remain fixed, and ``fix_centroid`` keeps the
        centroid at its initial location.  ``callback`` receives
        ``(iter_idx, energy_value, verts_numpy, volume)`` on each step.
        """

        V = torch.nn.Parameter(self.verts.clone())
        c0 = V.detach().mean(dim=0) if fix_centroid else None

        pinned = None
        pinned_pos = None
        if pinned_idx is not None:
            pinned = torch.as_tensor(list(pinned_idx), device=self.device, dtype=torch.long)
            pinned_pos = V.detach()[pinned].clone()

        if optimizer.lower() == "lbfgs":
            opt = torch.optim.LBFGS([V], lr=1.0, max_iter=20, line_search_fn="strong_wolfe")
        elif optimizer.lower() == "adam":
            opt = torch.optim.Adam([V], lr=1e-2)
        else:
            raise ValueError(f"Unknown optimizer '{optimizer}'")

        last_gnorm = None
        for it in range(1, maxiter + 1):
            def closure():
                opt.zero_grad()
                with torch.no_grad():
                    if fix_centroid:
                        V.data -= V.data.mean(dim=0) - c0
                    if pinned is not None:
                        V.data[pinned] = pinned_pos
                E = obj(V)
                E.backward()
                if pinned is not None:
                    V.grad[pinned] = 0
                return E

            opt.step(closure)

            with torch.no_grad():
                if fix_centroid:
                    V.data -= V.data.mean(dim=0) - c0
                if pinned is not None:
                    V.data[pinned] = pinned_pos

            opt.zero_grad()
            E = obj(V)
            E.backward()
            if pinned is not None:
                V.grad[pinned] = 0

            g = V.grad.detach()
            last_gnorm = float(torch.linalg.vector_norm(g)) if g is not None else None

            if callback:
                V_np = V.detach().cpu().numpy()
                callback(it, float(E.detach().cpu().numpy()), V_np, float(self.volume(V)))

            if last_gnorm is not None and last_gnorm < tol:
                break

        self.update_vertices(V.detach())
        if verbose:
            print(f"[optimize] iters={it} grad_norm={last_gnorm:.3e}")
            print(f"final vol={self.volume():.6g}")
        return {"niter": it, "grad_norm": last_gnorm}

    def optimize_pressure(
        self,
        pressure: float,
        pinned_idx: Optional[Iterable[int]] = None,
        fix_centroid: bool = True,
        maxiter: int = 200,
        tol: float = 1e-6,
        on_step=None,
        verbose: bool = True,
    ):
        return self._optimize(
            lambda V: self.total_energy(V, pressure),
            pinned_idx=pinned_idx,
            fix_centroid=fix_centroid,
            maxiter=maxiter,
            tol=tol,
            verbose=verbose,
            callback=on_step,
            optimizer="LBFGS")

    def optimize_volume(
        self,
        V_target: float,
        beta: float = 1.0,
        pinned_idx: Optional[Iterable[int]] = None,
        fix_centroid: bool = True,
        maxiter: int = 200,
        tol: float = 1e-6,
        verbose: bool = True,
        callback=None,
    ):
        return self._optimize(
            lambda V: self.strech_energy(V) + self.bend_energy(V) + 0.5 * beta * (self.volume(V) - V_target) ** 2,
            pinned_idx=pinned_idx,
            fix_centroid=fix_centroid,
            maxiter=maxiter,
            tol=tol,
            verbose=verbose,
            callback=callback,
            optimizer="LBFGS")


    # -----------------
    # Diagnostics
    # -----------------

    def quick_diag(shell, P):
        print("N:", shell.verts.shape[0], "E_scale:", shell.edge_k_scale, "B_scale:", shell.bend_k_scale)
        print("SignedVolume:", shell.volume(shell.verts))
        Vt = shell.verts.clone(); Vt.requires_grad_()
        E = shell.total_energy(Vt, P); E.backward()
        gnorm = torch.linalg.vector_norm(Vt.grad).item()
        print("E0:", float(E.cpu().detach().numpy()), "||grad||:", gnorm)