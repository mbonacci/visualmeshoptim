from re import M, S
import re
from tkinter import SE
import numpy as np
from sympy import E
import torch
from typing import Iterable, Optional, Callable
from scipy.optimize import minimize
import pyvista as pv
from zmq import device
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

    def _optimize_autograd(
        self,
        obj : Callable[[torch.Tensor], float],
        pinned_idx: Optional[Iterable[int]] = None,
        fix_centroid: bool = True,
        method: str = "L-BFGS-B",
        maxiter: int = 200,
        tol: float = 1e-6,
        verbose: bool = True,
        callback=None
    ):
        V0 = self.verts.detach().cpu().numpy()
        N = V0.shape[0]
        pinned = np.zeros(N, dtype=bool)
        if pinned_idx is not None:
            pinned[np.unique(np.asarray(list(pinned_idx), dtype=int))] = True
        free = np.where(~pinned)[0]
        c0 = V0.mean(axis=0) 

        iter_k = {"k": 0}
        def cb(xk):
            iter_k["k"] += 1
            V_np = unpack(xk)
            if fix_centroid: V_np -= (V_np.mean(axis=0) - c0)
            self.update_vertices(V_np)  
            if callback:
                E_val, _ = obj_and_grad(xk)  # cheap: one forward pass
                callback(iter_k["k"], E_val, V_np, self.volume())

        def pack(V): return V[free].reshape(-1, 3).ravel()
        def unpack(xfree): V_np = V0.copy(); V_np[free] = xfree.reshape(-1, 3); return V_np

        def obj_and_grad(x):
            U = torch.as_tensor(unpack(x), device=self.device, dtype=torch.double)
            U.requires_grad_()
            E = obj(U)
            E.backward()
            grad = U.grad.detach().cpu().numpy()[free].reshape(-1, 3).ravel()
            return float(E.detach().cpu().numpy()), grad


        x0 = pack(V0)
        res = minimize(obj_and_grad, x0,
                       jac=True, method=method,
                       options={"maxiter": maxiter,  "gtol": tol, "disp": verbose},
                       callback=cb)
        V_final = unpack(res.x)
        self.update_vertices(V_final)
        if verbose:
            print(f"[optimize_autograd] status={res.status} iters={res.nit} msg={res.message}")
            print(f"final vol={self.volume():.6g}")
        return res

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
        return self._optimize_autograd(
            lambda V: self.total_energy(V, pressure), 
            pinned_idx, fix_centroid, "CG", maxiter, tol, verbose, on_step)

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
        return self._optimize_autograd(
            lambda V: self.strech_energy(V) + self.bend_energy(V) + 0.5 * beta * (self.volume(V) - V_target) ** 2,
            pinned_idx, fix_centroid, "L-BFGS-B", maxiter, tol, verbose, callback)


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