"""Example problem demonstrating a pollen shell optimization."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch

from ..geometry import (
    build_bending_adjacency,
    edge_lengths,
    signed_dihedrals,
    signed_volume,
    to_scalar_layer_np,
    wrap_angle,
)
from ..mesh_io import MeshData, load_npz_mesh, load_obj_mesh, load_off_mesh, load_pt_mesh, load_stl_mesh
from ..problem import MeshProblem
from ..utils import TunableParameter


class PollenShellProblem(MeshProblem):
    """Optimization problem based on the historical pollen shell model."""

    def __init__(
        self,
        mesh_path: str,
        *,
        edge_k_scale: float = 1.0,
        bend_k_scale: float = 0.1,
        rest_bend_from_mesh: bool = True,
        device: str | None = None,
        dtype: torch.dtype = torch.double,
    ) -> None:
        self.mesh_path = Path(mesh_path)
        self.edge_k_scale = float(edge_k_scale)
        self.bend_k_scale = float(bend_k_scale)
        self.rest_bend_from_mesh = bool(rest_bend_from_mesh)
        self.edges: torch.Tensor | None = None
        self.bend_edges: torch.Tensor | None = None
        self.vertex_stiffness: torch.Tensor | None = None
        self.k_edge: torch.Tensor | None = None
        self.k_bend: torch.Tensor | None = None
        self.L0: torch.Tensor | None = None
        self.theta0: torch.Tensor | None = None
        super().__init__(device=device, dtype=dtype)
        self.register_parameter(
            "use_pressure_objective",
            TunableParameter(True, dtype="bool", description="Toggle between pressure and volume objective."),
        )
        self.register_parameter(
            "pressure",
            TunableParameter(-2.0, dtype="float", description="Pressure applied to the shell."),
        )
        self.register_parameter(
            "target_volume",
            TunableParameter(0.0, dtype="float", description="Target volume for the volume objective."),
        )
        self.register_parameter(
            "beta",
            TunableParameter(1.0, dtype="float", min_value=0.0, description="Volume penalty weight."),
        )
        super().initialize()
        initial_volume = float(self.volume(self.get_vertices()))
        self.set_parameter("target_volume", initial_volume)

    # ------------------------------------------------------------------
    def load_mesh(self) -> MeshData:
        suffix = self.mesh_path.suffix.lower()
        if suffix == ".npz":
            return load_npz_mesh(self.mesh_path)
        if suffix == ".pt":
            return load_pt_mesh(self.mesh_path)
        if suffix == ".obj":
            return load_obj_mesh(self.mesh_path)
        if suffix == ".off":
            return load_off_mesh(self.mesh_path)
        if suffix == ".stl":
            return load_stl_mesh(self.mesh_path)
        raise ValueError(f"Unsupported mesh format: {self.mesh_path}")

    def prepare_data(self, data: MeshData) -> None:
        v_layer = None
        for key, values in data.vertex_attributes.items():
            if key.startswith("v_"):
                v_layer = values
                break
        if v_layer is None:
            raise ValueError("Expected at least one vertex attribute prefixed with 'v_'")
        self.vertex_stiffness = torch.as_tensor(to_scalar_layer_np(np.asarray(v_layer)), device=self.device, dtype=self.dtype)
        self.edges = torch.as_tensor(data.require_edges(), device=self.device, dtype=torch.long)
        
        self.L0 = edge_lengths(self.vertices, self.edges)
        
        k_factor = 0.5 * (self.vertex_stiffness[self.edges[:, 0]] + self.vertex_stiffness[self.edges[:, 1]])
        self.k_edge = self.edge_k_scale * k_factor
        
        bend_edges_np = build_bending_adjacency(self.faces.detach().cpu().numpy())
        self.bend_edges = torch.as_tensor(bend_edges_np, device=self.device, dtype=torch.long)
        if self.bend_edges.numel() > 0:
            if self.rest_bend_from_mesh:
                self.theta0 = signed_dihedrals(self.vertices, self.bend_edges)
            else:
                self.theta0 = torch.zeros((self.bend_edges.shape[0],), device=self.device, dtype=self.dtype)
            kb_factor = 0.5 * (
                self.vertex_stiffness[self.bend_edges[:, 0]] + self.vertex_stiffness[self.bend_edges[:, 1]]
            )
            self.k_bend = self.bend_k_scale * kb_factor
        else:
            self.theta0 = torch.zeros((0,), device=self.device, dtype=self.dtype)
            self.k_bend = torch.zeros((0,), device=self.device, dtype=self.dtype)

    # ------------------------------------------------------------------
    def get_visualization_quantities(self) -> Dict[str, torch.Tensor]:
        if self.vertex_stiffness is None:
            return {}
        return {"Stiffness": self.vertex_stiffness}

    # ------------------------------------------------------------------
    def objective(self, vertices: torch.Tensor) -> torch.Tensor:
        stretch = self._stretch_energy(vertices)
        bend = self._bend_energy(vertices)
        if bool(self.get_parameter_value("use_pressure_objective")):
            pressure = float(self.get_parameter_value("pressure"))
            return stretch + bend + self._pressure_energy(vertices, pressure)
        target = float(self.get_parameter_value("target_volume"))
        beta = float(self.get_parameter_value("beta"))
        volume_term = 0.5 * beta * (self.volume(vertices) - target) ** 2
        return stretch + bend + volume_term

    # ------------------------------------------------------------------
    def _stretch_energy(self, vertices: torch.Tensor) -> torch.Tensor:
        if self.edges is None or self.L0 is None or self.k_edge is None:
            raise RuntimeError("Problem not prepared")
        diff = vertices[self.edges[:, 1]] - vertices[self.edges[:, 0]]
        lengths = torch.linalg.vector_norm(diff, dim=1)
        dL = lengths - self.L0
        return 0.5 * torch.sum(self.k_edge * (dL ** 2))

    def _bend_energy(self, vertices: torch.Tensor) -> torch.Tensor:
        if self.bend_edges is None or self.theta0 is None or self.k_bend is None:
            raise RuntimeError("Problem not prepared")
        theta = signed_dihedrals(vertices, self.bend_edges)
        dtheta = wrap_angle(theta - self.theta0)
        return 0.5 * torch.sum(self.k_bend * (dtheta ** 2))

    def _pressure_energy(self, vertices: torch.Tensor, pressure: float) -> torch.Tensor:
        return -float(pressure) * signed_volume(vertices, self.get_faces())

    def volume(self, vertices: torch.Tensor | None = None) -> torch.Tensor:
        verts = vertices if vertices is not None else self.get_vertices()
        return signed_volume(verts, self.get_faces())
