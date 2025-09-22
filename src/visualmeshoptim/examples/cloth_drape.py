"""Mass-spring cloth draping problem for the visual mesh optimizer."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from ..mesh_io import MeshData
from ..problem import MeshProblem
from ..utils import TunableParameter


@dataclass(frozen=True)
class SpringConnectivity:
    """Container storing spring endpoints and rest lengths."""

    edges: torch.Tensor
    rest_lengths: torch.Tensor


class ClothDrapeProblem(MeshProblem):
    """Simple static cloth draping problem using a mass-spring model.

    The cloth is represented as an ``rows`` x ``cols`` grid initially lying in the
    plane ``y = 0`` with spacing ``spacing`` between neighbouring vertices.  Each
    vertex is assigned the same ``vertex_mass``.  Structural springs connect
    horizontal and vertical neighbours (and optionally diagonals), while bending
    springs connect second-nearest neighbours to discourage sharp folds.  Gravity
    acts in the negative ``y`` direction, pulling the cloth down as the optimizer
    minimises the total potential energy.

    Vertices specified in ``fixed_indices`` remain at their initial locations.
    By default the entire top edge of the cloth (``row = 0``) is fixed so that the
    cloth can drape under its own weight.
    """

    def __init__(
        self,
        *,
        rows: int = 20,
        cols: int = 30,
        spacing: float = 0.05,
        vertex_mass: float = 1.0,
        include_diagonals: bool = True,
        include_bending: bool = True,
        fixed_indices: Sequence[int] | None = None,
        stretch_stiffness: float = 250.0,
        bend_stiffness: float = 25.0,
        gravity: float = 9.81,
        device: str | None = None,
        dtype: torch.dtype = torch.double,
    ) -> None:
        super().__init__(device=device, dtype=dtype)
        if rows < 2 or cols < 2:
            raise ValueError("Cloth grid must have at least 2x2 vertices")
        if spacing <= 0.0:
            raise ValueError("Vertex spacing must be positive")
        if vertex_mass <= 0.0:
            raise ValueError("Vertex mass must be positive")

        self.rows = int(rows)
        self.cols = int(cols)
        self.spacing = float(spacing)
        self.vertex_mass = float(vertex_mass)
        self.include_diagonals = bool(include_diagonals)
        self.include_bending = bool(include_bending)
        self._fixed_indices_input = tuple(int(i) for i in fixed_indices) if fixed_indices is not None else None

        self.stretch_springs: SpringConnectivity | None = None
        self.bend_springs: SpringConnectivity | None = None
        self.mass_per_vertex: torch.Tensor | None = None
        self.fixed_mask: torch.Tensor | None = None
        self.fixed_positions: torch.Tensor | None = None

        self.register_parameter(
            "stretch_stiffness",
            TunableParameter(
                float(stretch_stiffness),
                dtype="float",
                min_value=0.0,
                description="Hooke constant for structural springs.",
            ),
        )
        self.register_parameter(
            "bend_stiffness",
            TunableParameter(
                float(bend_stiffness),
                dtype="float",
                min_value=0.0,
                description="Hooke constant for bending springs.",
            ),
        )
        self.register_parameter(
            "gravity",
            TunableParameter(
                float(gravity),
                dtype="float",
                description="Gravitational acceleration acting along -y.",
            ),
        )

    # ------------------------------------------------------------------
    def load_mesh(self) -> MeshData:
        verts = np.zeros((self.rows * self.cols, 3), dtype=np.float64)
        faces: list[tuple[int, int, int]] = []
        half_w = 0.5 * (self.cols - 1) * self.spacing
        half_h = 0.5 * (self.rows - 1) * self.spacing

        def idx(r: int, c: int) -> int:
            return r * self.cols + c

        for r in range(self.rows):
            for c in range(self.cols):
                index = idx(r, c)
                x = (c * self.spacing) - half_w
                y = 0.0
                z = (r * self.spacing) - half_h
                verts[index, :] = (x, y, z)
                if r + 1 < self.rows and c + 1 < self.cols:
                    v0 = idx(r, c)
                    v1 = idx(r, c + 1)
                    v2 = idx(r + 1, c)
                    v3 = idx(r + 1, c + 1)
                    faces.append((v0, v1, v2))
                    faces.append((v2, v1, v3))

        mesh = MeshData(vertices=verts, faces=np.asarray(faces, dtype=np.int64))
        self.metadata["rows"] = self.rows
        self.metadata["cols"] = self.cols
        self.metadata["spacing"] = self.spacing
        return mesh

    # ------------------------------------------------------------------
    def prepare_data(self, data: MeshData) -> None:  # noqa: D401 - see class docstring
        total_vertices = data.vertices.shape[0]
        self.mass_per_vertex = torch.full(
            (total_vertices,), self.vertex_mass, device=self.device, dtype=self.dtype
        )
        self.fixed_positions = self.initial_vertices.clone().to(device=self.device, dtype=self.dtype).detach()
        self.fixed_mask = torch.zeros((total_vertices,), dtype=torch.bool, device=self.device)

        fixed_indices = self._select_fixed_indices()
        if fixed_indices.size:
            self.fixed_mask[torch.as_tensor(fixed_indices, device=self.device, dtype=torch.long)] = True

        self.stretch_springs = self._build_structural_springs()
        self.bend_springs = self._build_bending_springs() if self.include_bending else None

    # ------------------------------------------------------------------
    def objective(self, vertices: torch.Tensor) -> torch.Tensor:
        energy = torch.zeros((), device=vertices.device, dtype=vertices.dtype)
        stretch_k = float(self.get_parameter_value("stretch_stiffness"))
        if self.stretch_springs is None:
            raise RuntimeError("Problem not prepared")
        energy = energy + self._spring_energy(vertices, self.stretch_springs, stretch_k)

        bend_k = float(self.get_parameter_value("bend_stiffness"))
        if self.include_bending and self.bend_springs is not None and self.bend_springs.edges.numel():
            energy = energy + self._spring_energy(vertices, self.bend_springs, bend_k)

        gravity = float(self.get_parameter_value("gravity"))
        if self.mass_per_vertex is None:
            raise RuntimeError("Problem not prepared")
        # Potential energy m g y
        energy = energy + gravity * torch.sum(self.mass_per_vertex * vertices[:, 1])
        return energy

    # ------------------------------------------------------------------
    def apply_constraints(self, vertices: torch.Tensor) -> torch.Tensor:
        if self.fixed_mask is None or self.fixed_positions is None:
            return vertices
        if not torch.any(self.fixed_mask):
            return vertices
        return torch.where(self.fixed_mask.unsqueeze(1), self.fixed_positions, vertices)

    # ------------------------------------------------------------------
    def get_visualization_quantities(self) -> dict[str, torch.Tensor]:
        if self.fixed_mask is None:
            return {}
        return {"Fixed vertices": self.fixed_mask.to(dtype=self.dtype)}

    # ------------------------------------------------------------------
    def _spring_energy(
        self,
        vertices: torch.Tensor,
        springs: SpringConnectivity,
        stiffness: float,
    ) -> torch.Tensor:
        if springs.edges.numel() == 0 or stiffness == 0.0:
            return torch.zeros((), device=vertices.device, dtype=vertices.dtype)
        diffs = vertices[springs.edges[:, 1]] - vertices[springs.edges[:, 0]]
        lengths = torch.linalg.vector_norm(diffs, dim=1)
        delta = lengths - springs.rest_lengths
        return 0.5 * stiffness * torch.sum(delta * delta)

    # ------------------------------------------------------------------
    def _build_structural_springs(self) -> SpringConnectivity:
        edges: list[tuple[int, int]] = []
        lengths: list[float] = []
        diag_length = math.sqrt(2.0) * self.spacing

        def add_edge(a: int, b: int, rest: float) -> None:
            edges.append((a, b))
            lengths.append(rest)

        for r in range(self.rows):
            for c in range(self.cols):
                idx = r * self.cols + c
                if c + 1 < self.cols:
                    add_edge(idx, idx + 1, self.spacing)
                if r + 1 < self.rows:
                    add_edge(idx, idx + self.cols, self.spacing)
                if self.include_diagonals:
                    if r + 1 < self.rows and c + 1 < self.cols:
                        add_edge(idx, idx + self.cols + 1, diag_length)
                    if r + 1 < self.rows and c - 1 >= 0:
                        add_edge(idx, idx + self.cols - 1, diag_length)

        edges_tensor = torch.as_tensor(edges, device=self.device, dtype=torch.long) if edges else torch.empty((0, 2), dtype=torch.long, device=self.device)
        rest_tensor = (
            torch.as_tensor(lengths, device=self.device, dtype=self.dtype)
            if lengths
            else torch.empty((0,), device=self.device, dtype=self.dtype)
        )
        return SpringConnectivity(edges=edges_tensor, rest_lengths=rest_tensor)

    # ------------------------------------------------------------------
    def _build_bending_springs(self) -> SpringConnectivity:
        edges: list[tuple[int, int]] = []
        lengths: list[float] = []
        axial_rest = 2.0 * self.spacing
        diag_rest = math.sqrt(8.0) * self.spacing

        def add_edge(a: int, b: int, rest: float) -> None:
            edges.append((a, b))
            lengths.append(rest)

        for r in range(self.rows):
            for c in range(self.cols):
                idx = r * self.cols + c
                if c + 2 < self.cols:
                    add_edge(idx, idx + 2, axial_rest)
                if r + 2 < self.rows:
                    add_edge(idx, idx + 2 * self.cols, axial_rest)
                if self.include_diagonals:
                    if r + 2 < self.rows and c + 2 < self.cols:
                        add_edge(idx, idx + 2 * self.cols + 2, diag_rest)
                    if r + 2 < self.rows and c - 2 >= 0:
                        add_edge(idx, idx + 2 * self.cols - 2, diag_rest)

        edges_tensor = torch.as_tensor(edges, device=self.device, dtype=torch.long) if edges else torch.empty((0, 2), dtype=torch.long, device=self.device)
        rest_tensor = (
            torch.as_tensor(lengths, device=self.device, dtype=self.dtype)
            if lengths
            else torch.empty((0,), device=self.device, dtype=self.dtype)
        )
        return SpringConnectivity(edges=edges_tensor, rest_lengths=rest_tensor)

    # ------------------------------------------------------------------
    def _select_fixed_indices(self) -> np.ndarray:
        if self._fixed_indices_input is not None:
            indices = np.unique(np.asarray(self._fixed_indices_input, dtype=np.int64))
        else:
            indices = np.arange(self.cols, dtype=np.int64)
        valid = (indices >= 0) & (indices < self.rows * self.cols)
        if not np.all(valid):
            raise ValueError("Fixed vertex indices out of range")
        return indices

