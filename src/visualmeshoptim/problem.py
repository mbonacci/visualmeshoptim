"""Problem abstractions for the visual mesh optimizer."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch

from .mesh_io import MeshData
from .utils import TunableParameter, resolve_device


class MeshProblem(ABC):
    """Abstract base class describing a mesh optimization problem."""

    def __init__(self, *, device: Optional[str] = None, dtype: torch.dtype = torch.double) -> None:
        self.device = resolve_device(device)
        self.dtype = dtype
        self.vertices: Optional[torch.Tensor] = None
        self.faces: Optional[torch.Tensor] = None
        self.initial_vertices: Optional[torch.Tensor] = None
        self.mesh_data: Optional[MeshData] = None
        self.metadata: Dict[str, Any] = {}
        self._parameters: Dict[str, TunableParameter] = {}
        self._is_initialized = False

    @abstractmethod
    def load_mesh(self) -> MeshData:
        """Return the mesh data used for the problem."""

    def initialize(self) -> None:
        data = self.load_mesh()
        if not isinstance(data, MeshData):
            raise TypeError("load_mesh must return a MeshData instance")
        self.mesh_data = data
        self.vertices = torch.as_tensor(data.vertices, device=self.device, dtype=self.dtype)
        self.faces = torch.as_tensor(data.faces, device=self.device, dtype=torch.long)
        self.initial_vertices = self.vertices.detach().clone()
        self.prepare_data(data)
        self._is_initialized = True

    @abstractmethod
    def prepare_data(self, data: MeshData) -> None:
        """Prepare auxiliary tensors after mesh loading."""

    @abstractmethod
    def objective(self, vertices: torch.Tensor) -> torch.Tensor:
        """Return the scalar objective value for ``vertices``."""

    def apply_constraints(self, vertices: torch.Tensor) -> torch.Tensor:
        """Return constrained vertices. Subclasses can override."""
        return vertices

    def on_step(self, iteration: int, vertices: torch.Tensor, value: torch.Tensor) -> None:
        """Hook executed after each optimization step."""
        return None

    def get_visualization_quantities(self) -> Dict[str, torch.Tensor]:
        return {}

    def require_initialized(self) -> None:
        if not self._is_initialized:
            raise RuntimeError("Problem has not been initialized; call initialize() first")

    def is_initialized(self) -> bool:
        return self._is_initialized

    def get_vertices(self) -> torch.Tensor:
        self.require_initialized()
        assert self.vertices is not None
        return self.vertices

    def get_faces(self) -> torch.Tensor:
        self.require_initialized()
        assert self.faces is not None
        return self.faces

    def update_vertices(self, new_vertices: torch.Tensor) -> None:
        self.require_initialized()
        self.vertices = torch.as_tensor(new_vertices, device=self.device, dtype=self.dtype)

    def reset_vertices(self) -> None:
        self.require_initialized()
        assert self.initial_vertices is not None
        self.vertices = self.initial_vertices.clone().to(device=self.device)

    # Parameter management -------------------------------------------------
    def register_parameter(self, name: str, parameter: TunableParameter) -> None:
        self._parameters[name] = parameter

    def parameters(self) -> Dict[str, TunableParameter]:
        return self._parameters

    def get_parameter(self, name: str) -> TunableParameter:
        return self._parameters[name]

    def set_parameter(self, name: str, value: Any) -> None:
        if name not in self._parameters:
            raise KeyError(name)
        self._parameters[name].update(value)

    def get_parameter_value(self, name: str) -> Any:
        return self._parameters[name].value
