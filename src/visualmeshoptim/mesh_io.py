"""Mesh input/output helpers."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from .geometry import unique_edges_from_faces


@dataclass
class MeshData:
    """Simple container representing a triangular mesh."""

    vertices: np.ndarray
    faces: np.ndarray
    edges: Optional[np.ndarray] = None
    vertex_attributes: Dict[str, np.ndarray] = field(default_factory=dict)

    def clone(self) -> "MeshData":
        return MeshData(
            vertices=np.array(self.vertices, copy=True),
            faces=np.array(self.faces, copy=True),
            edges=None if self.edges is None else np.array(self.edges, copy=True),
            vertex_attributes={k: np.array(v, copy=True) for k, v in self.vertex_attributes.items()},
        )

    def require_edges(self) -> np.ndarray:
        if self.edges is None:
            self.edges = unique_edges_from_faces(self.faces)
        return self.edges


def _ensure_triangle_faces(faces: np.ndarray) -> np.ndarray:
    faces = np.asarray(faces, dtype=np.int64)
    if faces.ndim != 2:
        raise ValueError("faces array must be 2-D")
    if faces.shape[1] == 3:
        return faces
    if faces.shape[1] == 0:
        return np.empty((0, 3), dtype=np.int64)
    raise ValueError("Only triangular faces are supported")


def load_npz_mesh(path: str | Path) -> MeshData:
    """Load a mesh stored with :func:`numpy.savez` or :func:`numpy.savez_compressed`."""
    path = Path(path)
    data = np.load(path, allow_pickle=False)
    vertices = np.asarray(data["vertices"], dtype=np.float64)
    faces = _ensure_triangle_faces(data["faces"])
    edges = np.asarray(data["edges"], dtype=np.int64) if "edges" in data else None
    vertex_attributes: Dict[str, np.ndarray] = {}
    for key in data.files:
        if key.startswith("v_"):
            vertex_attributes[key] = np.asarray(data[key])
    return MeshData(vertices=vertices, faces=faces, edges=edges, vertex_attributes=vertex_attributes)


def load_pt_mesh(path: str | Path) -> MeshData:
    """Load a mesh stored via :func:`torch.save`."""
    path = Path(path)
    payload = torch.load(path, map_location="cpu")
    vertices = np.asarray(payload["vertices"], dtype=np.float64)
    faces = _ensure_triangle_faces(np.asarray(payload["faces"], dtype=np.int64))
    edges = payload.get("edges")
    if edges is not None:
        edges = np.asarray(edges, dtype=np.int64)
    vertex_attributes = {
        key: np.asarray(value)
        for key, value in payload.items()
        if isinstance(key, str) and key.startswith("v_")
    }
    return MeshData(vertices=vertices, faces=faces, edges=edges, vertex_attributes=vertex_attributes)


def _load_with_trimesh(path: str | Path) -> MeshData:
    path = Path(path)
    try:
        import trimesh
    except Exception as exc:  # pragma: no cover - exercised when trimesh missing
        raise ImportError("trimesh is required to load this file format") from exc
    mesh = trimesh.load_mesh(path, process=False)
    if not hasattr(mesh, "faces"):
        raise ValueError(f"Unable to load triangular faces from {path}")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = _ensure_triangle_faces(np.asarray(mesh.faces))
    edges = unique_edges_from_faces(faces)
    vertex_attributes: Dict[str, np.ndarray] = {}
    if mesh.visual.kind == "vertex" and mesh.visual.vertex_colors.size:
        vertex_attributes["v_colors"] = np.asarray(mesh.visual.vertex_colors[:, :3], dtype=np.float64)
    return MeshData(vertices=vertices, faces=faces, edges=edges, vertex_attributes=vertex_attributes)


def load_obj_mesh(path: str | Path) -> MeshData:
    return _load_with_trimesh(path)


def load_off_mesh(path: str | Path) -> MeshData:
    return _load_with_trimesh(path)


def load_stl_mesh(path: str | Path) -> MeshData:
    return _load_with_trimesh(path)


def save_obj_mesh(path: str | Path, mesh: MeshData) -> None:
    _save_with_trimesh(path, mesh)


def save_off_mesh(path: str | Path, mesh: MeshData) -> None:
    _save_with_trimesh(path, mesh)


def save_stl_mesh(path: str | Path, mesh: MeshData) -> None:
    _save_with_trimesh(path, mesh)


def _save_with_trimesh(path: str | Path, mesh: MeshData) -> None:
    path = Path(path)
    try:
        import trimesh
    except Exception as exc:  # pragma: no cover - exercised when trimesh missing
        raise ImportError("trimesh is required to export this file format") from exc
    export_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
    export_mesh.export(path)


def export_active_blender_mesh(output_path: str | Path, *, vertex_attribute_prefix: str = "v_") -> None:
    """Export the active Blender object into ``output_path`` as a ``.pt`` file."""
    try:  # pragma: no cover - requires Blender runtime
        import bpy
    except Exception as exc:  # pragma: no cover - requires Blender runtime
        raise ImportError("This helper must be executed inside Blender") from exc

    obj = bpy.context.object
    if obj is None:
        raise RuntimeError("No active object selected in Blender")
    mesh = obj.data
    out_path = bpy.path.abspath(str(output_path))

    verts = [tuple(v.co[:]) for v in mesh.vertices]
    faces = []
    for poly in mesh.polygons:
        if len(poly.vertices) == 3:
            faces.append(tuple(poly.vertices[:]))
        elif len(poly.vertices) == 4:
            v = poly.vertices
            faces.append((v[0], v[1], v[2]))
            faces.append((v[0], v[2], v[3]))
    edges = [tuple(e.vertices[:]) for e in mesh.edges]

    payload: Dict[str, torch.Tensor] = {
        "vertices": torch.tensor(verts, dtype=torch.float32),
        "faces": torch.tensor(faces, dtype=torch.long) if faces else torch.empty((0, 3), dtype=torch.long),
        "edges": torch.tensor(edges, dtype=torch.long) if edges else torch.empty((0, 2), dtype=torch.long),
    }

    for attr in getattr(mesh, "color_attributes", []):
        if attr.domain == "POINT" and attr.data_type in {"FLOAT_COLOR", "BYTE_COLOR"}:
            values = [tuple(d.color[:3]) for d in attr.data]
            tensor = torch.tensor(values, dtype=torch.float32) if values else torch.empty((0, 3), dtype=torch.float32)
            payload[f"{vertex_attribute_prefix}{attr.name}"] = tensor

    torch.save(payload, out_path)
