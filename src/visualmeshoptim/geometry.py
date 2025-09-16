"""Geometry helpers shared across problems."""
from __future__ import annotations

import math
import numpy as np
import torch


def faces_to_pv_faces(faces: np.ndarray) -> np.ndarray:
    """Convert triangle indices to the flat Polyscope representation."""
    faces = np.asarray(faces, dtype=np.int32)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape (T, 3)")
    flat = np.empty(faces.shape[0] * 4, dtype=np.int32)
    flat[0::4] = 3
    flat[1::4] = faces[:, 0]
    flat[2::4] = faces[:, 1]
    flat[3::4] = faces[:, 2]
    return flat


def unique_edges_from_faces(faces: np.ndarray) -> np.ndarray:
    """Return the sorted unique edges implied by the faces."""
    faces = np.asarray(faces, dtype=np.int64)
    e01 = np.sort(faces[:, [0, 1]], axis=1)
    e12 = np.sort(faces[:, [1, 2]], axis=1)
    e20 = np.sort(faces[:, [2, 0]], axis=1)
    edges = np.vstack((e01, e12, e20))
    return np.unique(edges, axis=0)


def build_bending_adjacency(faces: np.ndarray) -> np.ndarray:
    """Return hinge quadruples ``(i, j, k, l)`` for each interior edge."""
    faces = np.asarray(faces, dtype=np.int64)
    edge_map: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for fid, (a, b, c) in enumerate(faces):
        for i, j, k in ((a, b, c), (b, c, a), (c, a, b)):
            key = tuple(sorted((int(i), int(j))))
            edge_map.setdefault(key, []).append((fid, int(k)))
    hinges = []
    for (i, j), adj in edge_map.items():
        if len(adj) == 2:
            (_, k), (_, l) = adj
            hinges.append((i, j, k, l))
    return np.array(hinges, dtype=np.int64) if hinges else np.zeros((0, 4), dtype=np.int64)


def triangle_normals(verts: np.ndarray, faces: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True) + eps
    return normals / norms


def compute_rest_lengths(verts: np.ndarray, edges: np.ndarray) -> np.ndarray:
    vi = verts[edges[:, 0]]
    vj = verts[edges[:, 1]]
    return np.linalg.norm(vi - vj, axis=1)


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle + math.pi, 2.0 * math.pi) - math.pi


def edge_lengths(verts: torch.Tensor, edges: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    vec = verts[edges[:, 1]] - verts[edges[:, 0]]
    return torch.linalg.vector_norm(vec, dim=1) + 0.0 * eps


def signed_volume(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    return torch.sum(torch.sum(torch.cross(v0, v1, dim=1) * v2, dim=1)) / 6.0


def signed_dihedrals(verts: torch.Tensor, bend_edges: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    if bend_edges.numel() == 0:
        return verts.new_zeros((0,))
    i, j, k, l = (bend_edges[:, idx] for idx in range(4))
    xi, xj, xk, xl = verts[i], verts[j], verts[k], verts[l]
    e = xj - xi
    e = e / (torch.linalg.vector_norm(e, dim=1, keepdim=True) + eps)
    n1 = torch.cross(xj - xi, xk - xi, dim=1)
    n2 = torch.cross(xi - xj, xl - xj, dim=1)
    n1 = n1 / (torch.linalg.vector_norm(n1, dim=1, keepdim=True) + eps)
    n2 = n2 / (torch.linalg.vector_norm(n2, dim=1, keepdim=True) + eps)
    sin_term = torch.sum(torch.cross(n1, n2, dim=1) * e, dim=1)
    cos_term = torch.clamp(torch.sum(n1 * n2, dim=1), -1.0, 1.0)
    return torch.atan2(sin_term, cos_term)


def to_scalar_layer(layer: torch.Tensor) -> torch.Tensor:
    if layer.ndim == 2 and layer.shape[1] == 3:
        weights = torch.tensor([0.2126, 0.7152, 0.0722], dtype=layer.dtype, device=layer.device)
        return torch.sum(layer * weights, dim=1)
    if layer.ndim == 2 and layer.shape[1] == 1:
        return layer[:, 0]
    if layer.ndim == 1:
        return layer
    raise ValueError("Unsupported vertex layer shape")


def to_scalar_layer_np(layer: np.ndarray) -> np.ndarray:
    arr = np.asarray(layer)
    if arr.ndim == 2 and arr.shape[1] == 3:
        weights = np.array([0.2126, 0.7152, 0.0722], dtype=arr.dtype)
        return (arr * weights).sum(axis=1)
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0]
    if arr.ndim == 1:
        return arr
    raise ValueError("Unsupported vertex layer shape")
