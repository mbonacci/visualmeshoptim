import numpy as np
import torch
import math

def faces_to_pv_faces(faces):
    """Convert (T,3) to PyVista faces flat array [3,i,j,k, 3,...]."""
    T = faces.shape[0]
    out = np.empty(T * 4, dtype=np.int32)
    out[0::4] = 3
    out[1::4] = faces[:, 0]
    out[2::4] = faces[:, 1]
    out[3::4] = faces[:, 2]
    return out

def unique_edges_from_faces(faces):
    """Get sorted unique edges (M,2) from faces (T,3)."""
    f = faces
    e = np.vstack([
        np.sort(f[:, [0, 1]], axis=1),
        np.sort(f[:, [1, 2]], axis=1),
        np.sort(f[:, [2, 0]], axis=1),
    ])
    return np.unique(e, axis=0)

def build_bending_adjacency(faces):
    """
    Build bend-edges (B,4): for each interior undirected edge (i,j),
    find the two adjacent triangles (i,j,k) and (j,i,l) and store (i,j,k,l).
    """
    T = faces.shape[0]
    edge_map = {}  # (min(i,j),max(i,j)) -> [(face_id, opp), (face_id, opp)]
    for fid in range(T):
        a, b, c = faces[fid]
        for i, j, k in ((a, b, c), (b, c, a), (c, a, b)):
            e = tuple(sorted((i, j)))
            edge_map.setdefault(e, []).append((fid, k))
    bends = []
    for (i, j), adj in edge_map.items():
        if len(adj) == 2:
            (_, k), (_, l) = adj
            bends.append((i, j, k, l))
    return np.array(bends, dtype=int) if bends else np.zeros((0, 4), dtype=int)

def triangle_normals(verts, faces, eps=1e-12):
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)  # (T,3)
    n_hat = n / (np.linalg.norm(n, axis=1, keepdims=True) + eps)
    return n, n_hat

def compute_rest_dihedrals(verts, bend_edges, eps=1e-12):
    """theta0 from current shape (natural curvature) over bend-edges (i,j,k,l)."""
    if bend_edges.shape[0] == 0:
        return np.zeros((0,), dtype=float)
    i = bend_edges[:, 0]; j = bend_edges[:, 1]
    k = bend_edges[:, 2]; l = bend_edges[:, 3]
    xi = verts[i]; xj = verts[j]; xk = verts[k]; xl = verts[l]
    n1 = np.cross(xj - xi, xk - xi)
    n2 = np.cross(xi - xj, xl - xj)
    n1 /= (np.linalg.norm(n1, axis=1, keepdims=True) + eps)
    n2 /= (np.linalg.norm(n2, axis=1, keepdims=True) + eps)
    dot = np.clip(np.sum(n1 * n2, axis=1), -1.0, 1.0)
    return np.arccos(dot)

def compute_rest_lengths(verts, edges):
    vi = verts[edges[:, 0]]
    vj = verts[edges[:, 1]]
    return np.linalg.norm(vi - vj, axis=1)

def wrap_angle(x: torch.Tensor) -> torch.Tensor:
    return torch.remainder(x + math.pi, 2.0 * math.pi) - math.pi

def edge_lengths(verts: torch.Tensor, edges: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    e = verts[edges[:, 1]] - verts[edges[:, 0]]
    return torch.linalg.vector_norm(e, dim=1) + 0.0 * eps

def mesh_volume(verts, faces):
    """Signed volume = sum((v0 x v1) x v2)/6 over all faces."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    return np.einsum('ij,ij->i', np.cross(v0, v1), v2).sum() / 6.0

def signed_volume(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    v0 = verts[faces[:, 0]]; v1 = verts[faces[:, 1]]; v2 = verts[faces[:, 2]]
    vol = torch.sum(torch.sum(torch.cross(v0, v1, dim=1) * v2, dim=1)) / 6.0
    return vol

def signed_dihedrals(verts: torch.Tensor, bend_edges: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Signed dihedral at each hinge (i,j,k,l) with unit hinge direction.
    Angle is atan2((n1*n2)^e, n1*n2).
    """
    if bend_edges.numel() == 0:
        return verts.new_zeros((0,))
    i, j, k, l = bend_edges[:, 0], bend_edges[:, 1], bend_edges[:, 2], bend_edges[:, 3]
    xi, xj, xk, xl = verts[i], verts[j], verts[k], verts[l]
    e = xj - xi
    e = e / (torch.linalg.vector_norm(e, dim=1, keepdim=True) + eps)
    n1 = torch.cross(xj - xi, xk - xi, dim=1)
    n2 = torch.cross(xi - xj, xl - xj, dim=1)
    n1 = n1 / (torch.linalg.vector_norm(n1, dim=1, keepdim=True) + eps)
    n2 = n2 / (torch.linalg.vector_norm(n2, dim=1, keepdim=True) + eps)
    sin = torch.sum(torch.cross(n1, n2, dim=1) * e, dim=1)
    cos = torch.clamp(torch.sum(n1 * n2, dim=1), -1.0, 1.0)
    return torch.atan2(sin, cos)

def to_scalar_layer(layer: torch.Tensor) -> torch.Tensor:
    """
    Accept per-vertex layer in shape (N,), (N,1), or (N,3).
    If RGB, convert to luminance; if (N,1) squeeze; else return as is.
    """
    if layer.ndim == 2 and layer.shape[1] == 3:
        # Simple luminance
        w = torch.tensor([0.2126, 0.7152, 0.0722], dtype=layer.dtype, device=layer.device)
        return torch.sum(layer * w, dim=1)
    if layer.ndim == 2 and layer.shape[1] == 1:
        return layer[:, 0]
    if layer.ndim == 1:
        return layer
    raise ValueError("Unsupported vertex layer shape for stiffness factor.")

def to_scalar_layer_np(layer):
    """
    Accept per-vertex layer in shape (N,), (N,1), or (N,3).
    If RGB, convert to luminance; if (N,1) squeeze; else return as is.
    """
    arr = np.asarray(layer)
    if arr.ndim == 2 and arr.shape[1] == 3:
        # Simple luminance
        w = np.array([0.2126, 0.7152, 0.0722], dtype=arr.dtype)
        return (arr * w).sum(axis=1)
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0]
    if arr.ndim == 1:
        return arr
    raise ValueError("Unsupported vertex layer shape for stiffness factor.")