import bpy
import os
import torch

obj = bpy.context.object
mesh = obj.data

# Export as Torch tensors in a .pt file
out_path = bpy.path.abspath("//pollen.pt")

# === Export vertices ===
verts_list = [tuple(v.co[:]) for v in mesh.vertices]
verts = torch.tensor(verts_list, dtype=torch.float32)

# === Export faces (as triangles) ===
faces_list = []
for poly in mesh.polygons:
    if len(poly.vertices) == 3:
        faces_list.append(tuple(poly.vertices[:]))
    elif len(poly.vertices) == 4:
        # triangulate quads manually
        v = poly.vertices
        faces_list.append((v[0], v[1], v[2]))
        faces_list.append((v[0], v[2], v[3]))

# If mesh has no faces, create an empty long tensor of shape (0,3)
faces = torch.tensor(faces_list, dtype=torch.long) if faces_list else torch.empty((0, 3), dtype=torch.long)

# === Export edges (topology only) ===
edges_list = [tuple(e.vertices[:]) for e in mesh.edges]
edges = torch.tensor(edges_list, dtype=torch.long) if edges_list else torch.empty((0, 2), dtype=torch.long)

# === Export vertex color attributes (as per-vertex fields) ===
vertex_fields = {}
for attr in getattr(mesh, 'color_attributes', []):
    if attr.domain == 'POINT' and attr.data_type in {'FLOAT_COLOR', 'BYTE_COLOR'}:
        values_list = [tuple(d.color[:3]) for d in attr.data]
        # handle case of zero vertices gracefully
        values = torch.tensor(values_list, dtype=torch.float32) if values_list else torch.empty((0, 3), dtype=torch.float32)
        vertex_fields[attr.name] = values

# === Assemble Torch tensors and save to .pt ===
payload = {
    'vertices': verts,   # (N,3) float32
    'faces': faces,      # (T,3) int64
    'edges': edges,      # (M,2) int64
}
# Add exactly the vertex layers, prefixed with 'v_'
for k, v in vertex_fields.items():
    payload[f'v_{k}'] = v

torch.save(payload, out_path)
print(f"Exported mesh with vertex attributes to {out_path}")
