import bpy
import numpy as np
import os

obj = bpy.context.object
mesh = obj.data
out_path = bpy.path.abspath("//pollen.npz")

# === Export vertices ===
verts = np.array([v.co[:] for v in mesh.vertices], dtype=np.float32)

# === Export faces (as triangles) ===
faces = []
for poly in mesh.polygons:
    if len(poly.vertices) == 3:
        faces.append(poly.vertices[:])
    elif len(poly.vertices) == 4:
        # triangulate quads manually
        v = poly.vertices
        faces.append([v[0], v[1], v[2]])
        faces.append([v[0], v[2], v[3]])
faces = np.array(faces, dtype=np.int32)

# === Export edges ===
edges = np.array([e.vertices[:] for e in mesh.edges], dtype=np.int32)

# === Export vertex color attributes ===
vertex_fields = {}
for attr in mesh.color_attributes:
    if attr.domain == 'POINT' and attr.data_type in {'FLOAT_COLOR', 'BYTE_COLOR'}:
        values = np.array([d.color[:3] for d in attr.data], dtype=np.float32)
        vertex_fields[attr.name] = values

# === Compute edge colors (averages of vertex attributes) ===
edge_fields = {}
for name, vcolors in vertex_fields.items():
    ecolors = np.array([
        (vcolors[e[0]] + vcolors[e[1]]) / 2.0 for e in edges
    ], dtype=np.float32)
    edge_fields[name] = ecolors

# === Save all to .npz ===
np.savez(out_path,
         vertices=verts,
         faces=faces,
         edges=edges,
         **{f"v_{k}": v for k, v in vertex_fields.items()},
         **{f"e_{k}": e for k, e in edge_fields.items()})

print(f"Exported mesh with vertex + edge attributes to {out_path}")
