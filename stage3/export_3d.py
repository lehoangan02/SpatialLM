"""Layout text -> binary glTF (.glb) you can open in a 3D viewer: walls with real holes, door/window panels, floor.

    python stage3/export_3d.py LAYOUT.txt OUT.glb [--point_cloud aligned.ply] [--max_points 300000] [--thickness 0.1]

Source layout is z up, right-handed; glTF is y up, right-handed, so (x, y, z) -> (x, z, -y).
"""

import argparse
import json
import math
import struct
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spatiallm import Layout  # noqa: E402

COLOURS = {"wall": [0.82, 0.82, 0.78, 1], "floor": [0.55, 0.42, 0.28, 1], "door": [0.10, 0.60, 0.25, 1],
           "window": [0.15, 0.45, 0.90, 1]}
SLIVER = 1e-3


def gltf(p):
    p = np.asarray(p, dtype=np.float32)
    return np.c_[p[:, 0], p[:, 2], -p[:, 1]]


def box(center, t, n, size):
    """Corners and faces of a box: t, n = floor-plane axes, size = (along t, along n, along z). Flat-shaded: 24 vertices."""
    up = np.array([0, 0, 1.0])
    axes = [(t, size[0] / 2), (n, size[1] / 2), (up, size[2] / 2)]
    verts, normals, tris = [], [], []
    for i in range(3):
        (a, ha), (b, hb), (c, hc) = axes[i], axes[(i + 1) % 3], axes[(i + 2) % 3]
        for s in (1, -1):
            cen = center + s * a * ha
            k = len(verts)
            verts += [cen + b * hb * u + c * hc * v for u, v in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            normals += [s * a] * 4
            # handedness of the mapped frame decides winding; both orders checked against the normal below
            quad = [(k, k + 1, k + 2), (k, k + 2, k + 3)]
            e = np.cross(verts[k + 1] - verts[k], verts[k + 2] - verts[k])
            tris += quad if e @ (s * a) > 0 else [q[::-1] for q in quad]
    return np.array(verts), np.array(normals), np.array(tris)


def wall_blocks(L, H, cuts, t):
    """Split a wall of length L, height H into solid blocks around openings; the two ends grow by t to close the corners."""
    blocks, u = [], -t
    for u0, u1, z0, z1 in sorted(cuts):
        blocks += [(u, u0, 0.0, H), (u0, u1, 0.0, z0), (u0, u1, z1, H)]
        u = u1
    blocks.append((u, L + t, 0.0, H))
    return [b for b in blocks if b[1] - b[0] > SLIVER and b[3] - b[2] > SLIVER]


def floor_loop(walls, tol=0.05):
    segs = [(np.array([w.ax, w.ay]), np.array([w.bx, w.by])) for w in walls]
    first = segs.pop(0)
    loop, cur = [first[0], first[1]], first[1]
    while segs:
        k = next((i for i, (a, b) in enumerate(segs) if min(np.linalg.norm(a - cur), np.linalg.norm(b - cur)) < tol), None)
        if k is None:
            break
        a, b = segs.pop(k)
        cur = b if np.linalg.norm(a - cur) < tol else a
        loop.append(cur)
    return np.array(loop[:-1] if np.linalg.norm(loop[0] - loop[-1]) < tol else loop)


def build(layout, thickness):
    """Returns {part: [(verts, normals, tris), ...]} in the source frame."""
    parts = {k: [] for k in COLOURS}
    cx = np.mean([[w.ax, w.ay] for w in layout.walls] + [[w.bx, w.by] for w in layout.walls], axis=0)
    openings = [("door", o) for o in layout.doors] + [("window", o) for o in layout.windows]
    for w in layout.walls:
        a, b = np.array([w.ax, w.ay, 0.0]), np.array([w.bx, w.by, 0.0])
        L = np.linalg.norm(b - a)
        t = (b - a) / L
        n = np.array([t[1], -t[0], 0.0])
        if n[:2] @ (a[:2] - cx) < 0:  # always grow away from the room centre
            n = -n
        cuts = []
        for kind, o in openings:
            if o.wall_id != w.id:
                continue
            along = (np.array([o.position_x, o.position_y, 0.0]) - a) @ t
            z0, z1 = max(o.position_z - o.height / 2, 0.0), min(o.position_z + o.height / 2, w.height)
            if kind == "door" and z0 < 0.15:  # doors stand on the floor
                z0 = 0.0
            u0, u1 = max(along - o.width / 2, 0.0), min(along + o.width / 2, L)
            cuts.append((u0, u1, z0, z1))
            mid = a + t * (u0 + u1) / 2 + n * thickness / 2 + np.array([0, 0, (z0 + z1) / 2])
            parts[kind].append(box(mid, t, n, (u1 - u0, 0.03, z1 - z0)))
        for u0, u1, z0, z1 in wall_blocks(L, w.height, cuts, thickness):
            c = a + t * (u0 + u1) / 2 + n * thickness / 2 + np.array([0, 0, (z0 + z1) / 2])
            parts["wall"].append(box(c, t, n, (u1 - u0, thickness, z1 - z0)))
    loop = floor_loop(layout.walls)
    centre = np.r_[loop.mean(0), 0.0]
    v = np.vstack([centre] + [np.r_[p, 0.0] for p in loop])
    m = len(loop)
    tris = np.array([(0, 1 + i, 1 + (i + 1) % m) for i in range(m)])
    e = np.cross(v[tris[0, 1]] - v[0], v[tris[0, 2]] - v[0])
    parts["floor"].append((v, np.tile([0, 0, 1.0], (m + 1, 1)), tris if e[2] > 0 else tris[:, ::-1]))
    return parts


class Glb:
    def __init__(self):
        self.bin, self.views, self.accessors = bytearray(), [], []

    def add(self, arr, ctype, gltype, target=None, normalized=False, bounds=False):
        while len(self.bin) % 4:
            self.bin.append(0)
        self.views.append({"buffer": 0, "byteOffset": len(self.bin), "byteLength": arr.nbytes, **({"target": target} if target else {})})
        self.bin += arr.tobytes()
        acc = {"bufferView": len(self.views) - 1, "componentType": ctype, "count": len(arr) if arr.ndim > 1 else arr.size, "type": gltype}
        if normalized:
            acc["normalized"] = True
        if bounds:
            acc["min"], acc["max"] = arr.min(0).tolist(), arr.max(0).tolist()
        self.accessors.append(acc)
        return len(self.accessors) - 1


def write(parts, path, cloud=None):
    g = Glb()
    materials, meshes, nodes = [], [], []
    for name, items in parts.items():
        if not items:
            continue
        vs, ns, ts, off = [], [], [], 0
        for v, n, t in items:
            vs.append(v), ns.append(n), ts.append(t + off)
            off += len(v)
        prim = {"attributes": {"POSITION": g.add(gltf(np.vstack(vs)), 5126, "VEC3", 34962, bounds=True),
                               "NORMAL": g.add(gltf(np.vstack(ns)), 5126, "VEC3", 34962)},
                "indices": g.add(np.vstack(ts).astype(np.uint32).ravel(), 5125, "SCALAR", 34963), "material": len(materials)}
        materials.append({"name": name, "doubleSided": True,
                          "pbrMetallicRoughness": {"baseColorFactor": COLOURS[name], "metallicFactor": 0, "roughnessFactor": 0.9}})
        meshes.append({"name": name, "primitives": [prim]})
        nodes.append({"name": name, "mesh": len(meshes) - 1})
    if cloud is not None:
        pts, cols = cloud
        rgba = np.c_[(np.clip(cols, 0, 1) ** 2.2 * 255).astype(np.uint8), np.full(len(cols), 255, np.uint8)]  # COLOR_0 is linear
        prim = {"attributes": {"POSITION": g.add(gltf(pts), 5126, "VEC3", 34962, bounds=True),
                               "COLOR_0": g.add(rgba, 5121, "VEC4", 34962, normalized=True)}, "mode": 0}
        meshes.append({"name": "scan", "primitives": [prim]})
        nodes.append({"name": "scan", "mesh": len(meshes) - 1})
    doc = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": list(range(len(nodes)))}], "nodes": nodes,
           "meshes": meshes, "materials": materials, "accessors": g.accessors, "bufferViews": g.views,
           "buffers": [{"byteLength": len(g.bin)}]}
    js = json.dumps(doc, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    g.bin += b"\0" * (-len(g.bin) % 4)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, 28 + len(js) + len(g.bin)))
        f.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
        f.write(struct.pack("<II", len(g.bin), 0x004E4942) + bytes(g.bin))


def export(layout_path, out_path, point_cloud=None, max_points=300_000, thickness=0.1):
    """Layout txt -> .glb, optionally with up to max_points scan points."""
    layout = Layout(open(layout_path).read())
    cloud = None
    if point_cloud:
        import open3d as o3d
        pc = o3d.io.read_point_cloud(point_cloud)
        P, C = np.asarray(pc.points), np.asarray(pc.colors)
        keep = np.random.default_rng(0).choice(len(P), min(max_points, len(P)), replace=False)
        cloud = (P[keep], C[keep])
    write(build(layout, thickness), out_path, cloud)
    return layout


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("layout")
    ap.add_argument("out")
    ap.add_argument("--point_cloud")
    ap.add_argument("--max_points", type=int, default=300_000)
    ap.add_argument("--thickness", type=float, default=0.1)
    a = ap.parse_args()
    layout = export(a.layout, a.out, a.point_cloud, a.max_points, a.thickness)
    print(f"{len(layout.walls)} walls, {len(layout.doors)} doors, {len(layout.windows)} windows -> {a.out} ({os.path.getsize(a.out) / 1e6:.2f} MB)")
