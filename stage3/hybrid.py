"""Stage 3 hybrid: snap SpatialLM's layout onto plane-fit geometry (the `hyb_snap` variant in TESTING.md).

SpatialLM walls are moved onto the nearest fitted wall plane and their ends onto fitted corners; collinear walls
are merged and doors/windows are re-projected onto the snapped walls. Furniture boxes are kept unchanged.

    python stage3/hybrid.py --point_cloud aligned.ply --spatiallm arch.txt [--spatiallm_boxes object.txt] \
        --output final.txt [--floor 0]
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import planefit  # noqa: E402
from spatiallm import Layout  # noqa: E402

AXIS_TOL = np.deg2rad(15)
SNAP_DIST = 0.35


def wall_axis(ax, ay, bx, by):
    """0 = plane x=const (wall runs along y), 1 = plane y=const, None = oblique."""
    ang = np.arctan2(by - ay, bx - ax) % np.pi
    if min(ang, np.pi - ang) < AXIS_TOL:
        return 1
    if abs(ang - np.pi / 2) < AXIS_TOL:
        return 0
    return None


def plane_coords(fit, axis):
    cs = [(c["c"], c["support"]) for c in fit["candidates"] if c["axis"] == axis]
    for w in fit["walls"]:
        a, b = w["a"], w["b"]
        if wall_axis(a[0], a[1], b[0], b[1]) == axis:
            cs.append((a[axis], 1.0))
    return cs


def nearest(cs, value, tol=None):
    tol = SNAP_DIST if tol is None else tol
    best = None
    for c, sup in cs:
        d = abs(c - value)
        if d < tol and (best is None or d - 0.1 * sup < best[1]):
            best = (c, d - 0.1 * sup)
    return None if best is None else best[0]


def bbox_lines(layout):
    return "\n".join(b.to_language_string() for b in layout.bboxes)


def snap_variant(fit, sl):
    floor, ceil = fit["floor"], fit["ceil"]
    walls, remap = [], {}
    for w in sl.walls:
        ax = wall_axis(w.ax, w.ay, w.bx, w.by)
        a, b = np.array([w.ax, w.ay]), np.array([w.bx, w.by])
        if ax is not None:
            other = 1 - ax
            c = nearest(plane_coords(fit, ax), (a[ax] + b[ax]) / 2)
            c = (a[ax] + b[ax]) / 2 if c is None else c
            perp = plane_coords(fit, other)
            ta, tb = nearest(perp, a[other]), nearest(perp, b[other])
            a[ax], b[ax] = c, c
            a[other] = a[other] if ta is None else ta
            b[other] = b[other] if tb is None else tb
        if np.linalg.norm(b - a) < 0.1:
            continue
        remap[w.id] = len(walls)
        walls.append(dict(a=a, b=b, openings=[], axis=ax))
    walls, remap = merge_collinear(walls, remap)
    for o in sl.doors + sl.windows:
        if o.wall_id not in remap:
            continue
        w = walls[remap[o.wall_id]]
        w["openings"].append(project_opening(o, w["a"], w["b"]))
    return planefit.to_layout_string(walls, floor, ceil, bbox_lines(sl))


def merge_collinear(walls, remap):
    """Snapping can put two walls on the same plane end to end; fuse those into one wall."""
    parent = list(range(len(walls)))
    for i, wi in enumerate(walls):
        for j in range(i):
            wj = walls[parent[j]]
            ax = wi["axis"]
            if ax is None or wj["axis"] != ax or abs(wi["a"][ax] - wj["a"][ax]) > 0.02:
                continue
            o = 1 - ax
            si, sj = sorted([wi["a"][o], wi["b"][o]]), sorted([wj["a"][o], wj["b"][o]])
            if si[0] <= sj[1] + 0.05 and sj[0] <= si[1] + 0.05:
                lo, hi = min(si[0], sj[0]), max(si[1], sj[1])
                wj["a"][o], wj["b"][o] = lo, hi
                parent[i] = parent[j]
                break
    keep = sorted(set(parent))
    new_index = {old: k for k, old in enumerate(keep)}
    return [walls[k] for k in keep], {wid: new_index[parent[idx]] for wid, idx in remap.items()}


def project_opening(o, a, b):
    u = (b - a) / np.linalg.norm(b - a)
    L = np.linalg.norm(b - a)
    width = min(o.width, L)
    s = np.clip((np.array([o.position_x, o.position_y]) - a) @ u, width / 2, L - width / 2)
    c = a + u * s
    return dict(kind=o.entity_label, x=float(c[0]), y=float(c[1]), z=float(o.position_z),
                width=float(width), height=float(o.height))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--point_cloud", required=True)
    ap.add_argument("--spatiallm", required=True, help="SpatialLM layout txt (walls/doors/windows[/boxes])")
    ap.add_argument("--spatiallm_boxes", help="optional separate object-pass layout for boxes")
    ap.add_argument("--output", required=True, help="output layout txt")
    ap.add_argument("--floor", type=float, help="known floor height (aligned clouds: 0); default: detect")
    ap.add_argument("--footprint", default="auto", choices=["auto", "sparse"],
                    help="plane-fit footprint: auto picks params by wall support (tested best), sparse = fixed params")
    args = ap.parse_args()

    with open(args.spatiallm) as f:
        sl = Layout(f.read())
    if args.spatiallm_boxes:
        with open(args.spatiallm_boxes) as f:
            sl.bboxes = Layout(f.read()).bboxes
    fit = planefit.run(args.point_cloud, floor=args.floor, auto=args.footprint == "auto")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(snap_variant(fit, sl))


if __name__ == "__main__":
    main()
