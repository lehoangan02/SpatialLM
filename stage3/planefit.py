"""Stage 3, geometry only: walls / doors / windows from plane fitting on an aligned point cloud.

Assumes the SpatialLM convention (z up, walls roughly parallel to x or y, metric scale).
Writes the layout in SpatialLM's text format so every variant can be scored and drawn the same way.

    python stage3/planefit.py --point_cloud outputs/aligned.ply --output outputs/planefit.txt
"""

import argparse
import json

import numpy as np
import open3d as o3d
from scipy import ndimage
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union


def load_points(path, voxel=0.025):
    pcd = o3d.io.read_point_cloud(path)
    pcd = pcd.voxel_down_sample(voxel)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=15, std_ratio=2.0)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    return np.asarray(pcd.points), np.asarray(pcd.normals)


def floor_and_ceiling(p, n, bin_size=0.02, floor=None):
    z = p[np.abs(n[:, 2]) > 0.9, 2]
    lo, hi = np.percentile(p[:, 2], 0.2), np.percentile(p[:, 2], 99.8)
    h, e = np.histogram(z, np.arange(lo, hi + bin_size, bin_size))
    h = ndimage.gaussian_filter1d(h.astype(float), 1)
    c = 0.5 * (e[1:] + e[:-1])
    if floor is None:
        # lowest layer that is at least a quarter as dense as the densest low layer (beds/tables sit higher)
        low = np.where(c < np.median(p[:, 2]))[0]
        i = low[np.argmax(h[low] > 0.25 * h[low].max())]
        while i + 1 < len(h) and h[i + 1] > h[i]:  # climb to the top of that layer
            i += 1
        floor = c[i]
    # MASt3R ceilings come out as several stacked layers: take the densest one, not the outermost
    upper = c > floor + 1.8
    ceiling = c[upper][np.argmax(h[upper])] if upper.any() and h[upper].max() > 0.15 * h.max() else hi
    return float(floor), float(ceiling)


def wall_candidates(p, n, floor, ceil, axis, min_len=0.4, bin_size=0.02):
    """Planes perpendicular to `axis` that look like walls (tall, vertically well covered)."""
    other = 1 - axis
    sel = (np.abs(n[:, axis]) > 0.85) & (p[:, 2] > floor + 0.2) & (p[:, 2] < ceil - 0.2)
    q = p[sel]
    if len(q) == 0:
        return []
    coords = q[:, axis]
    h, e = np.histogram(coords, np.arange(coords.min(), coords.max() + bin_size, bin_size))
    hs = ndimage.gaussian_filter1d(h.astype(float), 1)
    c = 0.5 * (e[1:] + e[:-1])
    order = np.argsort(-hs)
    peaks = []
    for i in order:
        if hs[i] < 60:
            break
        if all(abs(c[i] - c[j]) > 0.15 for j in peaks):
            peaks.append(i)
    zbins = max(int((ceil - floor) / 0.1), 1)
    out = []
    for i in peaks:
        pc = c[i]
        band = q[np.abs(coords - pc) < 0.06]
        t = band[:, other]
        tb = np.floor((t - t.min()) / 0.1).astype(int)
        zb = np.clip(np.floor((band[:, 2] - floor) / 0.1).astype(int), 0, zbins - 1)
        occ = np.zeros((tb.max() + 1, zbins), bool)
        occ[tb, zb] = True
        cover = occ.mean(1)
        good = cover > 0.3
        # bridge gaps up to 1.5 m (doorways) between covered stretches
        good = ndimage.binary_closing(good, structure=np.ones(15)) | good
        lab, k = ndimage.label(good)
        for j in range(1, k + 1):
            idx = np.where(lab == j)[0]
            length = (idx[-1] - idx[0] + 1) * 0.1
            if length < min_len or cover[idx].mean() < 0.3:
                continue
            refined = np.median(band[(tb >= idx[0]) & (tb <= idx[-1]), axis])
            out.append(dict(axis=axis, c=float(refined), t0=float(t.min() + idx[0] * 0.1),
                            t1=float(t.min() + (idx[-1] + 1) * 0.1), support=float(cover[idx].mean()),
                            length=float(length)))
    return out


def occupancy(p, floor, ceil, cell=0.05, close_iter=2):
    m = (p[:, 2] > floor - 0.05) & (p[:, 2] < ceil + 0.05)
    xy = p[m, :2]
    lo = xy.min(0) - 0.2
    ij = np.floor((xy - lo) / cell).astype(int)
    grid = np.zeros(ij.max(0) + 1, bool)
    grid[ij[:, 0], ij[:, 1]] = True
    grid = ndimage.binary_closing(grid, iterations=close_iter)
    return grid, lo, cell


def room_polygon(p, cands, floor, ceil, fill_thr=0.15, cell=0.1, close_iter=2):
    grid, lo, cell = occupancy(p, floor, ceil, cell, close_iter)
    xs = sorted({w["c"] for w in cands if w["axis"] == 0})
    ys = sorted({w["c"] for w in cands if w["axis"] == 1})
    occ_pts = np.argwhere(grid) * cell + lo
    xmin, ymin = np.percentile(occ_pts, 0.5, 0)
    xmax, ymax = np.percentile(occ_pts, 99.5, 0)
    def with_bounds(vals, vmin, vmax):
        vals = [v for v in vals if vmin - 0.3 < v < vmax + 0.3]
        # the scan's bounding box only stands in for a wall when no fitted wall is near it
        if not vals or min(vals) > vmin + 0.5:
            vals.append(vmin)
        if max(vals) < vmax - 0.5:
            vals.append(vmax)
        return sorted(set(vals))

    xs, ys = with_bounds(xs, xmin, xmax), with_bounds(ys, ymin, ymax)
    cells = []
    for x0, x1 in zip(xs[:-1], xs[1:]):
        for y0, y1 in zip(ys[:-1], ys[1:]):
            if x1 - x0 < 0.05 or y1 - y0 < 0.05:
                continue
            i0, i1 = int((x0 - lo[0]) / cell), max(int((x1 - lo[0]) / cell), int((x0 - lo[0]) / cell) + 1)
            j0, j1 = int((y0 - lo[1]) / cell), max(int((y1 - lo[1]) / cell), int((y0 - lo[1]) / cell) + 1)
            sub = grid[max(i0, 0):i1, max(j0, 0):j1]
            if sub.size and sub.mean() > fill_thr:
                cells.append(box(x0, y0, x1, y1))
    poly = unary_union(cells)
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda g: g.area)
    return Polygon(poly.exterior).simplify(0.01)


def polygon_walls(poly):
    pts = list(poly.exterior.coords)[:-1]
    # merge collinear runs
    keep = []
    for i in range(len(pts)):
        a, b, c = np.array(pts[i - 1]), np.array(pts[i]), np.array(pts[(i + 1) % len(pts)])
        if abs(np.cross(b - a, c - b)) > 1e-6:
            keep.append(pts[i])
    return [(np.array(keep[i]), np.array(keep[(i + 1) % len(keep)])) for i in range(len(keep))]


def wall_frame(a, b, poly):
    d = b - a
    L = np.linalg.norm(d)
    u = d / L
    nrm = np.array([-u[1], u[0]])
    mid = (a + b) / 2
    if not poly.buffer(1e-6).contains(Point(*(mid + 0.05 * nrm))):
        nrm = -nrm  # make the normal point into the room
    return u, nrm, L


def wall_support(p, a, b, floor, ceil, tol=0.07, cell=0.1):
    """Fraction of the wall rectangle that has points within `tol` of it."""
    d = b - a
    L = np.linalg.norm(d)
    u = d / L
    nrm = np.array([-u[1], u[0]])
    rel = p[:, :2] - a
    s = rel @ u
    dist = np.abs(rel @ nrm)
    m = (dist < tol) & (s >= 0) & (s <= L) & (p[:, 2] >= floor) & (p[:, 2] <= ceil)
    nu, nz = max(int(L / cell), 1), max(int((ceil - floor) / cell), 1)
    occ = np.zeros((nu, nz), bool)
    iu = np.clip((s[m] / cell).astype(int), 0, nu - 1)
    iz = np.clip(((p[m, 2] - floor) / cell).astype(int), 0, nz - 1)
    occ[iu, iz] = True
    return float(occ.mean())


def opening_evidence(p, a, b, nrm, floor, ceil, cell=0.05, tol=0.07):
    """Per-cell grids on the wall: wall points, points behind it, points in front of it."""
    d = b - a
    L = np.linalg.norm(d)
    u = d / L
    rel = p[:, :2] - a
    s = rel @ u
    off = rel @ nrm  # >0 inside the room
    inside = (s >= 0) & (s <= L) & (p[:, 2] >= floor) & (p[:, 2] <= ceil)
    nu, nz = max(int(L / cell), 1), max(int((ceil - floor) / cell), 1)

    def grid(m):
        g = np.zeros((nu, nz), bool)
        g[np.clip((s[m] / cell).astype(int), 0, nu - 1), np.clip(((p[m, 2] - floor) / cell).astype(int), 0, nz - 1)] = True
        return g

    on = grid(inside & (np.abs(off) < tol))
    behind = grid(inside & (off < -tol) & (off > -3.0))
    front = grid(inside & (off > tol) & (off < 1.5))
    return on, behind, front, u, L


def detect_openings(p, a, b, nrm, floor, ceil, cell=0.05, mode="empty"):
    """mode: "empty" = holes in the wall plane; "seethrough" = holes with scan points visible behind them;
    "both" = union of the two."""
    if mode == "both":
        out = detect_openings(p, a, b, nrm, floor, ceil, cell, "empty")
        extra = detect_openings(p, a, b, nrm, floor, ceil, cell, "seethrough")
        return out + [o for o in extra if not any(abs(o["x"] - q["x"]) + abs(o["y"] - q["y"]) < 0.3 for q in out)]
    on, behind, front, u, L = opening_evidence(p, a, b, nrm, floor, ceil, cell)
    on = ndimage.binary_closing(on, iterations=1)
    if mode == "seethrough":
        mask = ndimage.binary_closing(behind & ~on, structure=np.ones((3, 3)), iterations=2) & ~on
        mask = ndimage.binary_opening(mask, structure=np.ones((3, 3)))
    else:
        mask = ndimage.binary_opening(~on, structure=np.ones((5, 5)))
    lab, k = ndimage.label(mask)
    out = []
    H = ceil - floor
    for j in range(1, k + 1):
        cells = lab == j
        iu, iz = np.where(cells)
        u0, u1 = iu.min() * cell, (iu.max() + 1) * cell
        z0, z1 = iz.min() * cell, (iz.max() + 1) * cell
        w, h = u1 - u0, z1 - z0
        fill = cells.sum() / ((iu.max() - iu.min() + 1) * (iz.max() - iz.min() + 1))
        behind_frac = behind[cells].mean()
        front_frac = front[cells].mean()
        if fill < (0.45 if mode == "seethrough" else 0.6) or (behind_frac < 0.15 and front_frac > 0.3):
            continue  # ragged hole, or a gap explained by something standing in front of the wall
        if u0 < 0.05 or u1 > L - 0.05:
            continue  # holes touching the wall ends are corners/missing data, not openings
        kind = None
        if z0 < (0.4 if mode == "seethrough" else 0.15) and 1.5 < h < min(2.6, H) + 0.01 and 0.6 < w < 2.0:
            kind = "door"
        elif z0 > 0.3 and z1 < H - 0.05 and 0.4 < w < 3.0 and 0.4 < h < 2.2:
            kind = "window"
        if kind:
            cu = (u0 + u1) / 2
            center = a + u * cu
            out.append(dict(kind=kind, x=float(center[0]), y=float(center[1]), z=float(floor + (z0 + z1) / 2),
                            width=float(w), height=float(h), behind=float(behind_frac), front=float(front_frac)))
    return out


def run(point_cloud, detect_openings_flag=True, floor=None, opening_mode="both", **poly_kw):
    p, n = load_points(point_cloud)
    floor, ceil = floor_and_ceiling(p, n, floor=floor)
    cands = wall_candidates(p, n, floor, ceil, 0) + wall_candidates(p, n, floor, ceil, 1)
    if poly_kw.pop("auto", False):
        # sparse scans need a low fill threshold, dense ones with see-through views a high one:
        # keep the footprint whose walls are best backed by scan points (length-weighted)
        best = None
        for thr, cell in ((0.15, 0.1), (0.3, 0.1), (0.5, 0.1), (0.5, 0.05)):
            cand_poly = room_polygon(p, cands, floor, ceil, fill_thr=thr, cell=cell)
            segs = polygon_walls(cand_poly)
            lens = np.array([np.linalg.norm(b - a) for a, b in segs])
            sup = np.array([wall_support(p, a, b, floor, ceil) for a, b in segs])
            score = float((lens * sup).sum() / lens.sum())
            if best is None or score > best[0] + 0.02:
                best = (score, cand_poly)
        poly = best[1]
    else:
        poly = room_polygon(p, cands, floor, ceil, **poly_kw)
    walls = []
    for a, b in polygon_walls(poly):
        u, nrm, L = wall_frame(a, b, poly)
        w = dict(a=a, b=b, nrm=nrm, support=wall_support(p, a, b, floor, ceil), openings=[])
        if detect_openings_flag:
            w["openings"] = detect_openings(p, a, b, nrm, floor, ceil, mode=opening_mode)
        walls.append(w)
    return dict(points=p, normals=n, floor=floor, ceil=ceil, candidates=cands, polygon=poly, walls=walls)


def to_layout_string(walls, floor, ceil, bboxes_str=""):
    lines, di, wi = [], 0, 0
    for k, w in enumerate(walls):
        a, b = w["a"], w["b"]
        lines.append(f"wall_{k}=Wall({a[0]},{a[1]},{floor},{b[0]},{b[1]},{floor},{ceil - floor},0.0)")
    for k, w in enumerate(walls):
        for o in w["openings"]:
            if o["kind"] == "door":
                lines.append(f"door_{di}=Door(wall_{k},{o['x']},{o['y']},{o['z']},{o['width']},{o['height']})")
                di += 1
            else:
                lines.append(f"window_{wi}=Window(wall_{k},{o['x']},{o['y']},{o['z']},{o['width']},{o['height']})")
                wi += 1
    s = "\n".join(lines)
    if bboxes_str:
        s += "\n" + bboxes_str
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--point_cloud", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--no_openings", action="store_true")
    ap.add_argument("--floor", type=float, help="known floor height (aligned clouds: 0); default: detect")
    ap.add_argument("--opening_mode", default="both", choices=["empty", "seethrough", "both"])
    ap.add_argument("--auto_footprint", action="store_true", help="pick footprint params by wall point support")
    args = ap.parse_args()
    r = run(args.point_cloud, not args.no_openings, args.floor, args.opening_mode, auto=args.auto_footprint)
    with open(args.output, "w") as f:
        f.write(to_layout_string(r["walls"], r["floor"], r["ceil"]))
    print(f"floor={r['floor']:.2f} ceil={r['ceil']:.2f} candidates={len(r['candidates'])} walls={len(r['walls'])} "
          f"openings={sum(len(w['openings']) for w in r['walls'])}")
    with open(args.output.rsplit(".", 1)[0] + "_debug.json", "w") as f:
        json.dump(dict(floor=r["floor"], ceil=r["ceil"], candidates=r["candidates"],
                       walls=[dict(a=w["a"].tolist(), b=w["b"].tolist(), support=w["support"], openings=w["openings"])
                              for w in r["walls"]]), f, indent=1)
