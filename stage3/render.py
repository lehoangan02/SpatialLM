"""Top-down comparison image of several layouts over one point cloud, plus geometry-only proxy scores.

Proxy scores (no ground truth needed):
  wall_support   mean fraction of each predicted wall's area that has scan points within 7 cm
  wall_recall    fraction of observed wall-surface points within 10 cm of some predicted wall
  door/window    count, and fraction with a visible gap or see-through behind them in the scan

    python stage3/render.py --point_cloud aligned.ply --layouts name1=a.txt name2=b.txt --out cmp.png
"""

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import planefit  # noqa: E402
from spatiallm import Layout  # noqa: E402


def proxy_scores(p, n, floor, ceil, layout):
    walls = {w.id: w for w in layout.walls}
    sup = [planefit.wall_support(p, np.array([w.ax, w.ay]), np.array([w.bx, w.by]), floor, ceil)
           for w in layout.walls if np.hypot(w.bx - w.ax, w.by - w.ay) > 1e-3]
    wall_pts = p[(np.abs(n[:, 2]) < 0.3) & (p[:, 2] > floor + 0.3) & (p[:, 2] < ceil - 0.3)]
    near = np.zeros(len(wall_pts), bool)
    for w in layout.walls:
        a, b = np.array([w.ax, w.ay]), np.array([w.bx, w.by])
        L = np.linalg.norm(b - a)
        if L < 1e-3:
            continue
        u = (b - a) / L
        rel = wall_pts[:, :2] - a
        s = rel @ u
        near |= (np.abs(rel @ np.array([-u[1], u[0]])) < 0.10) & (s > -0.1) & (s < L + 0.1)
    gaps = []
    for o in layout.doors + layout.windows:
        w = walls.get(o.wall_id)
        if w is None:
            continue
        a, b = np.array([w.ax, w.ay]), np.array([w.bx, w.by])
        if np.linalg.norm(b - a) < 1e-3:
            continue
        u = (b - a) / np.linalg.norm(b - a)
        on, behind, _, _, L = planefit.opening_evidence(p, a, b, np.array([-u[1], u[0]]), floor, ceil)
        s = (np.array([o.position_x, o.position_y]) - a) @ u
        i0, i1 = int(max(s - o.width / 2, 0) / 0.05), int(min(s + o.width / 2, L) / 0.05)
        z0 = o.position_z - o.height / 2 - floor
        j0, j1 = int(max(z0, 0) / 0.05), int(min(z0 + o.height, ceil - floor) / 0.05)
        if i1 > i0 and j1 > j0:
            gaps.append(1 - on[i0:i1, j0:j1].mean() > 0.35 or behind[i0:i1, j0:j1].mean() > 0.2)
    return dict(walls=len(layout.walls), doors=len(layout.doors), windows=len(layout.windows),
                boxes=len(layout.bboxes),
                wall_support=float(np.mean(sup)) if sup else 0.0,
                wall_recall=float(near.mean()) if len(near) else 0.0,
                openings_with_gap=f"{sum(gaps)}/{len(gaps)}")


def draw(ax, p, layout, title):
    sub = p[np.random.default_rng(0).choice(len(p), min(len(p), 150000), replace=False)]
    ax.scatter(sub[:, 0], sub[:, 1], s=0.05, c="0.6", rasterized=True)
    wl = {w.id: w for w in layout.walls}
    for w in layout.walls:
        ax.plot([w.ax, w.bx], [w.ay, w.by], "-", color="k", lw=2)
    for o, col in [(d, "tab:green") for d in layout.doors] + [(x, "tab:blue") for x in layout.windows]:
        w = wl.get(o.wall_id)
        if w is None:
            continue
        u = np.array([w.bx - w.ax, w.by - w.ay])
        u = u / max(np.linalg.norm(u), 1e-6)
        c = np.array([o.position_x, o.position_y])
        e0, e1 = c - u * o.width / 2, c + u * o.width / 2
        ax.plot([e0[0], e1[0]], [e0[1], e1[1]], "-", color=col, lw=5, solid_capstyle="butt")
    for b in layout.bboxes:
        c, s = np.array([b.position_x, b.position_y]), np.array([b.scale_x, b.scale_y]) / 2
        R = np.array([[np.cos(b.angle_z), -np.sin(b.angle_z)], [np.sin(b.angle_z), np.cos(b.angle_z)]])
        corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]]) * s @ R.T + c
        ax.plot(corners[:, 0], corners[:, 1], "-", color="tab:orange", lw=0.8)
        ax.text(c[0], c[1], b.class_name, fontsize=5, color="tab:orange", ha="center")
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--point_cloud", required=True)
    ap.add_argument("--layouts", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--floor", type=float, help="known floor height (aligned clouds: 0); default: detect")
    args = ap.parse_args()
    p, n = planefit.load_points(args.point_cloud)
    floor, ceil = planefit.floor_and_ceiling(p, n, floor=args.floor)
    items = [x.split("=", 1) for x in args.layouts]
    rows = int(np.ceil(len(items) / args.cols))
    fig, axes = plt.subplots(rows, args.cols, figsize=(4.5 * args.cols, 4.5 * rows))
    axes = np.atleast_1d(axes).ravel()
    scores = {}
    for ax, (name, path) in zip(axes, items):
        layout = Layout(open(path).read()) if os.path.exists(path) else Layout()
        scores[name] = proxy_scores(p, n, floor, ceil, layout)
        s = scores[name]
        draw(ax, p, layout, f"{name}\nW{s['walls']} D{s['doors']} Win{s['windows']} B{s['boxes']}  "
                            f"sup {s['wall_support']:.2f} rec {s['wall_recall']:.2f}")
    for ax in axes[len(items):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    with open(args.out.rsplit(".", 1)[0] + "_scores.json", "w") as f:
        json.dump(dict(floor=floor, ceil=ceil, scores=scores), f, indent=1)
    for k, v in scores.items():
        print(f"{k:24s} {v}")


if __name__ == "__main__":
    main()
