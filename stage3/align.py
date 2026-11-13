"""Level, Manhattan-align and metric-scale a MASt3R-SLAM point cloud for stage 3.

Output follows the SpatialLM convention: z up, walls roughly parallel to x/y, floor at z=0,
1 unit = 1 m (scale set from floor-to-ceiling height, since MASt3R-SLAM scale is arbitrary).

    python stage3/align.py --input scan.ply --output outputs/aligned.ply --ceiling_height 2.6
"""

import argparse
import json
import os

import numpy as np
import open3d as o3d


def dominant_direction(normals, candidates, cos_thr):
    """Axis (sign-free) that the most normals point along."""
    counts = (np.abs(normals @ candidates.T) > cos_thr).sum(0)
    d = candidates[np.argmax(counts)]
    for _ in range(5):  # refine: principal axis of the inlier normals
        inl = normals[np.abs(normals @ d) > cos_thr]
        _, _, vt = np.linalg.svd(inl, full_matrices=False)
        d = vt[0]
    return d / np.linalg.norm(d)


def manhattan_frame(normals, angle_deg=10.0, seed=0):
    rng = np.random.default_rng(seed)
    cos_thr = np.cos(np.deg2rad(angle_deg))
    sin_thr = np.sin(np.deg2rad(angle_deg))
    sub = normals[rng.choice(len(normals), min(len(normals), 60000), replace=False)]
    cand = sub[rng.choice(len(sub), 3000, replace=False)]
    d1 = dominant_direction(sub, cand, cos_thr)
    ortho = sub[np.abs(sub @ d1) < sin_thr]
    cand2 = ortho[rng.choice(len(ortho), min(3000, len(ortho)), replace=False)]
    cand2 = cand2 - np.outer(cand2 @ d1, d1)
    cand2 /= np.linalg.norm(cand2, axis=1, keepdims=True)
    d2 = dominant_direction(ortho, cand2, cos_thr)
    d2 = d2 - (d2 @ d1) * d1
    d2 /= np.linalg.norm(d2)
    d3 = np.cross(d1, d2)
    return np.stack([d1, d2, d3])  # rows are the 3 axes


def histogram_peaks(z, bin_size, min_frac=0.02):
    lo, hi = np.percentile(z, 0.5), np.percentile(z, 99.5)
    bins = np.arange(lo, hi + bin_size, bin_size)
    h, e = np.histogram(z, bins)
    centers = 0.5 * (e[1:] + e[:-1])
    peaks = [i for i in range(1, len(h) - 1) if h[i] >= h[i - 1] and h[i] >= h[i + 1] and h[i] > min_frac * len(z)]
    return centers, h, peaks


def auto_up(points, normals, axes):
    """Guess the up axis without a hint: a single room is lower than it is wide, so take the Manhattan axis with the
    smallest extent; furniture tops (tables, beds, seats) sit in the lower half, which gives the sign."""
    extents = [np.percentile(points @ a, 99) - np.percentile(points @ a, 1) for a in axes]
    up_idx = int(np.argmin(extents))
    a = axes[up_idx]
    h = points @ a
    lo, hi = np.percentile(h, 1), np.percentile(h, 99)
    t = (h[np.abs(normals @ a) > 0.9] - lo) / (hi - lo)
    mid = t[(t > 0.15) & (t < 0.85)]
    sign = 1.0 if len(mid) == 0 or (mid < 0.5).mean() >= 0.5 else -1.0
    return up_idx, a * sign


def align(input_path, output_path, ceiling_height=2.6, no_scale=False, up_hint="auto", metric=False):
    """Write the aligned cloud to output_path and its transform to <output stem>_transform.json; return the transform."""
    pcd = o3d.io.read_point_cloud(input_path)
    pts = np.asarray(pcd.points)
    extent = np.percentile(pts, 99, 0) - np.percentile(pts, 1, 0)
    voxel = float(extent.max()) / 400.0  # ~1/400 of the scene, scale-free
    pcd = pcd.voxel_down_sample(voxel)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=30))
    normals = np.asarray(pcd.normals)

    axes = manhattan_frame(normals)
    if up_hint == "auto":
        up_idx, up = auto_up(np.asarray(pcd.points), normals, axes)
    else:
        hint = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}[up_hint[-1]]
        hint = np.array(hint, float) * (-1 if up_hint.startswith("-") else 1)
        up_idx = int(np.argmax(np.abs(axes @ hint)))
        up = axes[up_idx] * np.sign(axes[up_idx] @ hint)
    horiz = [axes[i] for i in range(3) if i != up_idx]
    x_axis = horiz[0]
    y_axis = np.cross(up, x_axis)
    rot = np.stack([x_axis, y_axis, up])  # world->aligned

    p = np.asarray(pcd.points) @ rot.T
    n = np.asarray(pcd.normals) @ rot.T
    horizontal_surf = np.abs(n[:, 2]) > 0.9
    centers, h, peaks = histogram_peaks(p[horizontal_surf, 2], voxel)
    # MASt3R floors/ceilings come out as several stacked layers: take the densest layer in each half
    mid = np.median(p[:, 2])
    low = [i for i in peaks if centers[i] < mid]
    high = [i for i in peaks if centers[i] > mid]
    floor_z = centers[max(low, key=lambda i: h[i])] if low else np.percentile(p[:, 2], 1)
    ceil_z = centers[max(high, key=lambda i: h[i])] if high else np.percentile(p[:, 2], 99)
    if metric:
        # metric clouds (RTAB-Map): floor = densest layer of ALL points in the lowest quarter of the height range,
        # so furniture tops (beds, tables) are never mistaken for the floor
        z = p[:, 2]
        lo, hi = np.percentile(z, 1), np.percentile(z, 99)
        zz = z[(z >= lo) & (z <= lo + 0.25 * (hi - lo))]
        hh, ee = np.histogram(zz, np.arange(zz.min(), zz.max() + voxel, voxel))
        floor_z = 0.5 * (ee[np.argmax(hh)] + ee[np.argmax(hh) + 1])
        no_scale = True
    raw_height = ceil_z - floor_z
    scale = 1.0 if no_scale else ceiling_height / raw_height
    print(f"up axis (input frame) = {up.round(3)}  floor={floor_z:.3f} ceiling={ceil_z:.3f} raw height={raw_height:.3f}")
    print(f"scale factor = {scale:.4f}")

    # apply to the full-resolution cloud, then thin to 1 cm to keep files small
    full = o3d.io.read_point_cloud(input_path)
    fp = (np.asarray(full.points) @ rot.T - np.array([0, 0, floor_z])) * scale
    full.points = o3d.utility.Vector3dVector(fp)
    full = full.voxel_down_sample(0.01)
    full, _ = full.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    o3d.io.write_point_cloud(output_path, full)
    q = np.asarray(full.points)
    print(f"wrote {output_path}: {len(q)} pts, extent min {q.min(0).round(2)} max {q.max(0).round(2)}")

    transform = {"rotation": rot.tolist(), "floor_z_raw": float(floor_z), "scale": float(scale),
                 "raw_height": float(raw_height), "note": "aligned = (R @ p - [0,0,floor_z_raw]) * scale"}
    with open(output_path.rsplit(".", 1)[0] + "_transform.json", "w") as f:
        json.dump(transform, f, indent=2)
    return transform


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--ceiling_height", type=float, default=2.6, help="metres, used to fix the scale")
    ap.add_argument("--no_scale", action="store_true", help="keep the input scale")
    ap.add_argument("--up_hint", default="auto",
                    help="input axis that roughly points up (x, -y, ...), or auto: smallest room extent (default)")
    args = ap.parse_args()
    align(args.input, args.output, args.ceiling_height, args.no_scale, args.up_hint)


if __name__ == "__main__":
    main()
