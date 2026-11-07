"""Level, Manhattan-align and metric-scale a MASt3R-SLAM point cloud for stage 3.

Output follows the SpatialLM convention: z up, walls roughly parallel to x/y, floor at z=0,
1 unit = 1 m (scale set from floor-to-ceiling height, since MASt3R-SLAM scale is arbitrary).

    python stage3/align.py --input scan.ply --output outputs/aligned.ply --ceiling_height 2.6
"""

import argparse
import json

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--ceiling_height", type=float, default=2.6, help="metres, used to fix the scale")
    ap.add_argument("--no_scale", action="store_true", help="keep the input scale")
    ap.add_argument("--up_hint", default="-y", help="input axis that roughly points up (MASt3R-SLAM camera frame: -y)")
    args = ap.parse_args()

    pcd = o3d.io.read_point_cloud(args.input)
    pts = np.asarray(pcd.points)
    extent = np.percentile(pts, 99, 0) - np.percentile(pts, 1, 0)
    voxel = float(extent.max()) / 400.0  # ~1/400 of the scene, scale-free
    pcd = pcd.voxel_down_sample(voxel)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=30))
    normals = np.asarray(pcd.normals)

    axes = manhattan_frame(normals)
    hint = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}[args.up_hint[-1]]
    hint = np.array(hint, float) * (-1 if args.up_hint.startswith("-") else 1)
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
    raw_height = ceil_z - floor_z
    scale = 1.0 if args.no_scale else args.ceiling_height / raw_height
    print(f"up axis (input frame) = {up.round(3)}  floor={floor_z:.3f} ceiling={ceil_z:.3f} raw height={raw_height:.3f}")
    print(f"scale factor = {scale:.4f}")

    # apply to the full-resolution cloud, then thin to 1 cm to keep files small
    full = o3d.io.read_point_cloud(args.input)
    fp = (np.asarray(full.points) @ rot.T - np.array([0, 0, floor_z])) * scale
    full.points = o3d.utility.Vector3dVector(fp)
    full = full.voxel_down_sample(0.01)
    full, _ = full.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    o3d.io.write_point_cloud(args.output, full)
    q = np.asarray(full.points)
    print(f"wrote {args.output}: {len(q)} pts, extent min {q.min(0).round(2)} max {q.max(0).round(2)}")

    with open(args.output.rsplit(".", 1)[0] + "_transform.json", "w") as f:
        json.dump({"rotation": rot.tolist(), "floor_z_raw": float(floor_z), "scale": float(scale),
                   "raw_height": float(raw_height), "note": "aligned = (R @ p - [0,0,floor_z_raw]) * scale"}, f, indent=2)


if __name__ == "__main__":
    main()
