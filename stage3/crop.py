"""Geometry-first preprocessing for SpatialLM: keep only points inside the plane-fit room footprint
(plus a margin) and between floor and ceiling, dropping what was seen through doors/windows.

    python stage3/crop.py --point_cloud in.ply --output out.ply [--margin 0.3] [--auto_footprint]
"""

import argparse
import os
import sys

import numpy as np
import open3d as o3d
from shapely import contains_xy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import planefit  # noqa: E402


def crop(input_path, output_path, margin=0.3, floor=0.0, auto_footprint=False):
    """Keep points inside the plane-fit footprint (+margin) and between floor and ceiling; return (kept, total)."""
    fit = planefit.run(input_path, False, floor, auto=auto_footprint)
    region = fit["polygon"].buffer(margin)
    pcd = o3d.io.read_point_cloud(input_path)
    p = np.asarray(pcd.points)
    keep = contains_xy(region, p[:, 0], p[:, 1]) & (p[:, 2] > fit["floor"] - 0.15) & (p[:, 2] < fit["ceil"] + 0.15)
    out = pcd.select_by_index(np.where(keep)[0])
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    o3d.io.write_point_cloud(output_path, out)
    return int(keep.sum()), len(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--point_cloud", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--margin", type=float, default=0.3)
    ap.add_argument("--floor", type=float, default=0.0)
    ap.add_argument("--auto_footprint", action="store_true")
    args = ap.parse_args()
    kept, total = crop(args.point_cloud, args.output, args.margin, args.floor, args.auto_footprint)
    print(f"kept {kept}/{total} points")

if __name__ == "__main__":
    main()
