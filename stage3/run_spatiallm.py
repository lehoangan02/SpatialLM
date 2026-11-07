"""Run SpatialLM over several point clouds and settings with one model load, logging time and VRAM.

    python stage3/run_spatiallm.py --model_path models/SpatialLM1.1-Qwen-0.5B --inputs a.ply b.ply \
        --out_dir outputs/sl --detect_type all --seeds 0 --dtype float16
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from inference import generate_layout, preprocess_point_cloud  # noqa: E402
from spatiallm import Layout  # noqa: E402
from spatiallm.pcd import cleanup_pcd, get_points_and_colors, load_o3d_pcd  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--detect_type", nargs="+", default=["all"], choices=["all", "arch", "object"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--num_beams", type=int, default=1)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=getattr(torch, args.dtype))
    model.to("cuda")
    model.set_point_backbone_dtype(torch.float32)
    model.eval()
    load_s = time.time() - t0
    num_bins = model.config.point_config["num_bins"]
    grid_size = Layout.get_grid_size(num_bins)
    template = os.path.join(REPO, "code_template.txt")
    os.makedirs(args.out_dir, exist_ok=True)
    log = []

    for path in args.inputs:
        pcd = cleanup_pcd(load_o3d_pcd(path), voxel_size=grid_size)
        points, colors = get_points_and_colors(pcd)
        min_extent = np.min(points, axis=0)
        input_pcd = preprocess_point_cloud(points, colors, grid_size, num_bins)
        name = os.path.splitext(os.path.basename(path))[0]
        for dt in args.detect_type:
            for seed in args.seeds:
                torch.cuda.reset_peak_memory_stats()
                t1 = time.time()
                err = None
                try:
                    with torch.no_grad():
                        layout = generate_layout(model, input_pcd, tokenizer, template, top_k=args.top_k,
                                                 top_p=args.top_p, temperature=args.temperature,
                                                 num_beams=args.num_beams, seed=seed, detect_type=dt)
                    layout.translate(min_extent)
                    text = layout.to_language_string()
                except Exception as e:  # keep going so one OOM doesn't kill the sweep
                    err, text, layout = repr(e), "", None
                dt_s = time.time() - t1
                sub = os.path.join(args.out_dir, f"{dt}_s{seed}")
                os.makedirs(sub, exist_ok=True)
                with open(os.path.join(sub, f"{name}.txt"), "w") as f:
                    f.write(text)
                rec = dict(scene=name, detect_type=dt, seed=seed, seconds=round(dt_s, 1),
                           peak_vram_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
                           n_points=int(len(points)), error=err,
                           walls=len(layout.walls) if layout else 0, doors=len(layout.doors) if layout else 0,
                           windows=len(layout.windows) if layout else 0, boxes=len(layout.bboxes) if layout else 0)
                print("RESULT", json.dumps(rec), flush=True)
                log.append(rec)
                torch.cuda.empty_cache()

    with open(os.path.join(args.out_dir, f"runlog{args.tag}.json"), "w") as f:
        json.dump(dict(model=args.model_path, dtype=args.dtype, load_seconds=round(load_s, 1), runs=log), f, indent=1)


if __name__ == "__main__":
    main()
