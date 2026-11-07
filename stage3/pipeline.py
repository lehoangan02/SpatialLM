"""Whole stage-3 pipeline in one command: raw room scan -> layout (txt + glTF). See CONFIG.md.

    python stage3/pipeline.py SCAN.ply                          # defaults: Llama-1B, seeds 0 1 2, 2.6 m ceiling
    python stage3/pipeline.py SCAN.ply --model qwen --seeds 0 --ceiling_height 2.4 --out_dir outputs/kitchen

Steps: align -> crop to the room footprint -> thin to the point budget -> SpatialLM `arch` (one run per seed) and `object` on the
cropped cloud -> snap each seed onto plane-fit geometry (hyb_snap) -> keep the seed with the highest wall
support -> export .glb. If a Llama-1B run fails (usually out of memory on 4 GB), it is retried with Qwen-0.5B.

Output (<out_dir>, default outputs/<scan name>/):
    layout.txt               final layout of the chosen seed (SpatialLM text format, metres, z up, floor at z=0)
    room.glb                 3D model of the layout
    room_with_scan.glb       3D model of the layout plus scan points
    seeds.png                top-down comparison of every seed's final layout
    summary.json             settings, chosen seed, per-seed scores, run times and errors
    seeds/s<k>.txt           final layout of every seed
    intermediate/
        aligned.ply                  scan in the layout frame
        aligned_transform.json       raw -> aligned: aligned = (R @ p - [0,0,floor_z_raw]) * scale
        cropped.ply                  aligned scan cropped to the room footprint
        spatiallm/arch_s<k>.txt      raw SpatialLM walls/doors/windows per seed
        spatiallm/object_s<k>.txt    raw SpatialLM furniture boxes (before removing repeats / outside boxes)
"""

import argparse
import contextlib
import gc
import json
import os
import shutil
import sys
import time

import numpy as np

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # less fragmentation on 4 GB
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import align as align_mod  # noqa: E402
import crop as crop_mod  # noqa: E402
import export_3d  # noqa: E402
import hybrid  # noqa: E402
import planefit  # noqa: E402
import render  # noqa: E402
from spatiallm import Layout  # noqa: E402

MODELS = {"llama": "models/SpatialLM1.1-Llama-1B", "qwen": "models/SpatialLM1.1-Qwen-0.5B"}


def log(msg):
    print(f"[pipeline] {msg}", flush=True)


class SpatialLM:
    """One loaded SpatialLM model; caches the preprocessed input clouds."""

    def __init__(self, name, dtype, max_points):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name, self.torch = name, torch
        path = os.path.join(REPO, MODELS[name])
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=getattr(torch, dtype))
        self.model.to("cuda")
        self.model.set_point_backbone_dtype(torch.float32)
        self.model.eval()
        self.num_bins = self.model.config.point_config["num_bins"]
        self.grid_size = Layout.get_grid_size(self.num_bins)
        self.max_points = max_points
        self.cache = {}
        log(f"loaded {MODELS[name]} ({dtype}) in {time.time() - t0:.1f} s")

    def _input(self, cloud):
        if cloud not in self.cache:
            from inference import preprocess_point_cloud
            from spatiallm.pcd import cleanup_pcd, get_points_and_colors, load_o3d_pcd

            raw = load_o3d_pcd(cloud)
            pcd, voxel = cleanup_pcd(raw, voxel_size=self.grid_size), self.grid_size
            # memory grows with the point count: thin dense scans until they fit the budget
            while len(pcd.points) > self.max_points:
                voxel += 0.005
                pcd = cleanup_pcd(raw.voxel_down_sample(voxel), voxel_size=self.grid_size)
            if voxel > self.grid_size:
                log(f"thinned {os.path.basename(cloud)} with a {voxel * 100:.1f} cm voxel to {len(pcd.points)} points")
            points, colors = get_points_and_colors(pcd)
            self.cache[cloud] = (preprocess_point_cloud(points, colors, self.grid_size, self.num_bins),
                                 np.min(points, axis=0), len(points), round(voxel, 3))
        return self.cache[cloud]

    def run(self, cloud, task, seed):
        """Return a run record; record['text'] is the layout text ('' on failure)."""
        from inference import generate_layout

        torch = self.torch
        inp, min_extent, n_points, voxel = self._input(cloud)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        text, err = "", None
        try:
            with torch.no_grad(), contextlib.redirect_stdout(open(os.devnull, "w")):  # it prints every token
                layout = generate_layout(self.model, inp, self.tokenizer, os.path.join(REPO, "code_template.txt"),
                                         seed=seed, detect_type=task)
            layout.translate(min_extent)
            text = layout.to_language_string()
            if not text.strip():
                err = "empty output"
        except Exception as e:  # out of memory surfaces here (often as Empty() from the generation thread)
            err = repr(e)
        rec = dict(task=task, seed=seed, model=self.name, n_points=n_points, voxel_m=voxel, seconds=round(time.time() - t0, 1),
                   peak_vram_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2), error=err, text=text)
        torch.cuda.empty_cache()
        log(f"{task} seed {seed} [{self.name}]: " + (f"FAILED ({err})" if err else f"ok in {rec['seconds']} s"))
        return rec

    def close(self):
        del self.model
        gc.collect()
        self.torch.cuda.empty_cache()


def run_spatiallm(jobs, args):
    """jobs: list of (cloud, task, seed). Runs with the chosen model, retries failures with Qwen if allowed."""
    results = {}
    lm = SpatialLM(args.model, args.dtype, args.max_input_points)
    for job in jobs:
        results[job] = lm.run(*job)
    failed = [j for j, r in results.items() if r["error"]]
    if failed and args.model != "qwen" and not args.no_fallback:
        log(f"{len(failed)} run(s) failed with {args.model}; retrying with qwen")
        lm.close()
        lm = SpatialLM("qwen", "float16", args.max_input_points)
        for job in failed:
            results[job] = lm.run(*job)
    lm.close()
    return results


def clean_boxes(boxes, region):
    """Drop exact repeats (SpatialLM sometimes loops on one object) and boxes centred outside the room."""
    from shapely.geometry import Point

    seen, keep = set(), []
    for b in boxes:
        key = (b.class_name,) + tuple(round(v, 2) for v in (b.position_x, b.position_y, b.position_z, b.angle_z,
                                                            b.scale_x, b.scale_y, b.scale_z))
        if key in seen or not region.contains(Point(b.position_x, b.position_y)):
            continue
        seen.add(key)
        keep.append(b)
    for i, b in enumerate(keep):
        b.id = i
    if len(keep) < len(boxes):
        log(f"boxes: kept {len(keep)} of {len(boxes)} (dropped repeats and boxes outside the room)")
    return keep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scan", help="input point cloud (.ply), e.g. MASt3R-SLAM output")
    ap.add_argument("--out_dir", help="output folder (default: outputs/<scan name>)")
    ap.add_argument("--model", choices=sorted(MODELS), default="llama", help="SpatialLM model (default: llama)")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="seeds for the arch pass (default: 0 1 2)")
    ap.add_argument("--object_seed", type=int, default=0, help="seed for the furniture pass (default: 0)")
    ap.add_argument("--object_input", choices=["cropped", "full"], default="cropped",
                    help="cloud for the furniture pass (default: cropped; the full cloud can make SpatialLM loop on "
                         "objects seen through doors/windows)")
    ap.add_argument("--max_input_points", type=int, default=120_000,
                    help="thin SpatialLM inputs to at most this many points (default 120000, fits Llama-1B on 4 GB)")
    ap.add_argument("--no_fallback", action="store_true", help="do not retry failed Llama runs with Qwen")
    ap.add_argument("--ceiling_height", type=float, default=2.6, help="real ceiling height in metres (sets the scale)")
    ap.add_argument("--up_hint", default="auto",
                    help="input axis that roughly points up (x, -y, z, ...); auto = smallest room extent (default)")
    ap.add_argument("--aligned", action="store_true",
                    help="input is already aligned (z up, floor at z=0, metres): skip alignment")
    ap.add_argument("--no_crop", action="store_true", help="run the arch pass on the full cloud")
    ap.add_argument("--margin", type=float, default=0.3, help="crop margin around the footprint in metres")
    ap.add_argument("--footprint", choices=["auto", "sparse"], default="auto", help="plane-fit footprint mode")
    ap.add_argument("--glb_points", type=int, default=300_000, help="scan points in room_with_scan.glb")
    args = ap.parse_args()

    t_start = time.time()
    name = os.path.splitext(os.path.basename(args.scan))[0]
    out = os.path.abspath(args.out_dir or os.path.join(REPO, "outputs", name))
    inter = os.path.join(out, "intermediate")
    sl_dir = os.path.join(inter, "spatiallm")
    for d in (os.path.join(out, "seeds"), sl_dir):
        os.makedirs(d, exist_ok=True)
    log(f"output folder: {out}")

    # 1. align
    aligned = os.path.join(inter, "aligned.ply")
    if args.aligned:
        shutil.copyfile(args.scan, aligned)
        transform = None
        log("input already aligned: copied")
    else:
        transform = align_mod.align(args.scan, aligned, args.ceiling_height, up_hint=args.up_hint)

    # 2. crop
    arch_input = aligned
    if not args.no_crop:
        arch_input = os.path.join(inter, "cropped.ply")
        kept, total = crop_mod.crop(aligned, arch_input, args.margin, 0.0)
        log(f"cropped: kept {kept}/{total} points")

    # 3. SpatialLM
    object_input = arch_input if args.object_input == "cropped" else aligned
    jobs = [(arch_input, "arch", s) for s in args.seeds] + [(object_input, "object", args.object_seed)]
    runs = run_spatiallm(jobs, args)
    for (_, task, seed), r in runs.items():
        with open(os.path.join(sl_dir, f"{task}_s{seed}.txt"), "w") as f:
            f.write(r["text"])
    fit = planefit.run(aligned, floor=0.0, auto=args.footprint == "auto")
    boxes_run = runs[(object_input, "object", args.object_seed)]
    boxes = clean_boxes(Layout(boxes_run["text"]).bboxes, fit["polygon"].buffer(args.margin)) if not boxes_run["error"] else []
    if boxes_run["error"]:
        log("WARNING: furniture pass failed; layouts will have no boxes")

    # 4. snap every successful seed onto the plane fit, score it
    p, n = planefit.load_points(aligned)
    floor, ceil = planefit.floor_and_ceiling(p, n, floor=0.0)
    seeds = {}
    for s in args.seeds:
        r = runs[(arch_input, "arch", s)]
        if r["error"]:
            continue
        sl = Layout(r["text"])
        sl.bboxes = boxes
        text = hybrid.snap_variant(fit, sl)
        path = os.path.join(out, "seeds", f"s{s}.txt")
        with open(path, "w") as f:
            f.write(text)
        seeds[s] = dict(path=path, scores=render.proxy_scores(p, n, floor, ceil, Layout(text)))
    if not seeds:
        sys.exit("[pipeline] every arch run failed; nothing to output (see intermediate/spatiallm and the log above)")

    # 5. pick the seed whose walls are best backed by scan points
    best = max(seeds, key=lambda s: seeds[s]["scores"]["wall_support"])
    shutil.copyfile(seeds[best]["path"], os.path.join(out, "layout.txt"))
    log(f"chosen seed: {best} (wall support {seeds[best]['scores']['wall_support']:.2f})")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(seeds), figsize=(4.5 * len(seeds), 4.5), squeeze=False)
    for ax, (s, d) in zip(axes[0], seeds.items()):
        sc = d["scores"]
        render.draw(ax, p, Layout(open(d["path"]).read()),
                    f"seed {s}{'  (chosen)' if s == best else ''}\nW{sc['walls']} D{sc['doors']} Win{sc['windows']} "
                    f"B{sc['boxes']}  sup {sc['wall_support']:.2f} rec {sc['wall_recall']:.2f}")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "seeds.png"), dpi=130)
    plt.close(fig)

    # 6. 3D export
    layout = export_3d.export(os.path.join(out, "layout.txt"), os.path.join(out, "room.glb"))
    export_3d.export(os.path.join(out, "layout.txt"), os.path.join(out, "room_with_scan.glb"), aligned, args.glb_points)

    summary = dict(
        input=os.path.abspath(args.scan),
        settings={k: v for k, v in vars(args).items() if k not in ("scan", "out_dir")},
        transform=transform,
        chosen_seed=best,
        result=dict(walls=len(layout.walls), doors=len(layout.doors), windows=len(layout.windows),
                    boxes=len(layout.bboxes)),
        seeds={f"s{s}": d["scores"] for s, d in seeds.items()},
        spatiallm_runs=[{k: v for k, v in r.items() if k != "text"} for r in runs.values()],
        total_seconds=round(time.time() - t_start, 1),
    )
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    log(f"done in {summary['total_seconds']} s: {summary['result']} -> {out}/layout.txt, room.glb")


if __name__ == "__main__":
    main()
