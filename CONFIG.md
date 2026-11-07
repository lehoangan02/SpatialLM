# Chosen configuration

This is the pipeline chosen from the tests in [TESTING.md](TESTING.md), for turning a room scan into a layout on a
4 GB GPU. On the 10-scene ground-truth subset it scores wall F1@.25 **0.861**, door **0.731** and window **0.719**.
Raw SpatialLM (Qwen-0.5B, full cloud) scores 0.743 / 0.561 / 0.569 (TESTING.md §3.5).

## Summary

| Setting | Choice | Why (TESTING.md section) |
|---|---|---|
| Alignment | `stage3/align.py`, `--ceiling_height` = real ceiling height (default 2.6 m) | A wrong scale halves wall F1 (§3.7) |
| Input for walls / doors / windows | Cloud cropped to the room footprint (`stage3/crop.py`, margin 0.3 m) | +0.11 to +0.14 window F1 (§3.5) |
| Input for furniture boxes | Full aligned cloud | Cropping cuts furniture near walls; objects 0.423 vs 0.460 (§3.3) |
| Model | `SpatialLM1.1-Llama-1B` | Best on all three classes when it fits (§3.5) |
| Fallback model | `SpatialLM1.1-Qwen-0.5B` | Llama-1B can run out of memory on 4 GB (§3.1); Qwen within noise after snapping |
| Precision | `bfloat16` (`float16` is equivalent) | No quality difference (§3.2) |
| Tasks | Two passes: `arch` on the cropped cloud, `object` on the full cloud | Better than one `all` run (§3.3) |
| Decoding | Default sampling: temperature 0.6, top-k 10, top-p 0.95, 1 beam | Greedy was inconsistent (§3.3, §3.7) |
| Seeds | `arch` seeds 0, 1, 2; keep the result with the highest wall support | Seeds vary; best-of-3 is cheap (§3.8) |
| Post-processing | `stage3/hybrid.py` = `hyb_snap` with `--footprint auto` | Largest single gain: +0.07–0.09 walls, +0.15–0.20 doors (§3.4) |
| Not used | Voting, wall pruning, plane-fit walls, scan-gap opening check | Small gains or harmful (§3.3, §3.4) |

Required code change: the flash-attn fallback in `spatiallm/model/sonata_encoder.py` (already in this repo).

## Requirements

- Conda env with the SpatialLM dependencies (see README.md), plus `open3d`, `shapely`, `scipy`, `matplotlib` for `stage3/`.
- Model weights in `models/`:
  ```bash
  huggingface-cli download manycore-research/SpatialLM1.1-Llama-1B --local-dir models/SpatialLM1.1-Llama-1B
  huggingface-cli download manycore-research/SpatialLM1.1-Qwen-0.5B --local-dir models/SpatialLM1.1-Qwen-0.5B
  ```
- `models/`, `SpatialLM-Testset/` and `outputs/` are gitignored.

## Pipeline

Run the steps in this order. The examples use `scan.ply` (MASt3R-SLAM output) and write to `outputs/room/`.

```bash
O=outputs/room
L=models/SpatialLM1.1-Llama-1B
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 1. Level, Manhattan-align, floor at z=0, metric scale (set the real ceiling height)
python stage3/align.py --input scan.ply --output $O/room.ply --ceiling_height 2.6

# 2. Crop to the room footprint (+0.3 m) and between floor and ceiling
python stage3/crop.py --point_cloud $O/room.ply --output $O/crop/room.ply

# 3. SpatialLM: walls/doors/windows from the cropped cloud (3 seeds), boxes from the full cloud
python stage3/run_spatiallm.py --model_path $L --dtype bfloat16 \
    --inputs $O/crop/room.ply --out_dir $O/sl_crop --detect_type arch --seeds 0 1 2
python stage3/run_spatiallm.py --model_path $L --dtype bfloat16 \
    --inputs $O/room.ply --out_dir $O/sl_full --detect_type object --seeds 0

# 4. Snap each arch result onto the plane-fit geometry and attach the boxes
for s in 0 1 2; do
  python stage3/hybrid.py --point_cloud $O/room.ply --floor 0 \
      --spatiallm $O/sl_crop/arch_s$s/room.txt --spatiallm_boxes $O/sl_full/object_s0/room.txt \
      --output $O/final/s$s/room.txt
done

# 5. Compare the seeds; keep the one with the highest wall_support
python stage3/render.py --point_cloud $O/room.ply --floor 0 --cols 3 --out $O/compare.png \
    --layouts s0=$O/final/s0/room.txt s1=$O/final/s1/room.txt s2=$O/final/s2/room.txt

# 6. Export the chosen layout to glTF
python stage3/export_3d.py $O/final/s0/room.txt $O/room.glb
python stage3/export_3d.py $O/final/s0/room.txt $O/room_with_scan.glb --point_cloud $O/room.ply
```

### When Llama-1B runs out of memory

`run_spatiallm.py` does not crash on an out-of-memory error. It writes an empty layout and logs `"error": "Empty()"`
in the `RESULT` line and in `runlog.json`. If that happens:

1. Use the other seeds that did finish. Out-of-memory depends on output length, so other seeds of the same cloud often run.
2. If all seeds fail, or the `object` pass fails, rerun that step with
   `--model_path models/SpatialLM1.1-Qwen-0.5B --dtype float16`. Qwen peaks at about 2.5 GB.

## Evaluating on the test set

```bash
T=SpatialLM-Testset
python stage3/score.py --gt_dir $T/layout --metadata $T/test_subset.csv \
    --label_mapping $T/benchmark_categories.tsv --pred final=outputs/test/final/s0 [--json outputs/scores.json]
```

For the test set, `hybrid.py` used the original (uncropped) test cloud as `--point_cloud` and `--floor 0`, with the
`arch` result from the cropped cloud.

## Script reference (`stage3/`)

| Script | Role |
|---|---|
| `align.py` | Align and scale a raw scan (`--ceiling_height`, `--up_hint`, `--no_scale`) |
| `crop.py` | Crop to the plane-fit footprint (`--margin`, `--auto_footprint`) |
| `planefit.py` | Geometry-only layout; used by `crop.py`, `hybrid.py` and `render.py`, and runnable on its own |
| `run_spatiallm.py` | Batch SpatialLM inference with time / VRAM logging; writes `<out_dir>/<task>_s<seed>/<name>.txt` |
| `hybrid.py` | `hyb_snap` post-processing (`--footprint auto|sparse`, default `auto`) |
| `render.py` | Top-down comparison image and GT-free proxy scores |
| `score.py` | F1 against GT with SpatialLM's `eval.py` |
| `export_3d.py` | Layout to `.glb` |
