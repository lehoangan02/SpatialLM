# Chosen configuration

This is the pipeline chosen from the tests in [TESTING.md](TESTING.md), for turning a room scan into a layout on a
4 GB GPU. On the 10-scene ground-truth subset it scores wall F1@.25 **0.861**, door **0.731** and window **0.719**.
Raw SpatialLM (Qwen-0.5B, full cloud) scores 0.743 / 0.561 / 0.569 (TESTING.md §3.5).

## Summary

| Setting | Choice | Why (TESTING.md section) |
|---|---|---|
| Alignment | `stage3/align.py`, `--ceiling_height` = real ceiling height (default 2.6 m) | A wrong scale halves wall F1 (§3.7) |
| Up axis | `--up_hint auto`: Manhattan axis with the smallest room extent; sign from furniture tops | Room 2's up axis is not the camera `-y` axis; a fixed hint laid the room on its side |
| Input size | Thin each SpatialLM input to ≤ 120k points (voxel grown in 0.5 cm steps) | Room 2 at 555k–604k points ran out of memory with both models; 119k ran (§3.1) |
| Input for walls / doors / windows | Cloud cropped to the room footprint (`stage3/crop.py`, margin 0.3 m) | +0.11 to +0.14 window F1 (§3.5) |
| Input for furniture boxes | Cropped cloud; exact repeats and boxes outside the room removed | On room 2 the full cloud made SpatialLM loop (174 copies of one box outside the room, 6 min). On the GT set the full cloud was slightly better (0.460 vs 0.423, one seed), so `--object_input full` is still available (§3.10) |
| Model | `SpatialLM1.1-Llama-1B` | Best on all three classes when it fits (§3.5) |
| Fallback model | `SpatialLM1.1-Qwen-0.5B` | Llama-1B can run out of memory on 4 GB (§3.1); Qwen within noise after snapping |
| Precision | `bfloat16` (`float16` is equivalent) | No quality difference (§3.2) |
| Tasks | Two passes: `arch` and `object`, both on the cropped cloud | Better than one `all` run (§3.3) |
| Decoding | Default sampling: temperature 0.6, top-k 10, top-p 0.95, 1 beam | Greedy was inconsistent (§3.3, §3.7) |
| Seeds | `arch` seeds 0, 1, 2; keep the result with the highest wall support | Seeds vary; best-of-3 is cheap (§3.8) |
| Post-processing | `stage3/hybrid.py` = `hyb_snap` with `--footprint auto` | Largest single gain: +0.07–0.09 walls, +0.15–0.20 doors (§3.4) |
| Not used | Voting, wall pruning, plane-fit walls, scan-gap opening check | Small gains or harmful (§3.3, §3.4) |

Required code changes (already in this repo):

- the flash-attn fallback in `spatiallm/model/sonata_encoder.py`;
- `prepare_inputs_for_generation` in `spatiallm/model/spatiallm_{llama,qwen}.py` passes `num_logits_to_keep`
  through, so generation does not build logits for the whole prompt. This saves 0.12–0.19 GB and gives identical
  output. Without it, Llama-1B ran out of memory on room 2 (§3.10).

## Requirements

- Conda env with the SpatialLM dependencies (see README.md), plus `open3d`, `shapely`, `scipy`, `matplotlib` for `stage3/`.
- Model weights in `models/`:
  ```bash
  huggingface-cli download manycore-research/SpatialLM1.1-Llama-1B --local-dir models/SpatialLM1.1-Llama-1B
  huggingface-cli download manycore-research/SpatialLM1.1-Qwen-0.5B --local-dir models/SpatialLM1.1-Qwen-0.5B
  ```
- `models/`, `SpatialLM-Testset/` and `outputs/` are gitignored.

## One command

```bash
python stage3/pipeline.py scan.ply
```

This runs every step below with the chosen settings, with one model load. It retries failed Llama-1B runs with
Qwen-0.5B and picks the best seed. Total time on the T400 is about 1.5–3 minutes per room.

| Option | Default | Meaning |
|---|---|---|
| `scan` (positional) | – | Input point cloud (`.ply`), e.g. MASt3R-SLAM output |
| `--out_dir` | `outputs/<scan name>` | Output folder |
| `--model` | `llama` | `llama` (SpatialLM1.1-Llama-1B) or `qwen` (SpatialLM1.1-Qwen-0.5B) |
| `--dtype` | `bfloat16` | `bfloat16`, `float16` or `float32` |
| `--seeds` | `0 1 2` | Seeds for the walls/doors/windows pass; the best one is kept |
| `--object_seed` | `0` | Seed for the furniture pass |
| `--object_input` | `cropped` | Cloud for the furniture pass: `cropped` or `full` |
| `--no_fallback` | off | Do not retry failed Llama runs with Qwen |
| `--max_input_points` | `120000` | Thin each SpatialLM input to at most this many points |
| `--ceiling_height` | `2.6` | Real ceiling height in metres (sets the scale) |
| `--up_hint` | `auto` | Up axis of the input (`x`, `-y`, `z`, …), or `auto` |
| `--aligned` | off | Input is already aligned (z up, floor at z=0, metres); skip alignment |
| `--no_crop` | off | Run the walls/doors/windows pass on the full cloud |
| `--margin` | `0.3` | Crop margin around the room footprint (m) |
| `--footprint` | `auto` | Plane-fit footprint mode, `auto` or `sparse` |
| `--glb_points` | `300000` | Scan points stored in `room_with_scan.glb` |

Examples:

```bash
python stage3/pipeline.py ~/scans/kitchen.ply --ceiling_height 2.4
python stage3/pipeline.py scan.ply --model qwen --seeds 0 --out_dir outputs/quick
python stage3/pipeline.py SpatialLM-Testset/pcd/scene0000_00.ply --aligned
```

Output:

```
outputs/<scan name>/
├── layout.txt              final layout (chosen seed)
├── room.glb                3D model of the layout
├── room_with_scan.glb      3D model of the layout + scan points
├── seeds.png               top-down view of every seed's final layout
├── summary.json            settings, chosen seed, per-seed scores, per-run time / VRAM / errors
├── seeds/
│   └── s0.txt, s1.txt, …   final layout of every seed
└── intermediate/
    ├── aligned.ply                 scan in the layout frame
    ├── aligned_transform.json      raw -> aligned: aligned = (R @ p - [0,0,floor_z_raw]) * scale
    ├── cropped.ply                 aligned scan cropped to the room footprint
    └── spatiallm/
        ├── arch_s0.txt, …          raw SpatialLM walls/doors/windows per seed
        └── object_s0.txt           raw SpatialLM furniture boxes
```

`layout.txt` uses SpatialLM's text format in the aligned frame (metres, z up, floor at z=0):

```
wall_0=Wall(ax,ay,az,bx,by,bz,height,thickness)
door_0=Door(wall_0,x,y,z,width,height)
window_0=Window(wall_1,x,y,z,width,height)
bbox_0=Bbox(chair,x,y,z,angle_z,scale_x,scale_y,scale_z)
```

## Step by step

The same pipeline as separate scripts, for debugging single steps. These scripts do not thin the input. For dense
scans, voxel-downsample the clouds (4–5 cm) before step 3, or use `pipeline.py`.

```bash
O=outputs/room
L=models/SpatialLM1.1-Llama-1B
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 1. Level, Manhattan-align, floor at z=0, metric scale (set the real ceiling height)
python stage3/align.py --input scan.ply --output $O/room.ply --ceiling_height 2.6

# 2. Crop to the room footprint (+0.3 m) and between floor and ceiling
python stage3/crop.py --point_cloud $O/room.ply --output $O/crop/room.ply

# 3. SpatialLM on the cropped cloud: walls/doors/windows (3 seeds) and boxes
python stage3/run_spatiallm.py --model_path $L --dtype bfloat16 \
    --inputs $O/crop/room.ply --out_dir $O/sl_crop --detect_type arch object --seeds 0 1 2

# 4. Snap each arch result onto the plane-fit geometry and attach the boxes
for s in 0 1 2; do
  python stage3/hybrid.py --point_cloud $O/room.ply --floor 0 \
      --spatiallm $O/sl_crop/arch_s$s/room.txt --spatiallm_boxes $O/sl_crop/object_s0/room.txt \
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
| `pipeline.py` | The whole pipeline in one command |
| `align.py` | Align and scale a raw scan (`--ceiling_height`, `--up_hint`, `--no_scale`) |
| `crop.py` | Crop to the plane-fit footprint (`--margin`, `--auto_footprint`) |
| `planefit.py` | Geometry-only layout; used by `crop.py`, `hybrid.py` and `render.py`, and runnable on its own |
| `run_spatiallm.py` | Batch SpatialLM inference with time / VRAM logging; writes `<out_dir>/<task>_s<seed>/<name>.txt` |
| `hybrid.py` | `hyb_snap` post-processing (`--footprint auto|sparse`, default `auto`) |
| `render.py` | Top-down comparison image and GT-free proxy scores |
| `score.py` | F1 against GT with SpatialLM's `eval.py` |
| `export_3d.py` | Layout to `.glb` |
