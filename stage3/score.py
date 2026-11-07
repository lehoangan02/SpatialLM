"""Score several prediction folders against GT with SpatialLM's own eval.py metrics.

    python stage3/score.py --gt_dir T/layout --metadata T/test_subset.csv --label_mapping T/benchmark_categories.tsv \
        --pred name1=dir1 name2=dir2 ... [--json out.json]
"""

import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
spec = importlib.util.spec_from_file_location("sl_eval", os.path.join(REPO, "eval.py"))
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)
from spatiallm import Layout  # noqa: E402


def score_dir(pred_dir, gt_dir, scenes, class_map):
    t25, t50 = {}, {}
    for scene in scenes:
        p = os.path.join(pred_dir, f"{scene}.txt")
        pred = Layout(open(p).read()) if os.path.exists(p) else Layout()
        gt = Layout(open(os.path.join(gt_dir, f"{scene}.txt")).read())
        pred.bboxes = E.assign_class_map(pred.bboxes, class_map)
        gt.bboxes = E.assign_class_map(gt.bboxes, class_map)
        E.assign_minimum_scale(pred.bboxes, 0.1)
        E.assign_minimum_scale(gt.bboxes, 0.1)
        pl = {w.id: w for w in pred.walls}
        gl = {w.id: w for w in gt.walls}
        pi = [e for e in pred.walls if E.is_valid_wall(e)] + [e for e in pred.doors + pred.windows if E.is_valid_dw(e, pl)]
        gi = [e for e in gt.walls if E.is_valid_wall(e)] + [e for e in gt.doors + gt.windows if E.is_valid_dw(e, gl)]
        for c in E.LAYOUTS:
            pe = [b for b in pi if E.get_entity_class(b) == c]
            ge = [b for b in gi if E.get_entity_class(b) == c]
            t25.setdefault(c, []).append(E.calc_layout_tp(pe, ge, pl, gl, iou_threshold=0.25))
            t50.setdefault(c, []).append(E.calc_layout_tp(pe, ge, pl, gl, iou_threshold=0.50))
        po = [b for b in pred.bboxes if b.class_name in E.OBJECTS]
        go = [b for b in gt.bboxes if b.class_name in E.OBJECTS]
        for c in E.OBJECTS:
            pe = [b for b in po if E.get_entity_class(b) == c]
            ge = [b for b in go if E.get_entity_class(b) == c]
            t25.setdefault(c, []).append(E.calc_bbox_tp(pe, ge, 0.25))

    def f1(tuples):
        v = np.ma.masked_where([t.masked for t in tuples], [t.f1 for t in tuples]).mean()
        return None if v is np.ma.masked else float(v)

    res = {f"{c}@.25": f1(t25[c]) for c in E.LAYOUTS}
    res.update({f"{c}@.50": f1(t50[c]) for c in E.LAYOUTS})
    obj = [f1(t25[c]) for c in E.OBJECTS]
    obj = [v for v in obj if v is not None]
    res["objects_mean@.25"] = float(np.mean(obj)) if obj else None
    for c in ("wall", "door", "window"):
        res[f"n_pred_{c}"] = int(sum(t.num_pred for t in t25[c]))
        res[f"n_gt_{c}"] = int(sum(t.num_gt for t in t25[c]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--label_mapping", required=True)
    ap.add_argument("--pred", nargs="+", required=True)
    ap.add_argument("--json")
    args = ap.parse_args()
    scenes = pd.read_csv(args.metadata)["id"].tolist()
    class_map = E.read_label_mapping(args.label_mapping, "spatiallm59", "spatiallm20")
    rows = {}
    for item in args.pred:
        name, d = item.split("=", 1)
        rows[name] = score_dir(d, args.gt_dir, scenes, class_map)
    cols = ["wall@.25", "wall@.50", "door@.25", "door@.50", "window@.25", "window@.50", "objects_mean@.25"]
    fmt = lambda v: "  -  " if v is None else f"{v:.3f}"
    print(f"{'variant':28s}" + "".join(f"{c:>17s}" for c in cols) + "   pred walls/doors/windows (gt)")
    for name, r in rows.items():
        counts = f"   {r['n_pred_wall']}/{r['n_pred_door']}/{r['n_pred_window']} ({r['n_gt_wall']}/{r['n_gt_door']}/{r['n_gt_window']})"
        print(f"{name:28s}" + "".join(f"{fmt(r[c]):>17s}" for c in cols) + counts)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
