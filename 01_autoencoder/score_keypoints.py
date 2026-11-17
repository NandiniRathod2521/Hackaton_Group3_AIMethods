#!/usr/bin/env python
"""Score keypoint CSVs: how unusual is the movement, and what postures were held?

Works on the dataset's keypoint files and on the CSVs written by video_to_keypoints.py.
Needs keypoint_utils.py in the same folder (or on PYTHONPATH).

1. Anomaly score (needs the trained keypoint model, models/keypoint/):
   each 48-frame window goes through the autoencoder; windows whose reconstruction error is above the
   threshold stored in config.json are flagged.
2. Posture summary (no model needed): share of time with the trunk bent forward, knees bent or arms raised,
   computed from joint angles.

The posture thresholds are indicative values in the spirit of the REBA ergonomic method
(trunk flexion over 20 deg, knee bend over 30 deg, arm elevation over 90 deg).
They are NOT a medical or legal assessment; change them with the options below.

Usage
    python score_keypoints.py keypoints_out --model-dir models/keypoint
    python score_keypoints.py clip.keypoints.csv --angles-only
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import keypoint_utils as ku


def posture_summary(feats, valid, args):
    a = ku.joint_angles(feats)
    trunk = a["trunk_lean"]
    knee = 180 - np.minimum(a["knee_left"], a["knee_right"])
    arm = np.maximum(a["shoulder_left"], a["shoulder_right"])
    ok = valid[:, [5, 6, 11, 12]].all(axis=1)          # frames where shoulders and hips were really seen
    if ok.sum() < 10:
        ok = np.ones(len(feats), bool)
    share = lambda v, t: float((v[ok] > t).mean())
    return {
        "trunk_lean_median": round(float(np.median(trunk[ok])), 1),
        "knee_bend_median": round(float(np.median(knee[ok])), 1),
        "arm_elevation_median": round(float(np.median(arm[ok])), 1),
        f"time_trunk_over_{args.trunk}deg": round(share(trunk, args.trunk), 2),
        f"time_knee_over_{args.knee}deg": round(share(knee, args.knee), 2),
        f"time_arm_over_{args.arm}deg": round(share(arm, args.arm), 2),
        "frames_with_body_seen": round(float(ok.mean()), 2),
    }


def load_model(model_dir):
    import onnxruntime as ort
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    sc = np.load(model_dir / "scaler.npz")
    sess = ort.InferenceSession(str(model_dir / "model.onnx"), providers=["CPUExecutionProvider"])
    return sess, config, sc["mean"], sc["scale"]


def score_file(path, model, args):
    x, y, p = ku.read_keypoints(path)
    feats, valid = ku.clean_clip(x, y, p)
    row = {"clip": Path(path).name, "seconds": round(len(feats) / ku.FPS, 1),
           "mean_point_confidence": round(float(p[p > 0].mean()) if (p > 0).any() else 0.0, 2)}
    row.update(posture_summary(feats, valid, args))
    if model is not None:
        sess, config, mean, scale = model
        wins = ku.preprocess_for_model(x, y, p, mean, scale)
        row["windows_scored"] = len(wins)
        if len(wins):
            _, rec = sess.run(["embedding", "reconstruction"], {"keypoint_window": wins})
            err = ((rec - wins) ** 2).mean(axis=(1, 2))
            thr = config["threshold_mse"]
            row["anomaly_error_median"] = round(float(np.median(err)), 3)
            row["windows_above_threshold"] = round(float((err > thr).mean()), 2)
            row["threshold"] = round(float(thr), 3)
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="a keypoint CSV or a folder of them")
    ap.add_argument("--model-dir", default="models/keypoint", help="folder with model.onnx, scaler.npz, config.json")
    ap.add_argument("--angles-only", action="store_true", help="skip the autoencoder, posture summary only")
    ap.add_argument("--trunk", type=float, default=20, help="trunk lean threshold in degrees")
    ap.add_argument("--knee", type=float, default=30, help="knee bend threshold in degrees")
    ap.add_argument("--arm", type=float, default=90, help="arm elevation threshold in degrees")
    ap.add_argument("--out", default="clip_scores.csv", help="where to save the table")
    args = ap.parse_args(argv)

    src = Path(args.input)
    files = sorted(src.rglob("*.csv")) if src.is_dir() else [src]
    files = [f for f in files if "clip_scores" not in f.name]
    if not files:
        sys.exit(f"No CSV files in {src}")
    model = None if args.angles_only else load_model(args.model_dir)

    rows = []
    for f in files:
        try:
            rows.append(score_file(f, model, args))
        except Exception as e:
            print(f"{f.name}: FAILED - {e}")
    table = pd.DataFrame(rows)
    table.to_csv(args.out, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(table.to_string(index=False))
    print(f"\nSaved {args.out}")
    return table


if __name__ == "__main__":
    main()
