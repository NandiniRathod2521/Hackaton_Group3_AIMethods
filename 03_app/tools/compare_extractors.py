#!/usr/bin/env python
"""Compare keypoint extractors on the same clips: contract checks, agreement, and model scores.

Scoring and preprocessing reuse 02_model_server/keypoint_utils.py and the local keypoint model
(models/keypoint/), i.e. the same steps as Render's /predict/keypoints. YOLO is a comparison
point, not ground truth: the training data's pose detector is unknown.

Usage (from the repo root)
    python 03_app/tools/compare_extractors.py --frames <frames dir> --out 03_app/prototype_out
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "02_model_server"))
import keypoint_utils as ku  # noqa: E402

LR_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
CHECKS = {"flag_agreement_min": 0.90, "median_rel_score_diff_max": 0.25,
          "posture_median_diff_max_deg": 7.0, "lr_swap_rate_max": 0.02}


def load(path):
    d = pd.read_csv(path)
    x, y, p = ku.read_keypoints(path)
    return {"x": x, "y": y, "p": p, "det": d["detection_score"].to_numpy(), "ts": d["timestamp"].to_numpy()}


def score(kp, model):
    """Per-window reconstruction MSE keyed by each window's original start frame (as the API reports it)."""
    sess, mean, scale, thr, pmin = model
    wins, starts = ku.preprocess_for_model_with_starts(kp["x"], kp["y"], kp["p"], mean, scale, pmin=pmin)
    starts = starts.tolist()
    if not len(wins):
        return {}, thr
    rec = sess.run(["reconstruction"], {sess.get_inputs()[0].name: wins})[0]
    return dict(zip(starts, ((rec - wins) ** 2).mean(axis=(1, 2)).astype(float))), thr


def posture(kp, pmin):
    """Same as app.py _posture_summary (medians in degrees)."""
    feats, valid = ku.clean_clip(kp["x"], kp["y"], kp["p"], pmin=pmin)
    a = ku.joint_angles(feats)
    vis = valid[:, [5, 6, 11, 12]].all(axis=1)
    if vis.sum() < 10:
        vis = np.ones(len(feats), bool)
    return {"trunk": float(np.median(a["trunk_lean"][vis])),
            "knee": float(np.median((180 - np.minimum(a["knee_left"], a["knee_right"]))[vis])),
            "arm": float(np.median(np.maximum(a["shoulder_left"], a["shoulder_right"])[vis]))}


def contract(kp, manifest):
    """Checks on one extractor's output against the model/API input contract."""
    seen = kp["det"] > 0
    xs, ys = kp["x"][seen], kp["y"][seen]
    # anatomical order (image y points down) on frames where the reference says the person is upright
    yy = kp["y"]
    order = np.stack([yy[:, [5, 6]].mean(1) < yy[:, [11, 12]].mean(1),
                      yy[:, [11, 12]].mean(1) < yy[:, [13, 14]].mean(1),
                      yy[:, [13, 14]].mean(1) < yy[:, [15, 16]].mean(1)], 1)
    return {
        "rows": len(kp["p"]), "expected_rows": manifest["ticks"],
        "timestamp_step_ms": sorted(set(np.diff(kp["ts"]).tolist())),
        "person_found": float(seen.mean()),
        "conf_min": float(kp["p"].min()), "conf_max": float(kp["p"].max()),
        "conf_median_detected": float(np.median(kp["p"][seen])) if seen.any() else 0.0,
        "x_outside_frame": float(((xs < 0) | (xs > manifest["width"])).mean()) if seen.any() else 0.0,
        "y_outside_frame": float(((ys < 0) | (ys > manifest["height"])).mean()) if seen.any() else 0.0,
        "valid_points": float((kp["p"] >= 0.02).mean()),
        "_order_ok": order[seen],
    }


def agreement(a, b):
    """a vs reference b, on frames where both found a person. Distances in b's torso lengths."""
    both = (a["det"] > 0) & (b["det"] > 0)
    if both.sum() < 5:
        return None
    bx, by = b["x"][both], b["y"][both]
    ax, ay = a["x"][both], a["y"][both]
    torso = np.median(np.hypot(bx[:, [5, 6]].mean(1) - bx[:, [11, 12]].mean(1),
                               by[:, [5, 6]].mean(1) - by[:, [11, 12]].mean(1)))
    dist = np.hypot(ax - bx, ay - by) / torso
    swap_idx = np.arange(17)
    for l, r in LR_PAIRS:
        swap_idx[l], swap_idx[r] = r, l
    limbs = list(range(5, 17))
    direct = dist[:, limbs].mean(1)
    swapped = (np.hypot(ax[:, swap_idx] - bx, ay[:, swap_idx] - by) / torso)[:, limbs].mean(1)
    fa, _ = ku.clean_clip(a["x"], a["y"], a["p"])
    fb, _ = ku.clean_clip(b["x"], b["y"], b["p"])
    return {
        "frames_both": int(both.sum()),
        "per_joint_median": np.median(dist, axis=0),
        "median_err_torso": float(np.median(dist)),
        "pck_0.2": float((dist < 0.2).mean()),
        "lr_swap_rate": float((swapped < 0.7 * direct).mean()),
        "feature_rmse_torso": float(np.sqrt(((fa - fb) ** 2)[both].mean())),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", required=True, help="frames dir (for manifests)")
    ap.add_argument("--out", default=str(REPO / "03_app" / "prototype_out"))
    ap.add_argument("--model-dir", default=str(REPO / "models" / "keypoint"))
    ap.add_argument("--exclude", nargs="*", default=[], help="skip clips whose name contains any of these")
    ap.add_argument("--tag", default="", help="suffix for the report file names")
    args = ap.parse_args(argv)
    out, frames = Path(args.out), Path(args.frames)

    cfg = json.loads((Path(args.model_dir) / "config.json").read_text(encoding="utf-8"))
    sc = np.load(Path(args.model_dir) / "scaler.npz")
    model = (ort.InferenceSession(str(Path(args.model_dir) / "model.onnx"), providers=["CPUExecutionProvider"]),
             sc["mean"], sc["scale"], float(cfg["threshold_mse"]), float(cfg["pmin"]))
    pmin = model[4]

    sources = {
        "yolo_video": REPO / "keypoints_out",          # existing converter output (original video)
        "yolo_640": out / "yolo_frames_640",           # same frames as MoveNet, converter default size
        "yolo_960": out / "yolo_frames_960",
        "movenet_frames": out / "movenet_frames",
        "movenet_video": out / "movenet_video",        # real app path: browser decodes + samples video
    }
    pairs = [
        ("YOLO original video vs YOLO decoded frames (decode check)", "yolo_640", "yolo_video"),
        ("YOLO 960 vs YOLO 640 (detector noise floor)", "yolo_960", "yolo_640"),
        ("MoveNet vs YOLO, identical frames (detector)", "movenet_frames", "yolo_640"),
        ("MoveNet browser video vs MoveNet frames (sampling/decoding)", "movenet_video", "movenet_frames"),
        ("MoveNet browser video vs existing YOLO pipeline (end to end)", "movenet_video", "yolo_video"),
    ]
    clips = [c for c in json.loads((frames / "clips.json").read_text(encoding="utf-8"))
             if not any(e in c for e in args.exclude)]
    data = {s: {c: load(d / f"{c}.keypoints.csv") for c in clips if (d / f"{c}.keypoints.csv").exists()}
            for s, d in sources.items()}

    lines = ["# MoveNet vs YOLO keypoint comparison (prototype checks)", "",
             f"Model: `{cfg['model']}`, threshold {model[3]:.4f}, pmin {pmin}. Local ONNX scoring with the "
             "server's keypoint_utils. YOLO is a comparison point, not ground truth. Passing these checks "
             "does not validate the anomaly scores.", "",
             f"Clips: {len(clips)}" + (f" (excluded: {', '.join(args.exclude)})" if args.exclude else ""), ""]

    # 1. contract checks per extractor
    lines += ["## Input contract per extractor", "",
              "| extractor | clips | rows = expected | ts step (ms) | person found | conf min–max | median conf | x/y outside frame | valid points (≥ pmin) | body order ok |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for s in sources:
        rs = [contract(data[s][c], json.loads((frames / c / "manifest.json").read_text(encoding="utf-8"))) for c in data[s]]
        if not rs:
            continue
        order = np.concatenate([r["_order_ok"] for r in rs])
        lines.append(
            f"| {s} | {len(rs)} | {sum(r['rows'] == r['expected_rows'] for r in rs)}/{len(rs)} | "
            f"{sorted({v for r in rs for v in r['timestamp_step_ms']})} | {np.mean([r['person_found'] for r in rs]):.1%} | "
            f"{min(r['conf_min'] for r in rs):.3f}–{max(r['conf_max'] for r in rs):.3f} | "
            f"{np.median([r['conf_median_detected'] for r in rs]):.2f} | "
            f"{np.mean([r['x_outside_frame'] for r in rs]):.2%} / {np.mean([r['y_outside_frame'] for r in rs]):.2%} | "
            f"{np.mean([r['valid_points'] for r in rs]):.1%} | "
            f"sh<hip {order[:, 0].mean():.0%}, hip<knee {order[:, 1].mean():.0%}, knee<ankle {order[:, 2].mean():.0%} |")
    lines.append("")

    # 2. pairwise agreement and scores
    summary, per_clip = [], []
    for title, a_name, b_name in pairs:
        common = [c for c in clips if c in data[a_name] and c in data[b_name]]
        if not common:
            continue
        sa_all, sb_all, ag_rows = [], [], []
        for c in common:
            a, b = data[a_name][c], data[b_name][c]
            ag = agreement(a, b)
            sa, thr = score(a, model)
            sb, _ = score(b, model)
            pa, pb = posture(a, pmin), posture(b, pmin)
            shared = sorted(set(sa) & set(sb))
            sa_all += [sa[s] for s in shared]
            sb_all += [sb[s] for s in shared]
            row = {"pair": title, "clip": c.replace("WhatsApp Video 2026-11-17 at ", ""),
                   "windows_a": len(sa), "windows_b": len(sb), "same_window_starts": sorted(sa) == sorted(sb),
                   "pct_above_a": np.mean([v > thr for v in sa.values()]) if sa else np.nan,
                   "pct_above_b": np.mean([v > thr for v in sb.values()]) if sb else np.nan,
                   "median_score_a": np.median(list(sa.values())) if sa else np.nan,
                   "median_score_b": np.median(list(sb.values())) if sb else np.nan,
                   **{f"d_{k}": pa[k] - pb[k] for k in pa}}
            if ag:
                row.update({k: v for k, v in ag.items() if k != "per_joint_median"})
                ag_rows.append(ag)
            per_clip.append(row)
        sa_all, sb_all = np.array(sa_all), np.array(sb_all)
        thr = model[3]
        clip_rows = [r for r in per_clip if r["pair"] == title]
        res = {
            "pair": title, "clips": len(common), "windows_compared": len(sa_all),
            "same_window_starts": f"{sum(r['same_window_starts'] for r in clip_rows)}/{len(clip_rows)}",
            "flag_agreement": float(((sa_all > thr) == (sb_all > thr)).mean()),
            "flags_a": int((sa_all > thr).sum()), "flags_b": int((sb_all > thr).sum()),
            "spearman": float(spearmanr(sa_all, sb_all).statistic) if len(sa_all) > 2 else np.nan,
            "median_rel_score_diff": float(np.median(np.abs(sa_all - sb_all) / sb_all)),
            "max_abs_posture_diff_deg": float(np.nanmax(np.abs([[r["d_trunk"], r["d_knee"], r["d_arm"]] for r in clip_rows]))),
            "median_abs_posture_diff_deg": {k: float(np.median(np.abs([r[f"d_{k}"] for r in clip_rows]))) for k in ("trunk", "knee", "arm")},
            "median_err_torso": float(np.median([g["median_err_torso"] for g in ag_rows])),
            "pck_0.2": float(np.mean([g["pck_0.2"] for g in ag_rows])),
            "lr_swap_rate": float(np.average([g["lr_swap_rate"] for g in ag_rows], weights=[g["frames_both"] for g in ag_rows])),
            "feature_rmse_torso": float(np.median([g["feature_rmse_torso"] for g in ag_rows])),
            "per_joint_median_err": dict(zip(ku.COCO_POINT_NAMES, np.round(np.median([g["per_joint_median"] for g in ag_rows], axis=0), 3).tolist())),
        }
        res["checks"] = {
            "flag_agreement": res["flag_agreement"] >= CHECKS["flag_agreement_min"],
            "median_rel_score_diff": res["median_rel_score_diff"] <= CHECKS["median_rel_score_diff_max"],
            "posture_medians": res["max_abs_posture_diff_deg"] <= CHECKS["posture_median_diff_max_deg"],
            "lr_swap_rate": res["lr_swap_rate"] <= CHECKS["lr_swap_rate_max"],
        }
        summary.append(res)

    lines += ["## Agreement (A vs reference B)", "",
              "| comparison | windows | same starts | flag agreement | flags A/B | Spearman | median rel. score diff | max posture diff (°) | median point err (torso) | PCK@0.2 | L/R swaps | feature RMSE |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in summary:
        ok = lambda k: "✅" if r["checks"][k] else "❌"
        lines.append(f"| {r['pair']} | {r['windows_compared']} | {r['same_window_starts']} | {r['flag_agreement']:.0%} {ok('flag_agreement')} | "
                     f"{r['flags_a']}/{r['flags_b']} | {r['spearman']:.2f} | {r['median_rel_score_diff']:.0%} {ok('median_rel_score_diff')} | "
                     f"{r['max_abs_posture_diff_deg']:.1f} {ok('posture_medians')} | {r['median_err_torso']:.3f} | {r['pck_0.2']:.0%} | "
                     f"{r['lr_swap_rate']:.1%} {ok('lr_swap_rate')} | {r['feature_rmse_torso']:.3f} |")
    lines += ["", f"Prototype checks: flag agreement ≥ {CHECKS['flag_agreement_min']:.0%}, median relative score difference ≤ "
              f"{CHECKS['median_rel_score_diff_max']:.0%}, every clip's posture medians within {CHECKS['posture_median_diff_max_deg']:.0f}°, "
              f"L/R swaps ≤ {CHECKS['lr_swap_rate_max']:.0%}.", ""]

    lines += ["## Per-joint median error (torso lengths)", "", "| joint | " + " | ".join(r["pair"].split(" (")[0] for r in summary) + " |",
              "|---|" + "---|" * len(summary)]
    for j in ku.COCO_POINT_NAMES:
        lines.append(f"| {j} | " + " | ".join(f"{r['per_joint_median_err'][j]:.3f}" for r in summary) + " |")
    lines.append("")

    pc = pd.DataFrame(per_clip)
    pc.to_csv(out / f"comparison_per_clip{args.tag}.csv", index=False)
    (out / f"comparison_summary{args.tag}.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    cols = ["clip", "windows_a", "windows_b", "pct_above_a", "pct_above_b", "median_score_a", "median_score_b", "d_trunk", "d_knee", "d_arm", "median_err_torso", "lr_swap_rate"]
    for title in [p[0] for p in pairs]:
        sub = pc[pc["pair"] == title]
        if len(sub):
            lines += [f"### Per clip: {title}", "", sub[cols].round(3).to_markdown(index=False), ""]
    (out / f"comparison_report{args.tag}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
