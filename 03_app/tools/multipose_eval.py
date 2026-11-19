#!/usr/bin/env python
"""Score the EXPERIMENTAL multi-person tracks (from tools/multipose_eval.mjs) the way the app does.

Each scoreable segment is scored on its own with the server's keypoint_utils + local ONNX model and
the trained threshold (same as one /predict/keypoints request per segment). Single-person clips are
compared with SinglePose on the whole clip (the current app path; a reference, not ground truth):
windows are matched by time overlap (>= half a window) because segments use their own window grid.

Usage (from 03_app)
    python tools/multipose_eval.py --dir prototype_out/multipose_eval
"""
import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compare_extractors as ce  # noqa: E402  (also puts 02_model_server on sys.path)
import onnxruntime as ort  # noqa: E402

REPO = HERE.parents[1]
MULTI_PERSON = {"4.03.35": "one person close up, legs out of frame",
                "4.03.44": "three people standing, full body",
                "4.03.47": "two roofers, small, crouching, from above"}
W = 48


def short(clip):
    return clip.replace("WhatsApp Video 2026-11-17 at ", "").replace(" PM", "")


def arrays(frames):
    a = np.asarray(frames, dtype=float)
    return {"x": a[:, :, 0], "y": a[:, :, 1], "p": a[:, :, 2]}


def ref_scores(csv_text, model):
    d = pd.read_csv(io.StringIO(csv_text))
    x = d[[f"x{k}" for k in range(17)]].to_numpy(float)
    y = d[[f"y{k}" for k in range(17)]].to_numpy(float)
    p = d[[f"prob{k}" for k in range(17)]].to_numpy(float)
    return ce.score({"x": x, "y": y, "p": p}, model)[0]


def track_scores(track, model):
    out = {}
    for s in track["segments"]:
        if s["frames"] is None:
            continue
        local, _ = ce.score(arrays(s["frames"]), model)
        out.update({s["start"] + k: v for k, v in local.items()})
    return out


def match(multi, ref):
    pairs = []
    for a, va in multi.items():
        best = max(ref, key=lambda r: W - abs(a - r), default=None)
        if best is not None and W - abs(a - best) >= W // 2:
            pairs.append((va, ref[best]))
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=str(HERE.parent / "prototype_out" / "multipose_eval"))
    ap.add_argument("--model-dir", default=str(REPO / "models" / "keypoint"))
    args = ap.parse_args()
    cfg = json.loads((Path(args.model_dir) / "config.json").read_text(encoding="utf-8"))
    sc = np.load(Path(args.model_dir) / "scaler.npz")
    model = (ort.InferenceSession(str(Path(args.model_dir) / "model.onnx"), providers=["CPUExecutionProvider"]),
             sc["mean"], sc["scale"], float(cfg["threshold_mse"]), float(cfg["pmin"]))
    thr = model[3]

    single_rows, multi_rows = [], []
    for f in sorted((Path(args.dir) / "tracks").glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        clip = short(d["clip"])
        ref = ref_scores(d["single"], model)
        for name, c in d["configs"].items():
            tracks = c["tracks"]
            if clip in MULTI_PERSON:
                for t in tracks:
                    s = track_scores(t, model)
                    multi_rows.append({
                        "clip": clip, "config": name, "track": t["label"], "frames": t["present"],
                        "coverage": t["present"] / d["ticks"], "fragments": len(t["rawIds"]),
                        "rejoins": sum(i["kind"] == "rejoin" for i in t["issues"]),
                        "jumps": sum(i["kind"] == "jump" for i in t["issues"]),
                        "duplicate": t["duplicate"], "scoreable": t["scoreable"],
                        "windows_scored": len(s), "above": int(sum(v > thr for v in s.values())),
                    })
                if not tracks:
                    multi_rows.append({"clip": clip, "config": name, "track": "—", "frames": 0, "coverage": 0.0, "fragments": 0,
                                       "rejoins": 0, "jumps": 0, "duplicate": False, "scoreable": False, "windows_scored": 0, "above": 0})
                continue
            main_t = max(tracks, key=lambda t: t["present"], default=None)
            s = track_scores(main_t, model) if main_t else {}
            pairs = match(s, ref)
            va = np.array([a for a, _ in pairs]); vb = np.array([b for _, b in pairs])
            single_rows.append({
                "clip": clip, "config": name, "labelled_tracks": len(tracks),
                "coverage": (main_t["present"] / d["ticks"]) if main_t else 0.0,
                "ref_windows": len(ref), "scored_windows": len(s), "matched": len(pairs),
                "flag_agree": int(((va > thr) == (vb > thr)).sum()) if len(pairs) else 0,
                "above_ref": int((vb > thr).sum()) if len(pairs) else 0, "above_multi": int((va > thr).sum()) if len(pairs) else 0,
                "rel_diff_median": float(np.median(np.abs(va - vb) / vb)) if len(pairs) else np.nan,
            })

    sdf, mdf = pd.DataFrame(single_rows), pd.DataFrame(multi_rows)
    out = Path(args.dir)
    sdf.to_csv(out / "single_person_by_clip.csv", index=False)
    mdf.to_csv(out / "multi_person_tracks.csv", index=False)

    summ = sdf.groupby("config").agg(
        clips=("clip", "count"), ref_windows=("ref_windows", "sum"), scored=("scored_windows", "sum"),
        matched=("matched", "sum"), agree=("flag_agree", "sum"), above_ref=("above_ref", "sum"), above_multi=("above_multi", "sum"),
        extra_tracks=("labelled_tracks", lambda s: int((s - 1).clip(lower=0).sum())),
        min_coverage=("coverage", "min"), rel_diff=("rel_diff_median", "median"),
    )
    summ["flag_agreement"] = summ.agree / summ.matched
    mp = mdf.groupby(["config", "clip"]).agg(
        labelled=("track", lambda s: int((s != "—").sum())), scoreable=("scoreable", "sum"),
        windows=("windows_scored", "sum"), duplicates=("duplicate", "sum"), rejoins=("rejoins", "sum"), jumps=("jumps", "sum"),
    ).reset_index()
    (out / "summary.json").write_text(json.dumps({
        "threshold": thr, "single_person": summ.reset_index().to_dict("records"), "multi_person": mp.to_dict("records"),
    }, indent=1, default=float), encoding="utf-8")
    pd.set_option("display.width", 220)
    print("SINGLE-PERSON CLIPS (main track vs SinglePose whole clip)")
    print(summ[["clips", "ref_windows", "scored", "matched", "flag_agreement", "above_ref", "above_multi", "rel_diff", "min_coverage", "extra_tracks"]].round(3).to_string())
    print("\nMULTI-PERSON CLIPS (labelled / scoreable tracks, scored windows)")
    print(mp.pivot_table(index="config", columns="clip", values=["labelled", "scoreable", "windows"], aggfunc="first").to_string())


if __name__ == "__main__":
    main()
