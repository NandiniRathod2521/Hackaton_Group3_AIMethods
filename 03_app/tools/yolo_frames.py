#!/usr/bin/env python
"""Run the existing YOLO pose extraction on decoded frames (see decode_frames.py).

Uses YoloPose and pick_person from 01_autoencoder/video_to_keypoints.py unchanged, and
writes the same VTT-format CSV (frame_number, timestamp, x0,y0,prob0 ... detection_score).

Usage
    python 03_app/tools/yolo_frames.py <frames dir> --out 03_app/prototype_out/yolo_frames [--imgsz 640]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "01_autoencoder"))
import video_to_keypoints as v2k  # noqa: E402


def run_clip(clip_dir: Path, out_dir: Path, detector) -> None:
    manifest = json.loads((clip_dir / "manifest.json").read_text(encoding="utf-8"))
    rows = np.zeros((manifest["ticks"], v2k.N_KP * 3 + 1))
    prev_centre = None
    for tick in range(manifest["ticks"]):
        frame = cv2.imread(str(clip_dir / f"{tick:06d}.jpg"))
        pick = v2k.pick_person(detector(frame), prev_centre)
        if pick is not None:
            kp, box, score = pick
            prev_centre = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            rows[tick, :v2k.N_KP * 3] = kp.reshape(-1)
            rows[tick, -1] = score
    header = ["frame_number", "timestamp"] + [f"{c}{k}" for k in range(v2k.N_KP) for c in ("x", "y", "prob")] + ["detection_score"]
    with open(out_dir / f"{manifest['clip']}.keypoints.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for tick, r in enumerate(rows):
            w.writerow([tick, tick * 1000 // v2k.OUT_FPS] + [f"{v:.4f}" for v in r])
    print(f"{manifest['clip']}: {manifest['ticks']} frames, person in {(rows[:, -1] > 0).mean():.0%}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames", help="folder written by decode_frames.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--weights", default=str(REPO / "yolo11n-pose.pt"))
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    detector = v2k.YoloPose(args.weights, imgsz=args.imgsz)
    frames = Path(args.frames)
    for clip in json.loads((frames / "clips.json").read_text(encoding="utf-8")):
        run_clip(frames / clip, out, detector)


if __name__ == "__main__":
    main()
