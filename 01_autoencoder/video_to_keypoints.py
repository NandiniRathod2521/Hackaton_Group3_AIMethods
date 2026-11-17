#!/usr/bin/env python
"""Turn phone videos into keypoint CSVs in the VTT-ConIoT format.

For every video it writes <name>.keypoints.csv with the columns
    frame_number, timestamp, x0, y0, prob0, ... x16, y16, prob16, detection_score
(17 COCO body points, 25 fps, timestamps in ms), the same layout as the dataset's
Subject_NN_Task_X.*_keyPoints.csv files, so keypoint_utils.py can read them.

What it takes care of
  * rotation: phone videos are often stored sideways with a rotation tag; frames are rotated upright
  * frame rate: any input fps is resampled to 25 fps (the rate the model was trained on)
  * several people: the person with the largest box is followed through the clip
  * frames without a detection get x = y = 0 and prob = 0 (the cleaner treats them as missing)

Setup (once, on your own computer, in the project's virtual environment):
    pip install ultralytics opencv-python
The first run downloads the small pose model (yolo11n-pose.pt, about 6 MB).
Note: ultralytics is AGPL-3.0 licensed. Mention it in the README.

Usage
    python video_to_keypoints.py path/to/Videos --out keypoints_out
    python video_to_keypoints.py clip.mp4 --out keypoints_out --preview
--preview also writes <name>.overlay.mp4 with the skeleton drawn on the video, to check the result by eye.
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

OUT_FPS = 25
N_KP = 17
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
# COCO skeleton for the preview
EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
         (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]


class YoloPose:
    """Thin wrapper so the detector can be swapped (or faked in tests)."""

    def __init__(self, weights="yolo11n-pose.pt", device=None, imgsz=640):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.device, self.imgsz = device, imgsz

    def __call__(self, frame_bgr):
        """Return a list of (keypoints (17, 3) = x, y, prob; box (4,) = x1, y1, x2, y2; score)."""
        r = self.model.predict(frame_bgr, verbose=False, imgsz=self.imgsz, device=self.device)[0]
        if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
            return []
        kp = r.keypoints.data.cpu().numpy()
        box = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()
        return [(kp[i], box[i], float(conf[i])) for i in range(len(kp))]


def open_video(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1.0)       # rotate upright using the file's rotation tag
    return cap


def count_frames(path):
    cap = open_video(path)
    n = 0
    while cap.grab():
        n += 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return n, fps


def wanted_frames(n_src, src_fps):
    """Source frame index for each 25 fps output tick (nearest frame in time)."""
    duration = n_src / src_fps
    ticks = int(np.floor((duration - 1.0 / src_fps) * OUT_FPS)) + 1
    idx = np.rint(np.arange(ticks) * src_fps / OUT_FPS).astype(int)
    return np.clip(idx, 0, n_src - 1)


def pick_person(dets, prev_centre):
    """Choose the person to follow: largest box first, then the one closest to where they were."""
    if not dets:
        return None
    if prev_centre is None:
        return max(dets, key=lambda d: (d[1][2] - d[1][0]) * (d[1][3] - d[1][1]))
    def dist(d):
        b = d[1]
        return np.hypot((b[0] + b[2]) / 2 - prev_centre[0], (b[1] + b[3]) / 2 - prev_centre[1])
    return min(dets, key=dist)


def draw(frame, kp, thr=0.2):
    for a, b in EDGES:
        if kp[a, 2] >= thr and kp[b, 2] >= thr:
            cv2.line(frame, tuple(int(v) for v in kp[a, :2]), tuple(int(v) for v in kp[b, :2]), (0, 200, 255), 2)
    for x, y, p in kp:
        if p >= thr:
            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)


def process(path, out_dir, detector, preview=False):
    n_src, src_fps = count_frames(path)
    if n_src == 0:
        raise RuntimeError(f"{path.name}: no frames")
    needed = wanted_frames(n_src, src_fps)
    need_set = {}
    for tick, i in enumerate(needed):
        need_set.setdefault(int(i), []).append(tick)

    rows = np.zeros((len(needed), N_KP * 3 + 1), dtype=np.float64)   # x,y,prob per point + detection score
    box_heights, frame_h = [], None
    prev_centre, writer = None, None

    cap = open_video(path)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i in need_set:
            frame_h = frame.shape[0]
            pick = pick_person(detector(frame), prev_centre)
            kp = None
            if pick is not None:
                kp, box, score = pick
                prev_centre = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                box_heights.append(box[3] - box[1])
            for tick in need_set[i]:
                if kp is not None:
                    rows[tick, :N_KP * 3] = kp.reshape(-1)
                    rows[tick, -1] = score
                if preview:
                    out = frame.copy()
                    if kp is not None:
                        draw(out, kp)
                    scale = 640.0 / max(out.shape[:2])
                    out = cv2.resize(out, (int(out.shape[1] * scale), int(out.shape[0] * scale)))
                    if writer is None:
                        writer = cv2.VideoWriter(str(out_dir / f"{path.stem}.overlay.mp4"),
                                                 cv2.VideoWriter_fourcc(*"mp4v"), OUT_FPS,
                                                 (out.shape[1], out.shape[0]))
                    writer.write(out)
        i += 1
    cap.release()
    if writer is not None:
        writer.release()

    header = ["frame_number", "timestamp"] + [f"{c}{k}" for k in range(N_KP) for c in ("x", "y", "prob")] + ["detection_score"]
    out_csv = out_dir / f"{path.stem}.keypoints.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for tick, r in enumerate(rows):
            w.writerow([tick, tick * 1000 // OUT_FPS] + [f"{v:.4f}" for v in r])

    found = (rows[:, -1] > 0).mean()
    conf = rows[:, 2:N_KP * 3:3][rows[:, -1] > 0].mean() if found else 0.0
    notes = []
    if found < 0.8:
        notes.append("person found in fewer than 80% of frames")
    if box_heights and frame_h and np.median(box_heights) < 0.25 * frame_h:
        notes.append("person is small in the frame (under 25% of its height)")
    if len(needed) < 50:
        notes.append("shorter than one 48-frame window")
    print(f"{path.name}: {n_src} frames at {src_fps:.1f} fps -> {len(needed)} at {OUT_FPS} fps | person in {found:.0%} | "
          f"mean point confidence {conf:.2f}" + (" | WARNING: " + "; ".join(notes) if notes else ""))
    return out_csv


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="a video file or a folder of videos")
    ap.add_argument("--out", default="keypoints_out", help="output folder")
    ap.add_argument("--weights", default="yolo11n-pose.pt", help="ultralytics pose weights")
    ap.add_argument("--device", default=None, help="e.g. cpu or 0 (first GPU); default = automatic")
    ap.add_argument("--preview", action="store_true", help="also write an overlay video to check the skeleton")
    args = ap.parse_args(argv)

    src = Path(args.input)
    videos = sorted(p for p in (src.rglob("*") if src.is_dir() else [src]) if p.suffix.lower() in VIDEO_EXT)
    if not videos:
        sys.exit(f"No videos found in {src}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    detector = YoloPose(args.weights, device=args.device)
    for v in videos:
        try:
            process(v, out_dir, detector, preview=args.preview)
        except Exception as e:                      # keep going if one clip is broken
            print(f"{v.name}: FAILED - {e}")


if __name__ == "__main__":
    main()
