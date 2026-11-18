"""Shared keypoint preprocessing for the VTT-ConIoT video keypoints.

Used by the training notebook now and by the model server / app later, so that
training and deployment clean the keypoints in exactly the same way.

Input per frame: 17 COCO points (x, y, prob). Output per frame: 34 numbers
(x and y of every point, relative to the hip centre and divided by the torso
length), so camera distance and position in the picture no longer matter.

COCO order: 0 nose, 1 left eye, 2 right eye, 3 left ear, 4 right ear,
5 left shoulder, 6 right shoulder, 7 left elbow, 8 right elbow,
9 left wrist, 10 right wrist, 11 left hip, 12 right hip,
13 left knee, 14 right knee, 15 left ankle, 16 right ankle.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

N_KP = 17
N_FEAT = N_KP * 2
COCO_POINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
KEYPOINT_COLUMNS = ["frame_number", "timestamp"] + [
    f"{kind}{i}" for i in range(N_KP) for kind in ("x", "y", "prob")
] + ["detection_score"]
FPS = 25
WINDOW = 48            # frames = 1.92 s
STRIDE_TRAIN = 24
PMIN = 0.02            # points below this probability count as missing
MAX_GAP = 25           # interpolate gaps up to 1 s
CLIP = 6.0             # clip normalised coordinates (torso lengths)
MAX_MISSING = 0.30     # skip a window if more than 30 % of its points are missing
SCALE_FLOOR = 0.05

NORMAL = [13, 14, 15]
ANOMALOUS = [7, 8, 11, 12]
TRAIN_USERS, VAL_USERS, TEST_USERS = list(range(1, 10)), [10, 11], [12, 13]

FILE_RE = re.compile(r"Subject_(\d+)_Task_(\d+)\.")


def read_keypoints(path):
    """Read one keypoint CSV -> x, y, prob arrays of shape (frames, 17)."""
    d = pd.read_csv(path)
    missing = [c for c in KEYPOINT_COLUMNS if c not in d.columns]
    if missing:
        raise ValueError(f"{path} is missing expected keypoint columns: {missing}")
    x = d[[f"x{i}" for i in range(N_KP)]].to_numpy(dtype=np.float64)
    y = d[[f"y{i}" for i in range(N_KP)]].to_numpy(dtype=np.float64)
    p = d[[f"prob{i}" for i in range(N_KP)]].to_numpy(dtype=np.float64)
    return x, y, p


def _interp(a):
    return pd.DataFrame(a).interpolate(limit_direction="both", limit=MAX_GAP).to_numpy()


def clean_clip(x, y, p, pmin=PMIN):
    """Raw pixel keypoints -> (features (T, 34) float32, valid (T, 17) bool)."""
    valid = p >= pmin
    x = np.where(valid, x, np.nan)
    y = np.where(valid, y, np.nan)
    x, y = np.array(_interp(x)), np.array(_interp(y))
    # points still missing: use that point's median over the clip, else 0
    for a in (x, y):
        with np.errstate(all="ignore"):
            med = np.nanmedian(a, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        a[:] = np.where(np.isnan(a), med[None, :], a)
    hx, hy = x[:, [11, 12]].mean(1), y[:, [11, 12]].mean(1)
    sx, sy = x[:, [5, 6]].mean(1), y[:, [5, 6]].mean(1)
    torso = np.hypot(sx - hx, sy - hy)
    core = valid[:, [5, 6, 11, 12]].all(axis=1)
    t = np.median(torso[core]) if core.any() else np.median(torso)
    t = t if np.isfinite(t) and t > 1.0 else 100.0
    xn = np.clip((x - hx[:, None]) / t, -CLIP, CLIP)
    yn = np.clip((y - hy[:, None]) / t, -CLIP, CLIP)
    feats = np.stack([xn, yn], axis=-1).reshape(len(x), N_FEAT).astype(np.float32)
    return feats, valid


def find_files(data_dir):
    out = []
    for path in sorted(Path(data_dir).rglob("Subject_*_Task_*.csv")):
        m = FILE_RE.search(path.name)
        if m:
            out.append((int(m.group(1)), int(m.group(2)), path))
    return out


def load_records(data_dir, tasks, pmin=PMIN):
    """List of (user, task, features, valid) for the wanted tasks."""
    records = []
    for user, task, path in find_files(data_dir):
        if task in tasks:
            feats, valid = clean_clip(*read_keypoints(path), pmin=pmin)
            records.append((user, task, feats, valid))
    return records


def make_windows(records, users, tasks, stride, window=WINDOW, max_missing=MAX_MISSING):
    """Cut clips into windows. Returns (X (n, window, 34), meta (n, 2) = user, task)."""
    X, meta = [], []
    for user, task, feats, valid in records:
        if user not in users or task not in tasks:
            continue
        for s in range(0, len(feats) - window + 1, stride):
            if valid[s:s + window].mean() < 1 - max_missing:
                continue
            X.append(feats[s:s + window])
            meta.append((user, task))
    if not X:
        return np.zeros((0, window, N_FEAT), np.float32), np.zeros((0, 2), int)
    return np.stack(X).astype(np.float32), np.array(meta)


def build_splits(data_dir, pmin=PMIN):
    """Everything the notebook needs. Scaler is fitted on normal training windows only."""
    records = load_records(data_dir, NORMAL + ANOMALOUS, pmin=pmin)
    train_raw, train_meta = make_windows(records, TRAIN_USERS, NORMAL, STRIDE_TRAIN)
    val_raw, val_meta = make_windows(records, VAL_USERS, NORMAL, STRIDE_TRAIN)
    test_raw, test_meta = make_windows(records, TEST_USERS, NORMAL + ANOMALOUS, WINDOW)
    rows = train_raw.reshape(-1, N_FEAT)
    mean = rows.mean(axis=0)
    scale = np.maximum(rows.std(axis=0), SCALE_FLOOR)
    std = lambda a: ((a - mean) / scale).astype(np.float32)
    return {
        "train_x": std(train_raw), "val_x": std(val_raw), "test_x": std(test_raw),
        "val_meta": val_meta, "test_meta": test_meta,
        "train_meta": train_meta,
        "test_label": np.isin(test_meta[:, 1], ANOMALOUS).astype(int),
        "mean": mean, "scale": scale, "records": records,
        "pmin": pmin,
    }


def preprocess_for_model_with_starts(x, y, p, mean, scale, pmin=PMIN):
    """Like preprocess_for_model, but also returns each kept window's original start frame.

    Windows with too many missing points are dropped, so the i-th returned window does not
    necessarily start at frame i * WINDOW; `starts[i]` is its true first frame in the clip.
    """
    feats, valid = clean_clip(x, y, p, pmin=pmin)
    starts = [s for s in range(0, len(feats) - WINDOW + 1, WINDOW)
              if valid[s:s + WINDOW].mean() >= 1 - MAX_MISSING]
    if not starts:
        return np.zeros((0, WINDOW, N_FEAT), np.float32), np.zeros(0, dtype=int)
    wins = np.stack([(feats[s:s + WINDOW] - mean) / scale for s in starts]).astype(np.float32)
    return wins, np.asarray(starts, dtype=int)


def preprocess_for_model(x, y, p, mean, scale, pmin=PMIN):
    """Deployment path: raw keypoints of ONE clip -> standardised non-overlapping windows.

    Uses the same point cleaning, confidence threshold, centering, torso scaling,
    clipping, and train-fitted standardization as build_splits. The overlapping
    train/validation stride is training-time sampling; deployment is non-overlapping.
    """
    return preprocess_for_model_with_starts(x, y, p, mean, scale, pmin=pmin)[0]


def _angle(a, b, c):
    """Angle at b (degrees) between points a-b-c; arrays (T, 2)."""
    v1, v2 = a - b, c - b
    cos = (v1 * v2).sum(-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def joint_angles(feats):
    """Posture angles per frame from clean_clip features (T, 34) -> dict of (T,) arrays.

    Image y points down. Trunk lean is measured from vertical: 0 = upright.
    Knee 180 = straight leg. Shoulder elevation: angle between trunk and upper arm
    (about 20 = arm hanging, 90 = arm horizontal, above 90 = arm raised overhead).
    These feed the strain rules in the app (REBA-style thresholds).
    """
    P = feats.reshape(len(feats), N_KP, 2)
    hip, sho = P[:, [11, 12]].mean(1), P[:, [5, 6]].mean(1)
    trunk = sho - hip
    lean = np.degrees(np.arctan2(np.abs(trunk[:, 0]), -trunk[:, 1]))
    lean = np.where(trunk[:, 1] > 0, 180 - lean, lean)    # upside down = bent far over
    return {
        "trunk_lean": lean,
        "knee_left": _angle(P[:, 11], P[:, 13], P[:, 15]),
        "knee_right": _angle(P[:, 12], P[:, 14], P[:, 16]),
        "elbow_left": _angle(P[:, 5], P[:, 7], P[:, 9]),
        "elbow_right": _angle(P[:, 6], P[:, 8], P[:, 10]),
        "shoulder_left": _angle(P[:, 11], P[:, 5], P[:, 7]),
        "shoulder_right": _angle(P[:, 12], P[:, 6], P[:, 8]),
    }
