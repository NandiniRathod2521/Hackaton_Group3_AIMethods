"""Keypoint windows must report their original start frame/time, even when earlier windows are dropped.

Runs the real FastAPI app with the local model files (models/keypoint/, models/anomaly/).
    python -m pytest 02_model_server/test_window_starts.py -q
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
os.environ.pop("HF_KEYPOINT_REPO", None)
os.environ.pop("HF_IMU_REPO", None)

if not all((REPO / "models" / "keypoint" / f).is_file() for f in ("model.onnx", "scaler.npz", "config.json")):
    pytest.skip("local keypoint model files are not available", allow_module_level=True)

from fastapi.testclient import TestClient  # noqa: E402

import app as server  # noqa: E402
import keypoint_utils as ku  # noqa: E402

client = TestClient(server.app)
YOLO_CLIP = REPO / "keypoints_out" / "WhatsApp Video 2026-11-17 at 4.03.52 PM.keypoints.csv"


def walking_clip(n_windows: int) -> np.ndarray:
    """A plausible upright skeleton with small motion, fully visible: [frames, 17, 3]."""
    base = np.array([[300, 200], [290, 190], [310, 190], [280, 195], [320, 195], [260, 260], [340, 260],
                     [250, 330], [350, 330], [245, 400], [355, 400], [275, 420], [325, 420],
                     [275, 520], [325, 520], [275, 620], [325, 620]], dtype=float)
    t = np.arange(n_windows * ku.WINDOW)
    frames = np.repeat(base[None], len(t), axis=0)
    frames[:, 13:, 0] += 15 * np.sin(t / 4)[:, None]
    conf = np.full((len(t), 17, 1), 0.9)
    return np.concatenate([frames, conf], axis=2)


def post(frames: np.ndarray) -> dict:
    r = client.post("/predict/keypoints", json={"frames": frames.tolist()})
    assert r.status_code == 200, r.text
    return r.json()


def blank(frames: np.ndarray, window: int) -> np.ndarray:
    out = frames.copy()
    out[window * ku.WINDOW:(window + 1) * ku.WINDOW] = 0.0     # nobody detected -> window dropped
    return out


def starts(result: dict) -> list:
    return [w["start_sample"] for w in result["windows"]]


def test_no_dropped_windows_keeps_consecutive_starts():
    res = post(walking_clip(4))
    assert starts(res) == [0, 48, 96, 144]
    assert [w["start_seconds"] for w in res["windows"]] == [0.0, 1.92, 3.84, 5.76]
    assert [w["end_seconds"] for w in res["windows"]] == [1.92, 3.84, 5.76, 7.68]


def test_first_window_dropped():
    res = post(blank(walking_clip(4), 0))
    assert starts(res) == [48, 96, 144]
    assert res["windows"][0]["start_seconds"] == 1.92


def test_middle_window_dropped():
    res = post(blank(walking_clip(5), 2))
    assert starts(res) == [0, 48, 144, 192]
    assert [w["window"] for w in res["windows"]] == [1, 2, 3, 4]


def test_scores_unchanged_by_start_fix():
    frames = blank(walking_clip(5), 1)
    model = server.load_model("keypoint")
    old_windows = ku.preprocess_for_model(frames[..., 0], frames[..., 1], frames[..., 2],
                                          model["mean"], model["scale"], pmin=model["pmin"])
    expected = server._score_standardized_windows(old_windows, model)
    got = [w["reconstruction_score"] for w in post(frames)["windows"]]
    np.testing.assert_allclose(got, expected, rtol=1e-6)


@pytest.mark.skipif(not YOLO_CLIP.is_file(), reason="sample keypoint CSV not available")
def test_real_clip_with_dropped_first_window():
    x, y, p = ku.read_keypoints(YOLO_CLIP)
    res = post(np.stack([x, y, p], axis=-1))
    assert starts(res) == [48]                      # first 48 frames have too few visible points
    assert res["windows"][0]["start_seconds"] == 1.92


def test_imu_response_has_end_seconds():
    model = server.load_model("imu")
    windows = np.zeros((2, model["window"], len(model["features"])))
    r = client.post("/predict/imu", json={"windows": windows.tolist()})
    assert r.status_code == 200, r.text
    w = r.json()["windows"]
    assert [v["start_sample"] for v in w] == [0, model["stride"]]
    assert w[0]["end_seconds"] == round(model["window"] / model["sample_rate"], 3)
