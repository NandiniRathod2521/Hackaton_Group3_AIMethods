"""FastAPI model endpoints and a Gradio CSV interface for IMU/keypoint models."""

from __future__ import annotations

import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import List

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from model_loader import get_model_directory, model_is_available

ROOT = Path(__file__).resolve().parents[1]
KEYPOINT_HELPER = Path(__file__).resolve().parent / "keypoint_utils.py"
sys.path.insert(0, str(KEYPOINT_HELPER.parent))
import keypoint_utils as ku


class IMURequest(BaseModel):
    windows: List[List[List[float]]] = Field(
        description="Batch of raw IMU windows [batch, time, channels]"
    )


class KeypointRequest(BaseModel):
    frames: List[List[List[float]]] = Field(
        description="Raw pose sequence [frames, 17 COCO joints, (x, y, confidence)]"
    )


@lru_cache(maxsize=2)
def load_model(kind: str) -> dict:
    """Load the configured ONNX model, scaler and threshold on first use."""
    if kind not in {"imu", "keypoint"}:
        raise ValueError(f"Unsupported model kind: {kind}")
    directory = Path(get_model_directory(kind))
    try:
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        with np.load(directory / "scaler.npz") as scaler_file:
            mean = np.asarray(scaler_file["mean"], dtype=np.float32)
            scale = np.asarray(scaler_file["scale"], dtype=np.float32)
        threshold = config.get("threshold_mse", config.get("anomaly_threshold"))
        if threshold is None:
            raise ValueError("config.json does not contain the trained anomaly threshold")
        session = ort.InferenceSession(
            str(directory / "model.onnx"), providers=["CPUExecutionProvider"]
        )
        input_name = session.get_inputs()[0].name
        window = int(config.get("window", config.get("window_frames", 0)))
        if window <= 0:
            raise ValueError("config.json does not contain a valid window length")
        if kind == "imu":
            features = config.get("features")
            if not features or mean.shape != (len(features),) or scale.shape != mean.shape:
                raise ValueError("IMU feature order and scaler dimensions do not match")
            stride = int(config.get("evaluation_stride", window))
            sample_rate = float(config.get("sampling_rate_hz", 100))
            pmin = None
        else:
            features = ku.COCO_POINT_NAMES
            if mean.shape != (ku.N_FEAT,) or scale.shape != mean.shape:
                raise ValueError("Keypoint scaler must contain 34 values in helper feature order")
            stride = window
            sample_rate = float(config.get("fps", ku.FPS))
            pmin = float(config.get("pmin", config.get("preprocessing", {}).get("pmin", ku.PMIN)))
        return {
            "kind": kind, "directory": directory, "config": config, "session": session,
            "input_name": input_name, "mean": mean, "scale": scale,
            "threshold": float(threshold), "window": window, "stride": stride,
            "sample_rate": sample_rate, "features": features, "pmin": pmin,
        }
    except Exception as exc:
        raise RuntimeError(f"Could not load the {kind} model from {directory}: {exc}") from exc


def _score_standardized_windows(windows: np.ndarray, model: dict) -> np.ndarray:
    rebuilt = model["session"].run(
        ["reconstruction"], {model["input_name"]: windows}
    )[0]
    return np.mean(np.square(rebuilt - windows), axis=(1, 2)).astype(float)


def _prediction(scores: np.ndarray, model: dict, starts: np.ndarray) -> dict:
    threshold = model["threshold"]
    flagged = scores > threshold
    return {
        "model": model["config"].get("model", model["kind"]),
        "windows_analyzed": int(len(scores)),
        "threshold": threshold,
        "windows_above_threshold": int(flagged.sum()),
        "percentage_above_threshold": round(float(flagged.mean() * 100), 2),
        "interpretation": "Scores describe movement deviations from the model's reference data; they are not a diagnosis or safety certification.",
        "windows": [
            {
                "window": int(i + 1),
                "start_sample": int(starts[i]),
                "start_seconds": round(float(starts[i] / model["sample_rate"]), 3),
                "end_seconds": round(float((starts[i] + model["window"]) / model["sample_rate"]), 3),
                "reconstruction_score": float(scores[i]),
                "above_threshold": bool(flagged[i]),
            }
            for i in range(len(scores))
        ],
    }


def score_imu_windows(raw_windows: np.ndarray, model: dict, starts: np.ndarray | None = None) -> dict:
    """Score raw windows in config feature order using the saved training scaler."""
    raw = np.asarray(raw_windows, dtype=np.float32)
    expected = (model["window"], len(model["features"]))
    if raw.ndim != 3 or raw.shape[1:] != expected or raw.shape[0] == 0:
        raise HTTPException(status_code=422, detail=f"Expected a non-empty batch shaped [batch, {expected[0]}, {expected[1]}].")
    if not np.isfinite(raw).all():
        raise HTTPException(status_code=422, detail="IMU values contain missing or non-numeric data.")
    standardized = ((raw - model["mean"]) / model["scale"]).astype(np.float32)
    scores = _score_standardized_windows(standardized, model)
    if starts is None:
        starts = np.arange(len(raw), dtype=int) * model["stride"]
    return _prediction(scores, model, np.asarray(starts, dtype=int))


def _imu_csv(filepath: str) -> tuple[dict, pd.DataFrame]:
    if not filepath:
        raise gr.Error("Upload an IMU CSV file first.")
    try:
        frame = pd.read_csv(filepath)
    except (OSError, UnicodeError, pd.errors.ParserError, ValueError) as exc:
        raise gr.Error(f"Could not read this CSV: {exc}") from exc
    try:
        model = load_model("imu")
    except RuntimeError as exc:
        raise gr.Error(str(exc)) from exc
    missing = [name for name in model["features"] if name not in frame.columns]
    if missing:
        raise gr.Error("CSV is missing required IMU columns (in model order): " + ", ".join(missing))
    values = frame[model["features"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        bad = np.argwhere(~np.isfinite(values))[0]
        raise gr.Error(
            f"The IMU CSV has missing or non-numeric data in row {int(bad[0]) + 1}, "
            f"column {model['features'][int(bad[1])]}. Correct the file before analysis."
        )
    if len(values) < model["window"]:
        raise gr.Error(f"This recording has {len(values)} samples; at least {model['window']} are needed for one window.")
    starts = np.arange(0, len(values) - model["window"] + 1, model["stride"], dtype=int)
    windows = np.stack([values[s:s + model["window"]] for s in starts])
    result = score_imu_windows(windows, model, starts)
    table = pd.DataFrame(result["windows"])
    return result, table


def _posture_summary(features: np.ndarray, valid: np.ndarray) -> dict:
    angles = ku.joint_angles(features)
    visible = valid[:, [5, 6, 11, 12]].all(axis=1)
    if visible.sum() < 10:
        visible = np.ones(len(features), dtype=bool)
    trunk = angles["trunk_lean"]
    knee = 180 - np.minimum(angles["knee_left"], angles["knee_right"])
    arm = np.maximum(angles["shoulder_left"], angles["shoulder_right"])
    return {
        "median trunk lean (degrees)": round(float(np.median(trunk[visible])), 1),
        "median knee bend (degrees)": round(float(np.median(knee[visible])), 1),
        "median arm elevation (degrees)": round(float(np.median(arm[visible])), 1),
        "frames with shoulders/hips visible": round(float(visible.mean() * 100), 1),
        "note": "Approximate 2D posture indicators from keypoints; not fatigue, danger, or validated worker strain.",
    }


def score_keypoint_frames(raw_frames: np.ndarray, model: dict) -> dict:
    """Apply the trained keypoint cleaner, scaler, windowing and ONNX model."""
    raw = np.asarray(raw_frames, dtype=np.float64)
    if raw.ndim != 3 or raw.shape[1:] != (ku.N_KP, 3):
        raise HTTPException(status_code=422, detail="Expected frames shaped [frames, 17 COCO points, 3 values (x, y, confidence)].")
    if len(raw) < ku.WINDOW:
        raise HTTPException(status_code=422, detail=f"This recording has {len(raw)} frames; at least {ku.WINDOW} are needed for one window.")
    confidence = raw[:, :, 2]
    if not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 1)):
        raise HTTPException(status_code=422, detail="Keypoint confidence values must be finite and between 0 and 1.")
    features, valid = ku.clean_clip(raw[:, :, 0], raw[:, :, 1], confidence, pmin=model["pmin"])
    # starts = each kept window's original first frame (windows with too many missing points are dropped)
    windows, starts = ku.preprocess_for_model_with_starts(
        raw[:, :, 0], raw[:, :, 1], confidence, model["mean"], model["scale"], pmin=model["pmin"]
    )
    if not len(windows):
        raise HTTPException(status_code=422, detail="No valid keypoint windows remain. The clip may be too short or have more than 30% missing points in every window.")
    scores = _score_standardized_windows(windows, model)
    result = _prediction(scores, model, starts)
    result["posture_summary"] = _posture_summary(features, valid)
    return result


def _keypoint_csv(filepath: str) -> tuple[dict, pd.DataFrame]:
    if not filepath:
        raise gr.Error("Upload a keypoint CSV file first.")
    try:
        x, y, confidence = ku.read_keypoints(filepath)
    except (OSError, UnicodeError, pd.errors.ParserError, ValueError, TypeError) as exc:
        raise gr.Error(f"Could not read the expected keypoint CSV: {exc}") from exc
    try:
        model = load_model("keypoint")
    except RuntimeError as exc:
        raise gr.Error(str(exc)) from exc
    if len(x) < ku.WINDOW:
        raise gr.Error(f"This recording has {len(x)} frames; at least {ku.WINDOW} are needed for one window.")
    try:
        raw = np.stack([x, y, confidence], axis=-1)
        result = score_keypoint_frames(raw, model)
    except HTTPException as exc:
        raise gr.Error(str(exc.detail)) from exc
    return result, pd.DataFrame(result["windows"])


def _result_markdown(result: dict) -> str:
    return (
        f"**Windows analyzed:** {result['windows_analyzed']}  \n"
        f"**Configured threshold:** {result['threshold']:.6g} standardized reconstruction MSE  \n"
        f"**Above threshold:** {result['windows_above_threshold']} / {result['windows_analyzed']} "
        f"({result['percentage_above_threshold']:.2f}%)  \n\n"
        f"{result['interpretation']}"
    )


def _score_plot(table: pd.DataFrame, threshold: float, title: str):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(table["start_seconds"], table["reconstruction_score"], marker="o", linewidth=1.5)
    ax.axhline(threshold, color="crimson", linestyle="--", label=f"threshold {threshold:.5g}")
    ax.set(title=title, xlabel="Window start (seconds)", ylabel="Standardized reconstruction MSE")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def _analyze_imu_ui(filepath: str):
    result, table = _imu_csv(filepath)
    model = load_model("imu")
    return _result_markdown(result), table, _score_plot(table, result["threshold"], "IMU movement deviation over time")


def _analyze_keypoint_ui(filepath: str):
    result, table = _keypoint_csv(filepath)
    posture = result["posture_summary"]
    posture_md = "\n".join([
        "**Approximate 2D posture summary**",
        f"- Median trunk lean: {posture['median trunk lean (degrees)']}°",
        f"- Median knee bend: {posture['median knee bend (degrees)']}°",
        f"- Median arm elevation: {posture['median arm elevation (degrees)']}°",
        f"- Frames with shoulders/hips visible: {posture['frames with shoulders/hips visible']}%",
        f"\n_{posture['note']}_",
    ])
    result_md = _result_markdown(result) + "\n\n" + posture_md
    return result_md, table, _score_plot(table, result["threshold"], "Keypoint movement deviation over time")


app = FastAPI(
    title="Construction Worker Movement Analysis API",
    version="1.1.0",
    description=(
        "Experimental movement-deviation scoring. Scores compare uploaded movement with the selected models' "
        "reference data; they are not diagnoses, fatigue measurements, or safety certifications."
    ),
)


@app.get("/health")
def health():
    return {
        "status": "running",
        "models": {
            "imu": model_is_available("imu"),
            "keypoint": model_is_available("keypoint"),
        },
    }


@app.post("/predict/imu")
def predict_imu(request: IMURequest):
    try:
        model = load_model("imu")
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return score_imu_windows(np.asarray(request.windows, dtype=np.float32), model)


@app.post("/predict/keypoints")
def predict_keypoints(request: KeypointRequest):
    try:
        model = load_model("keypoint")
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return score_keypoint_frames(np.asarray(request.frames, dtype=np.float64), model)


with gr.Blocks(title="Movement deviation analysis") as demo:
    gr.Markdown(
        "# Movement deviation analysis\n"
        "Upload an IMU or keypoint CSV and compare its windows with the selected model's reference data. "
        "Scores are experimental movement-deviation indicators, not diagnosis, fatigue detection, or worker-safety certification. "
        "**Raw video is not accepted:** convert video locally to the expected keypoint CSV first using "
        "`01_autoencoder/video_to_keypoints.py`."
    )
    with gr.Tab("IMU CSV analysis"):
        imu_file = gr.File(label="IMU CSV", file_types=[".csv"], type="filepath")
        imu_button = gr.Button("Analyse", variant="primary")
        imu_summary = gr.Markdown()
        imu_table = gr.Dataframe(label="Per-window reconstruction scores", interactive=False)
        imu_chart = gr.Plot(label="Score over time")
        imu_button.click(_analyze_imu_ui, inputs=imu_file, outputs=[imu_summary, imu_table, imu_chart], api_name="analyze_imu", api_visibility="public")
    with gr.Tab("Keypoint CSV analysis"):
        keypoint_file = gr.File(label="Keypoint CSV", file_types=[".csv"], type="filepath")
        keypoint_button = gr.Button("Analyse", variant="primary")
        keypoint_summary = gr.Markdown()
        keypoint_table = gr.Dataframe(label="Per-window reconstruction scores", interactive=False)
        keypoint_chart = gr.Plot(label="Score over time")
        keypoint_button.click(_analyze_keypoint_ui, inputs=keypoint_file, outputs=[keypoint_summary, keypoint_table, keypoint_chart], api_name="analyze_keypoints", api_visibility="public")


# The Gradio mount is registered after the explicit FastAPI routes, so /health,
# /predict/* and /docs retain precedence while the interface serves the homepage.
app = gr.mount_gradio_app(app, demo, path="/")
