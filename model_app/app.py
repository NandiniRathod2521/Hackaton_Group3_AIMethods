import json
import os
from pathlib import Path

import gradio as gr
import numpy as np
import onnxruntime as ort
import pandas as pd

HERE = Path(__file__).parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
FEATURES = CONFIG["features"]
WINDOW = CONFIG["window"]
SAMPLE_RATE = CONFIG["sampling_rate_hz"]

scaler = np.load(HERE / "scaler.npz")
MEAN = scaler["mean"].astype(np.float32)
SCALE = scaler["scale"].astype(np.float32)

session = ort.InferenceSession(
    str(HERE / "model.onnx"),
    providers=["CPUExecutionProvider"],
)
INPUT_NAME = session.get_inputs()[0].name


def run_window(sensor_window):
    try:
        raw = np.asarray(sensor_window, dtype=np.float32)
    except (TypeError, ValueError):
        raise gr.Error("Input must be a numeric 200 × 18 sensor window.")

    expected_shape = (WINDOW, len(FEATURES))
    if raw.shape != expected_shape:
        raise gr.Error(
            f"Expected shape {expected_shape}; received {raw.shape}."
        )
    if not np.isfinite(raw).all():
        raise gr.Error("Input contains missing or non-numeric values.")

    # Apply the same training-set scaling used to train the model.
    x = ((raw - MEAN) / SCALE)[None, ...].astype(np.float32)

    embedding, rebuilt_scaled = session.run(
        ["embedding", "reconstruction"],
        {INPUT_NAME: x},
    )

    mse = float(np.mean((rebuilt_scaled - x) ** 2))
    rebuilt_raw = rebuilt_scaled[0] * SCALE + MEAN

    return embedding[0], rebuilt_raw, mse


def embed(sensor_window: list[list[float]]) -> list[float]:
    """Return the latent embedding for one raw 200 × 18 sensor window."""
    embedding, _, _ = run_window(sensor_window)
    return embedding.tolist()


def predict(sensor_window: list[list[float]]) -> dict:
    """Return reconstruction error and rebuilt values for one sensor window."""
    _, rebuilt_raw, mse = run_window(sensor_window)
    return {
        "reconstruction_mse_standardized": round(mse, 6),
        "reconstruction_raw": rebuilt_raw.tolist(),
    }


def analyse_csv(filepath: str) -> dict:
    """For the page: analyze the first two seconds of an IMU CSV."""
    if not filepath:
        raise gr.Error("Upload an IMU CSV first.")

    frame = pd.read_csv(filepath)
    missing = [name for name in FEATURES if name not in frame.columns]
    if missing:
        raise gr.Error(f"CSV is missing required sensor columns: {missing}")
    if len(frame) < WINDOW:
        raise gr.Error(f"CSV needs at least {WINDOW} rows.")

    values = frame[FEATURES].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=np.float32)

    embedding, _, mse = run_window(values[:WINDOW])
    return {
        "file": Path(filepath).name,
        "samples_used": WINDOW,
        "seconds_used": WINDOW / SAMPLE_RATE,
        "reconstruction_mse_standardized": round(mse, 6),
        "embedding": embedding.tolist(),
    }


with gr.Blocks(title="Motion Atlas model server") as demo:
    gr.Markdown(
        "# Motion Atlas model server\n"
        "Upload an IMU CSV to analyze its first two seconds. "
        "The API accepts one raw sensor window with 200 rows and "
        f"{len(FEATURES)} columns, ordered as listed in `config.json`."
    )

    csv_input = gr.File(
        label="Upload an IMU CSV",
        file_types=[".csv"],
        type="filepath",
    )
    analyse_button = gr.Button("Analyze first two seconds", variant="primary")
    page_result = gr.JSON(label="Result")

    analyse_button.click(
        analyse_csv,
        inputs=csv_input,
        outputs=page_result,
        api_visibility="private",
    )

    gr.api(embed, api_name="embed")
    gr.api(predict, api_name="predict")


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )