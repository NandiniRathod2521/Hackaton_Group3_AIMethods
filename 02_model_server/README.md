# Movement deviation model server

FastAPI serves the IMU and video-keypoint ONNX models. A Gradio interface is mounted at the service homepage for CSV uploads. The scores measure movement deviations from each model's reference data; they are experimental indicators, not diagnoses, fatigue measurements, or worker-safety certifications.

## Run locally

From the repository root, install the server requirements in the Python environment you intend to use, then start the same ASGI application used by Render:

```powershell
python -m pip install -r 02_model_server/requirements.txt
python -m uvicorn app:app --app-dir 02_model_server --reload --host 127.0.0.1 --port 8000
```

Open the interface at `http://127.0.0.1:8000/`. FastAPI docs are at `http://127.0.0.1:8000/docs`, and health is at `http://127.0.0.1:8000/health`.

For local development, the loader uses `models/anomaly/` for IMU and `models/keypoint/` for keypoints when each directory contains `model.onnx`, `scaler.npz`, and `config.json`. Model files are not committed. Alternatively, set `HF_IMU_REPO` and `HF_KEYPOINT_REPO` to Hugging Face model repository IDs and `HF_TOKEN` if either repository is private.

## Interface and inputs

The homepage has two modes:

- **IMU CSV analysis:** provide the configured IMU feature columns. The server uses the exact order in the loaded model's `config.json`, applies its saved scaler, and creates full windows at the model's configured evaluation stride.
- **Keypoint CSV analysis:** provide the dataset or converter format with `frame_number`, `timestamp`, `x0/y0/prob0` through `x16/y16/prob16`, and `detection_score`. The service uses `keypoint_utils.py` for COCO ordering, confidence filtering, interpolation, hip-centering, torso-length normalization, clipping, saved scaling, and non-overlapping 48-frame windows. It includes the available approximate 2D posture summary.

Each mode shows per-window reconstruction scores, the configured threshold, count and percentage above threshold, and a score-over-time chart. Short clips, missing required columns, and non-numeric IMU data receive a validation message. Keypoint CSVs with no valid windows receive a confidence/windowing message.

Raw video upload is not supported. Convert video locally to the expected keypoint CSV first with `01_autoencoder/video_to_keypoints.py` (which requires the user's local video-processing dependencies and pose weights).

## API

- `GET /health`
- `POST /predict/imu` with `{"windows": [[[...]] ]}` containing raw windows in the saved feature order and shape `[batch, window, channels]`.
- `POST /predict/keypoints` with `{"frames": [[[x, y, confidence], ...] ]}` containing raw frames shaped `[frames, 17, 3]` in COCO point order.
- `GET /docs` for the interactive FastAPI schema.

The prediction responses include each reconstruction score and whether it exceeds the threshold stored with the model. Keypoint responses also include approximate 2D posture indicators.

Each returned window has `start_sample`, `start_seconds` and `end_seconds` for its original position in the submitted recording. Keypoint windows with more than 30% missing points are dropped, so the `window` numbers are consecutive but the start times may have gaps. For example, if frames 0–47 are dropped, the first returned window starts at frame 48 (1.92 s). Run `python -m pytest 02_model_server/test_window_starts.py` to check this with the local model files. Thresholds are model-specific and are never recalculated from an uploaded file.

## Render

The Blueprint in `render.yaml` keeps `rootDir: 02_model_server`, installs `requirements.txt`, starts `uvicorn app:app --host 0.0.0.0 --port $PORT`, and checks `/health`. Set `HF_IMU_REPO` and `HF_KEYPOINT_REPO` to repositories containing each model's `model.onnx`, `scaler.npz`, and `config.json`; set `HF_TOKEN` when private repositories require it.

After deployment, open `https://<service-name>.onrender.com/` for the interface and `https://<service-name>.onrender.com/docs` for API docs. The health endpoint is `https://<service-name>.onrender.com/health`.

## Interpretation limits

“Above threshold” means that a window's reconstruction error exceeded the configured threshold for that model. For keypoints, the selected comparison tasks are movements different from the model's selected normal reference data. Approximate 2D posture angles do not establish fatigue, danger, or validated worker strain. Do not use these outputs as a diagnosis or safety certification.
