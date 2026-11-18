"""Resolve model artifacts from Render's Hugging Face repos or local repo models."""

import os
from functools import lru_cache
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_MODEL_DIRS = {
    "imu": REPO_ROOT / "models" / "anomaly",
    "keypoint": REPO_ROOT / "models" / "keypoint",
}
REPO_ENV_VARS = {"imu": "HF_IMU_REPO", "keypoint": "HF_KEYPOINT_REPO"}
REQUIRED_FILES = ("model.onnx", "scaler.npz", "config.json")


def _has_model_files(directory: Path) -> bool:
    return all((directory / filename).is_file() for filename in REQUIRED_FILES)


def model_is_available(kind: str) -> bool:
    """Readiness check that never makes a network request."""
    if kind not in LOCAL_MODEL_DIRS:
        return False
    return bool(os.getenv(REPO_ENV_VARS[kind])) or _has_model_files(LOCAL_MODEL_DIRS[kind])


@lru_cache(maxsize=2)
def get_model_directory(kind: str = "imu") -> str:
    if kind not in LOCAL_MODEL_DIRS:
        raise ValueError(f"Unknown model kind {kind!r}; choose one of {sorted(LOCAL_MODEL_DIRS)}")

    repo_id = os.getenv(REPO_ENV_VARS[kind])
    if repo_id:
        return snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            token=os.getenv("HF_TOKEN") or None,
            local_files_only=False,
        )

    local_dir = LOCAL_MODEL_DIRS[kind]
    if _has_model_files(local_dir):
        return str(local_dir)
    raise RuntimeError(
        f"{REPO_ENV_VARS[kind]} is not configured and local model files were not found in {local_dir}. "
        "Configure the Render Hugging Face repo or provide model.onnx, scaler.npz and config.json locally."
    )


if __name__ == "__main__":
    print("IMU model directory:", get_model_directory("imu"))
    print("Keypoint model directory:", get_model_directory("keypoint"))
