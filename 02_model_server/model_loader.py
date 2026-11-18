"""Resolve model artifacts from Render's Hugging Face repos or local repo models."""

import json
import math
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
# Optional folder inside the HF repo (e.g. "keypoint"). When unset, a folder named after the
# model kind is used if it holds the model files, otherwise the repo root.
SUBFOLDER_ENV_VARS = {"imu": "HF_IMU_SUBFOLDER", "keypoint": "HF_KEYPOINT_SUBFOLDER"}
REQUIRED_FILES = ("model.onnx", "scaler.npz", "config.json")
# Keys accepted for the validation-derived threshold, in priority order.
THRESHOLD_KEYS = ("threshold_mse", "anomaly_threshold")


def _has_model_files(directory: Path) -> bool:
    return all((directory / filename).is_file() for filename in REQUIRED_FILES)


def model_is_available(kind: str) -> bool:
    """Cheap check that model files are configured; does not prove they load."""
    if kind not in LOCAL_MODEL_DIRS:
        return False
    return bool(os.getenv(REPO_ENV_VARS[kind])) or _has_model_files(LOCAL_MODEL_DIRS[kind])


def read_threshold(config: dict, config_path: Path | str = "config.json") -> float:
    """Return the trained anomaly threshold from a model config, or raise a descriptive error."""
    key = next((k for k in THRESHOLD_KEYS if config.get(k) is not None), None)
    if key is None:
        raise ValueError(
            f"{config_path} does not contain the trained anomaly threshold: expected a top-level "
            f"numeric key {' or '.join(repr(k) for k in THRESHOLD_KEYS)}; found keys {sorted(config)}"
        )
    try:
        threshold = float(config[key])
    except (TypeError, ValueError):
        raise ValueError(f"{config_path} key {key!r} is not numeric: {config[key]!r}") from None
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError(f"{config_path} key {key!r} must be a positive finite number, got {threshold}")
    return threshold


def _resolve_subfolder(root: Path, kind: str) -> Path:
    subfolder = os.getenv(SUBFOLDER_ENV_VARS[kind], "").strip("/")
    if subfolder:
        directory = root / subfolder
    elif _has_model_files(root / kind):
        directory = root / kind
    else:
        directory = root
    missing = [f for f in REQUIRED_FILES if not (directory / f).is_file()]
    if missing:
        raise RuntimeError(
            f"Model folder {directory} is missing {', '.join(missing)}. "
            f"Set {SUBFOLDER_ENV_VARS[kind]} if the files live in a subfolder of the Hugging Face repo."
        )
    return directory


@lru_cache(maxsize=2)
def get_model_directory(kind: str = "imu") -> str:
    if kind not in LOCAL_MODEL_DIRS:
        raise ValueError(f"Unknown model kind {kind!r}; choose one of {sorted(LOCAL_MODEL_DIRS)}")

    repo_id = os.getenv(REPO_ENV_VARS[kind])
    if repo_id:
        root = snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            token=os.getenv("HF_TOKEN") or None,
            local_files_only=False,
        )
        return str(_resolve_subfolder(Path(root), kind))

    local_dir = LOCAL_MODEL_DIRS[kind]
    if _has_model_files(local_dir):
        return str(local_dir)
    raise RuntimeError(
        f"{REPO_ENV_VARS[kind]} is not configured and local model files were not found in {local_dir}. "
        "Configure the Render Hugging Face repo or provide model.onnx, scaler.npz and config.json locally."
    )


if __name__ == "__main__":
    for kind in LOCAL_MODEL_DIRS:
        directory = Path(get_model_directory(kind))
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        print(f"{kind}: {directory} | threshold {read_threshold(config, directory / 'config.json')}")
