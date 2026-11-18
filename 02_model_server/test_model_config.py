"""Model config/threshold validation, HF subfolder resolution, and /health load reporting.

    python -m pytest 02_model_server/test_model_config.py -q
"""
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

import model_loader as ml  # noqa: E402


def test_read_threshold_accepts_known_keys():
    assert ml.read_threshold({"threshold_mse": 0.5}) == 0.5
    assert ml.read_threshold({"anomaly_threshold": "0.25"}) == 0.25


@pytest.mark.parametrize("config", [{}, {"threshold": 0.5}, {"threshold_mse": None},
                                    {"threshold_mse": "abc"}, {"threshold_mse": 0},
                                    {"threshold_mse": float("nan")}])
def test_read_threshold_rejects_missing_or_invalid(config):
    with pytest.raises(ValueError, match="threshold|numeric|positive"):
        ml.read_threshold(config)


@pytest.mark.parametrize("folder", ["keypoint", "anomaly"])
def test_local_configs_have_threshold(folder):
    path = REPO / "models" / folder / "config.json"
    if not path.is_file():
        pytest.skip(f"{path} not available")
    assert ml.read_threshold(json.loads(path.read_text(encoding="utf-8")), path) > 0


def test_hf_upload_keypoint_config_matches_canonical():
    canonical, upload = REPO / "models/keypoint/config.json", REPO / "hf_upload/keypoint/config.json"
    if not (canonical.is_file() and upload.is_file()):
        pytest.skip("configs not available")
    assert json.loads(canonical.read_text()) == json.loads(upload.read_text())


def _write_model(directory: Path):
    directory.mkdir(parents=True)
    for name in ml.REQUIRED_FILES:
        (directory / name).write_text("x")


def test_subfolder_resolution(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_KEYPOINT_SUBFOLDER", raising=False)
    (tmp_path / "config.json").write_text("{}")  # stray root config must not win
    _write_model(tmp_path / "keypoint")
    assert ml._resolve_subfolder(tmp_path, "keypoint") == tmp_path / "keypoint"
    monkeypatch.setenv("HF_KEYPOINT_SUBFOLDER", "missing")
    with pytest.raises(RuntimeError, match="missing"):
        ml._resolve_subfolder(tmp_path, "keypoint")


def test_health_reports_actual_load():
    if not all((REPO / "models/keypoint" / f).is_file() for f in ml.REQUIRED_FILES):
        pytest.skip("local keypoint model files are not available")
    import os
    os.environ.pop("HF_KEYPOINT_REPO", None)
    os.environ.pop("HF_IMU_REPO", None)
    from fastapi.testclient import TestClient
    import app as server
    body = TestClient(server.app).get("/health").json()
    kp = body["models"]["keypoint"]
    assert kp["loaded"] is True and kp["error"] is None
    assert kp["threshold"] == pytest.approx(0.6588426828384399)
