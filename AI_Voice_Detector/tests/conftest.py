import base64
import importlib
import os
import sys
from pathlib import Path

import pytest


TEST_API_KEY = "test_key_123"


class DummyCalibrator:
    ready = False
    calibration_path = None


class DummyDetector:
    def __init__(self):
        self.calibrator = DummyCalibrator()

    def analyze(self, audio_input, input_type="file", audio_format="mp3", analysis_mode="full"):
        return {
            "status": "success",
            "language": "English",
            "classification": "AI_GENERATED",
            "confidenceScore": 0.87,
            "explanation": "Dummy detector response",
            "analysisMode": analysis_mode,
            "debug": {
                "analysis_mode": analysis_mode,
                "used_calibration": False
            }
        }

    def extract_scores(self, audio_input, input_type="file", audio_format="mp3"):
        return {
            "status": "success",
            "physics_score": 0.42,
            "dl_score": 0.84,
            "dl_label": "Fake/Deepfake",
            "physics_method": "Physics Analysis",
            "audio_duration": 1.0,
            "was_truncated": False
        }

    def reload_calibration(self, calibration_path=None):
        return bool(calibration_path and os.path.exists(calibration_path))


def load_app(tmp_path, monkeypatch, overrides=None):
    env = {
        "API_KEY": TEST_API_KEY,
        "SKIP_MODEL_LOAD": "true",
        "ENABLE_STREAMING": "true",
        "ENABLE_FEEDBACK_STORAGE": "true",
        "FEEDBACK_STORAGE_DIR": str(tmp_path / "feedback"),
        "FEEDBACK_MAX_BYTES": "2048",
        "CALIBRATION_PATH": str(tmp_path / "calibration.json"),
        "CALIBRATION_HISTORY_DIR": str(tmp_path / "calibration_history"),
        "CALIBRATION_HISTORY_MAX": "5",
        "STREAMING_PARTIAL_INTERVAL_SECONDS": "0.5"
    }
    if overrides:
        env.update(overrides)

    for key, value in env.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, str(value))

    if "app" in sys.modules:
        del sys.modules["app"]

    app_module = importlib.import_module("app")
    importlib.reload(app_module)

    dummy = DummyDetector()
    app_module.detector = dummy

    def init_detector():
        app_module.detector = dummy
        return True

    app_module.init_detector = init_detector

    return app_module


@pytest.fixture
def app_factory(tmp_path, monkeypatch):
    def _factory(**overrides):
        return load_app(tmp_path, monkeypatch, overrides=overrides)
    return _factory


@pytest.fixture
def app_module(app_factory):
    return app_factory()


@pytest.fixture
def client(app_module):
    return app_module.app.test_client()


@pytest.fixture
def api_headers():
    return {
        "Content-Type": "application/json",
        "x-api-key": TEST_API_KEY
    }


@pytest.fixture
def sample_audio_base64():
    return base64.b64encode(b"\x00" * 200).decode("utf-8")


def find_test_audio_files():
    base_dir = Path(__file__).resolve().parent.parent / "test_audio"
    if not base_dir.exists():
        return []
    return sorted([p for p in base_dir.iterdir() if p.suffix.lower() in [".mp3", ".wav"]])


def load_test_audio_base64(prefer_extension=".mp3"):
    candidates = find_test_audio_files()
    for path in candidates:
        if path.suffix.lower() == prefer_extension:
            return path, base64.b64encode(path.read_bytes()).decode("utf-8")
    if candidates:
        path = candidates[0]
        return path, base64.b64encode(path.read_bytes()).decode("utf-8")
    return None, None


@pytest.fixture
def test_audio_base64():
    path, b64_data = load_test_audio_base64(".mp3")
    if not b64_data:
        pytest.skip("No audio files found in test_audio/")
    return path, b64_data
