import base64
import json
from pathlib import Path

import pytest


def test_feedback_success_with_scoring(client, api_headers, app_module):
    audio_bytes = b"\x01" * 400
    payload = {
        "label": "AI_GENERATED",
        "audioFormat": "mp3",
        "audioBase64": base64.b64encode(audio_bytes).decode("utf-8"),
        "runDetection": True,
        "metadata": {"source": "unit-test"}
    }

    response = client.post("/api/feedback", json=payload, headers=api_headers)
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"

    storage_dir = Path(app_module.FEEDBACK_STORAGE_DIR)
    assert storage_dir.exists()
    stored_files = list(storage_dir.rglob("*.mp3"))
    assert stored_files, "Expected feedback audio file to be stored"
    meta_files = list(storage_dir.rglob("*.json"))
    assert meta_files, "Expected feedback metadata to be stored"

    metadata = json.loads(meta_files[0].read_text(encoding="utf-8"))
    assert metadata["label"] == "AI_GENERATED"
    assert "physics_score" in metadata
    assert "dl_score" in metadata


def test_feedback_invalid_label(client, api_headers, sample_audio_base64):
    payload = {
        "label": "UNKNOWN",
        "audioFormat": "mp3",
        "audioBase64": sample_audio_base64
    }
    response = client.post("/api/feedback", json=payload, headers=api_headers)
    assert response.status_code == 400


def test_feedback_disabled(app_factory, sample_audio_base64, api_headers):
    app_module = app_factory(ENABLE_FEEDBACK_STORAGE="false")
    client = app_module.app.test_client()

    payload = {
        "label": "HUMAN",
        "audioFormat": "mp3",
        "audioBase64": sample_audio_base64
    }
    response = client.post("/api/feedback", json=payload, headers=api_headers)
    assert response.status_code == 403


def test_feedback_too_large_payload(app_module, client, api_headers):
    big_audio = base64.b64encode(b"\x00" * (app_module.FEEDBACK_MAX_BYTES + 10)).decode("utf-8")
    payload = {
        "label": "AI_GENERATED",
        "audioFormat": "mp3",
        "audioBase64": big_audio
    }
    response = client.post("/api/feedback", json=payload, headers=api_headers)
    assert response.status_code == 413
