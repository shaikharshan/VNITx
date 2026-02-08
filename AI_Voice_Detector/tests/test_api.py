import json
from pathlib import Path

import pytest


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "healthy"
    assert payload["streaming_enabled"] is True


def test_voice_detection_success_with_sample_base64(client, api_headers, sample_audio_base64):
    payload = {
        "language": "English",
        "audioFormat": "mp3",
        "audioBase64": sample_audio_base64
    }
    response = client.post("/api/voice-detection", data=json.dumps(payload), headers=api_headers)
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["classification"] == "AI_GENERATED"


def test_voice_detection_success_with_test_audio(client, api_headers, test_audio_base64):
    path, audio_b64 = test_audio_base64
    if path.suffix.lower() != ".mp3":
        pytest.skip("test_audio file is not mp3 (endpoint only supports mp3).")

    payload = {
        "language": "English",
        "audioFormat": "mp3",
        "audioBase64": audio_b64
    }
    response = client.post("/api/voice-detection", data=json.dumps(payload), headers=api_headers)
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"


def test_voice_detection_missing_api_key(client, sample_audio_base64):
    payload = {
        "language": "English",
        "audioFormat": "mp3",
        "audioBase64": sample_audio_base64
    }
    response = client.post("/api/voice-detection", json=payload)
    assert response.status_code == 401


def test_voice_detection_invalid_api_key(client, api_headers, sample_audio_base64):
    payload = {
        "language": "English",
        "audioFormat": "mp3",
        "audioBase64": sample_audio_base64
    }
    headers = dict(api_headers)
    headers["x-api-key"] = "wrong_key"
    response = client.post("/api/voice-detection", json=payload, headers=headers)
    assert response.status_code == 403


def test_voice_detection_invalid_content_type(client, api_headers):
    response = client.post("/api/voice-detection", data="not json", headers=api_headers)
    assert response.status_code == 400


def test_voice_detection_missing_fields(client, api_headers):
    payload = {"language": "English"}
    response = client.post("/api/voice-detection", json=payload, headers=api_headers)
    assert response.status_code == 400


def test_voice_detection_unsupported_language(client, api_headers, sample_audio_base64):
    payload = {
        "language": "Spanish",
        "audioFormat": "mp3",
        "audioBase64": sample_audio_base64
    }
    response = client.post("/api/voice-detection", json=payload, headers=api_headers)
    assert response.status_code == 400


def test_voice_detection_unsupported_audio_format(client, api_headers, sample_audio_base64):
    payload = {
        "language": "English",
        "audioFormat": "wav",
        "audioBase64": sample_audio_base64
    }
    response = client.post("/api/voice-detection", json=payload, headers=api_headers)
    assert response.status_code == 400


def test_voice_detection_invalid_audio_payload(client, api_headers):
    payload = {
        "language": "English",
        "audioFormat": "mp3",
        "audioBase64": "short"
    }
    response = client.post("/api/voice-detection", json=payload, headers=api_headers)
    assert response.status_code == 400


def test_voice_detection_analysis_error(app_module, client, api_headers, sample_audio_base64):
    def error_analyze(*args, **kwargs):
        return {"status": "error", "error": "boom"}

    app_module.detector.analyze = error_analyze

    payload = {
        "language": "English",
        "audioFormat": "mp3",
        "audioBase64": sample_audio_base64
    }
    response = client.post("/api/voice-detection", json=payload, headers=api_headers)
    assert response.status_code == 500


def test_reload_calibration_not_found(client, api_headers):
    response = client.post("/api/reload-calibration", headers=api_headers)
    assert response.status_code == 404


def test_reload_calibration_success(app_module, client, api_headers):
    calibration_file = Path(app_module.CALIBRATION_PATH)
    calibration_file.parent.mkdir(parents=True, exist_ok=True)
    calibration_file.write_text("{}", encoding="utf-8")

    response = client.post("/api/reload-calibration", headers=api_headers)
    assert response.status_code == 200


def test_backup_and_rollback_calibration(app_module, client, api_headers):
    calibration_file = Path(app_module.CALIBRATION_PATH)
    calibration_file.parent.mkdir(parents=True, exist_ok=True)
    calibration_file.write_text('{"version": "original"}', encoding="utf-8")

    backup_response = client.post("/api/backup-calibration", headers=api_headers)
    assert backup_response.status_code == 200
    backup_payload = backup_response.get_json()
    version_id = backup_payload["versionId"]

    calibration_file.write_text('{"version": "new"}', encoding="utf-8")

    rollback_response = client.post(
        "/api/rollback-calibration",
        json={"versionId": version_id},
        headers=api_headers
    )
    assert rollback_response.status_code == 200
    assert calibration_file.read_text(encoding="utf-8") == '{"version": "original"}'


def test_backup_calibration_missing_file(client, api_headers):
    response = client.post("/api/backup-calibration", headers=api_headers)
    assert response.status_code == 404


def test_rollback_calibration_missing_version(client, api_headers):
    response = client.post("/api/rollback-calibration", json={}, headers=api_headers)
    assert response.status_code == 400


def test_calibration_history_list(app_module, client, api_headers):
    history_dir = Path(app_module.CALIBRATION_HISTORY_DIR)
    history_dir.mkdir(parents=True, exist_ok=True)
    history_file = history_dir / "calibration_20260207T120000Z_deadbeef.json"
    history_file.write_text("{}", encoding="utf-8")

    response = client.get("/api/calibration-history", headers=api_headers)
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "success"
    assert payload["history"]
