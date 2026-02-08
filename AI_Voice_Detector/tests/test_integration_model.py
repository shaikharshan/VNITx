import os
from pathlib import Path

import pytest

from detector import HybridEnsembleDetector


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_MODEL_TESTS", "").lower() not in ["1", "true", "yes"],
        reason="Integration tests require RUN_MODEL_TESTS=true and model weights available."
    )
]


def find_ai_miss_audio():
    env_path = os.environ.get("AI_MISS_AUDIO_PATH")
    if env_path and Path(env_path).exists():
        return Path(env_path)

    base_dir = Path(__file__).resolve().parent.parent / "test_audio"
    if not base_dir.exists():
        return None

    candidates = []
    for path in base_dir.iterdir():
        if path.suffix.lower() not in [".mp3", ".wav"]:
            continue
        name = path.stem.lower()
        if "miss" in name or "false" in name or "hard" in name:
            candidates.append(path)

    return candidates[0] if candidates else None


@pytest.mark.xfail(reason="Known false negative before retraining", strict=False)
def test_known_false_negative_ai_sample():
    audio_path = find_ai_miss_audio()
    if audio_path is None:
        pytest.skip("No known false-negative AI sample provided.")

    detector = HybridEnsembleDetector(
        deepfake_model_path=os.environ.get(
            "DEEPFAKE_MODEL_PATH",
            "garystafford/wav2vec2-deepfake-voice-detector"
        ),
        whisper_model_path=os.environ.get(
            "WHISPER_MODEL_PATH",
            "openai/whisper-base"
        ),
        use_local_deepfake_model=os.environ.get("USE_LOCAL_DEEPFAKE_MODEL", "false").lower() in ["1", "true"],
        use_local_whisper_model=os.environ.get("USE_LOCAL_WHISPER_MODEL", "false").lower() in ["1", "true"],
        max_audio_duration=int(os.environ.get("MAX_AUDIO_DURATION", "30"))
    )

    result = detector.analyze(str(audio_path), input_type="file")
    assert result["status"] == "success"
    assert result["classification"] == "AI_GENERATED"
