from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import cv2
import librosa
try:
    from mediapipe import solutions as mp_solutions
except Exception:
    mp_solutions = None
import numpy as np
try:
    from moviepy.editor import VideoFileClip
except Exception:
    from moviepy import VideoFileClip
from PIL import Image
from transformers import pipeline


@dataclass
class DeepfakeScore:
    score: float
    label: str
    is_deepfake: bool


class FrameDeepfakeDetector:
    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name = model_name or os.environ.get(
            "SENTINEL_DEEPFAKE_MODEL", "dima806/deepfake_vs_real_image_detection"
        )
        self._classifier = None

    def _get_classifier(self):
        if self._classifier is None:
            self._classifier = pipeline("image-classification", model=self.model_name)
        return self._classifier

    def score_frame(self, frame_bgr: np.ndarray) -> DeepfakeScore:
        image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        classifier = self._get_classifier()
        outputs = classifier(image, top_k=2)
        # Expect labels like "FAKE"/"REAL" or similar
        best = outputs[0]
        label = str(best.get("label", "UNKNOWN")).lower()
        score = float(best.get("score", 0.0))
        is_fake = "fake" in label or "deepfake" in label
        return DeepfakeScore(score=score if is_fake else 1.0 - score, label=label, is_deepfake=is_fake)


class AVSyncDetector:
    def __init__(self) -> None:
        self._mesh = None
        if mp_solutions is None:
            return
        try:
            self._mesh = mp_solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
            )
        except Exception:
            self._mesh = None

    @staticmethod
    def _mouth_openness(landmarks, image_h: int) -> float:
        upper = landmarks[13]
        lower = landmarks[14]
        return abs((lower.y - upper.y) * image_h)

    def compute_mouth_activity(self, frame_bgr: np.ndarray) -> float:
        if self._mesh is None:
            return 0.0
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._mesh.process(rgb)
        if not result.multi_face_landmarks:
            return 0.0
        landmarks = result.multi_face_landmarks[0].landmark
        return float(self._mouth_openness(landmarks, rgb.shape[0]))

    @staticmethod
    def compute_audio_energy(video_path: str, timestamps: list[float], window_sec: float = 0.2) -> list[float]:
        try:
            clip = VideoFileClip(video_path)
            if clip.audio is None:
                return [0.0 for _ in timestamps]
            audio = clip.audio.to_soundarray(fps=16000)
            clip.close()
        except Exception:
            return [0.0 for _ in timestamps]

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        sr = 16000
        energies = []
        for t in timestamps:
            start = int(max(0, (t - window_sec / 2) * sr))
            end = int(min(len(audio), (t + window_sec / 2) * sr))
            if end <= start:
                energies.append(0.0)
                continue
            segment = audio[start:end]
            rms = float(np.sqrt(np.mean(segment ** 2)))
            energies.append(rms)
        return energies

    @staticmethod
    def sync_score(mouth_activity: list[float], audio_energy: list[float]) -> float:
        if len(mouth_activity) < 3:
            return 0.0
        x = np.array(mouth_activity)
        y = np.array(audio_energy)
        if np.std(x) == 0 or np.std(y) == 0:
            return 0.0
        corr = float(np.corrcoef(x, y)[0, 1])
        return max(0.0, min(1.0, (corr + 1.0) / 2.0))
