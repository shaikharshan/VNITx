import io
import json
import tempfile
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.engines.visual_engine import VisualSecurityEngine
from src.video.video_deepfake import AVSyncDetector, FrameDeepfakeDetector


@dataclass
class VideoFrameResult:
    frame_index: int
    timestamp_sec: float
    ocr_text: str
    injection: Dict[str, Any]
    cross_modal: Dict[str, Any]
    ocr_vs_image: Dict[str, Any]
    caption_alignment: Dict[str, Any]
    deepfake_score: float
    deepfake_label: str
    deepfake_is_fake: bool
    final_score: float


class VideoAnalyzer:
    def __init__(self, engine: Optional[VisualSecurityEngine] = None) -> None:
        self.engine = engine or VisualSecurityEngine()
        self._deepfake = FrameDeepfakeDetector()
        self._avsync = AVSyncDetector()

    @staticmethod
    def _encode_frame(frame_bgr: np.ndarray) -> bytes:
        success, buffer = cv2.imencode(".jpg", frame_bgr)
        if not success:
            raise RuntimeError("Failed to encode frame")
        return buffer.tobytes()

    @staticmethod
    def _compute_final_score(
        injection_risk: float,
        ocr_confidence: float,
        audio_align: float,
        ocr_img_align: float,
        caption_align: float,
        deepfake_score: float,
    ) -> float:
        score = (
            0.35 * injection_risk
            + 0.15 * (1.0 - ocr_confidence)
            + 0.15 * (1.0 - audio_align)
            + 0.1 * (1.0 - ocr_img_align)
            + 0.1 * (1.0 - caption_align)
            + 0.15 * deepfake_score
        )
        return max(0.0, min(1.0, score))

    @staticmethod
    def _sample_step(native_fps: float, target_fps: float) -> int:
        if native_fps <= 0:
            native_fps = 30.0
        if target_fps <= 0:
            target_fps = 5.0
        step = int(round(native_fps / target_fps))
        return max(1, step)

    def analyze_capture(
        self,
        capture: cv2.VideoCapture,
        audio_transcript: str = "",
        target_fps: float = 5.0,
        max_frames: Optional[int] = None,
        run_injection: bool = True,
        run_cross_modal: bool = True,
        run_caption: bool = True,
        run_vision_deepfake: bool = True,
        run_avsync: bool = True,
        video_path: Optional[str] = None,
        log_path: Optional[Path] = None,
        max_frame_width: int = 640,
    ) -> Tuple[List[VideoFrameResult], Dict[str, Any]]:
        native_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        step = self._sample_step(native_fps, target_fps)

        results: List[VideoFrameResult] = []
        frame_index = 0
        processed = 0
        frame_logs: List[dict] = []

        mouth_activity: List[float] = []
        timestamps: List[float] = []

        while True:
            success, frame = capture.read()
            if not success:
                break
            if frame_index % step != 0:
                frame_index += 1
                continue

            if max_frame_width and frame.shape[1] > max_frame_width:
                scale = max_frame_width / float(frame.shape[1])
                frame = cv2.resize(frame, (max_frame_width, int(frame.shape[0] * scale)))

            frame_bytes = self._encode_frame(frame)
            text_payload = self.engine.extract_text(frame_bytes)
            ocr_text = text_payload.get("normalized_text", "")
            scores = [score for _, score in text_payload.get("scored", [])]
            ocr_conf = float(sum(scores) / len(scores)) if scores else 0.5

            if run_injection:
                injection = self.engine.detect_injection_from_text(ocr_text)
            else:
                injection = {"skipped": True, "risk_score": 0.0}

            if run_cross_modal and audio_transcript:
                cross_modal = self.engine.check_cross_modal(frame_bytes, audio_transcript)
            elif run_cross_modal:
                cross_modal = {"is_mismatch": True, "consistency_score": 0.0}
            else:
                cross_modal = {"skipped": True, "consistency_score": 0.0}

            ocr_vs_image = (
                self.engine.check_ocr_vs_image(frame_bytes, ocr_text)
                if ocr_text
                else {"is_mismatch": False, "consistency_score": 0.0}
            )

            caption_alignment = (
                self.engine.check_caption_alignment(frame_bytes, ocr_text)
                if run_caption and ocr_text
                else {"caption": "", "alignment_score": 0.0}
            )

            if run_vision_deepfake:
                deepfake_info = self._deepfake.score_frame(frame)
                deepfake_score = deepfake_info.score
                deepfake_label = deepfake_info.label
                deepfake_is_fake = deepfake_info.is_deepfake
            else:
                deepfake_score = 0.0
                deepfake_label = "skipped"
                deepfake_is_fake = False

            final_score = self._compute_final_score(
                float(injection.get("risk_score", 0.0)),
                ocr_conf,
                float(cross_modal.get("consistency_score", 0.0)),
                float(ocr_vs_image.get("consistency_score", 0.0)),
                float(caption_alignment.get("alignment_score", 0.0)),
                float(deepfake_score),
            )

            timestamp_sec = frame_index / (native_fps if native_fps > 0 else 30.0)
            if run_avsync:
                mouth_activity.append(self._avsync.compute_mouth_activity(frame))
                timestamps.append(timestamp_sec)

            results.append(
                VideoFrameResult(
                    frame_index=frame_index,
                    timestamp_sec=round(timestamp_sec, 3),
                    ocr_text=ocr_text,
                    injection=injection,
                    cross_modal=cross_modal,
                    ocr_vs_image=ocr_vs_image,
                    caption_alignment=caption_alignment,
                    deepfake_score=round(deepfake_score, 3),
                    deepfake_label=deepfake_label,
                    deepfake_is_fake=deepfake_is_fake,
                    final_score=round(final_score, 3),
                )
            )

            frame_logs.append(
                {
                    "frame_index": frame_index,
                    "timestamp_sec": round(timestamp_sec, 3),
                    "final_score": round(final_score, 3),
                    "deepfake_score": round(deepfake_score, 3),
                    "deepfake_label": deepfake_label,
                    "deepfake_is_fake": deepfake_is_fake,
                    "injection_risk": injection.get("risk_score", 0.0),
                    "injection_reason": injection.get("reason", ""),
                    "cross_modal_score": cross_modal.get("consistency_score", 0.0),
                    "ocr_vs_image_score": ocr_vs_image.get("consistency_score", 0.0),
                    "caption_alignment_score": caption_alignment.get("alignment_score", 0.0),
                    "caption": caption_alignment.get("caption", ""),
                    "ocr_text": ocr_text,
                }
            )

            processed += 1
            if max_frames and processed >= max_frames:
                break
            frame_index += 1

        avsync_score = 0.0
        if run_avsync and video_path:
            audio_energy = self._avsync.compute_audio_energy(video_path, timestamps)
            avsync_score = self._avsync.sync_score(mouth_activity, audio_energy)

        deepfake_scores = [frame.deepfake_score for frame in results]
        avg_deepfake = float(sum(deepfake_scores) / len(deepfake_scores)) if deepfake_scores else 0.0
        final_scores = [frame.final_score for frame in results]
        avg_final = float(sum(final_scores) / len(final_scores)) if final_scores else 0.0
        max_final = float(max(final_scores)) if final_scores else 0.0

        verdict = "REAL"
        if avg_deepfake >= 0.6 or avsync_score < 0.35 or max_final >= 0.7:
            verdict = "DEEPFAKE_SUSPECT"
        elif max_final >= 0.5:
            verdict = "SUSPICIOUS"

        summary = {
            "frames_processed": processed,
            "sample_step": step,
            "native_fps": native_fps,
            "avsync_score": round(avsync_score, 3),
            "avg_deepfake_score": round(avg_deepfake, 3),
            "avg_final_score": round(avg_final, 3),
            "max_final_score": round(max_final, 3),
            "verdict": verdict,
            "verdict_criteria": {
                "deepfake_avg_threshold": 0.6,
                "avsync_min_threshold": 0.35,
                "max_final_threshold": 0.7,
                "suspicious_max_final_threshold": 0.5,
            },
        }

        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "w", encoding="utf-8") as handle:
                for row in frame_logs:
                    handle.write(json.dumps(row) + "\n")

        return results, summary

    def analyze_video_bytes(
        self,
        video_bytes: bytes,
        audio_transcript: str = "",
        target_fps: float = 5.0,
        max_frames: Optional[int] = None,
        run_injection: bool = True,
        run_cross_modal: bool = True,
        run_caption: bool = True,
        run_vision_deepfake: bool = True,
        run_avsync: bool = True,
        log_path: Optional[Path] = None,
        max_frame_width: int = 640,
    ) -> Tuple[List[VideoFrameResult], Dict[str, Any]]:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as handle:
            handle.write(video_bytes)
            handle.flush()
            capture = cv2.VideoCapture(handle.name)
        try:
            return self.analyze_capture(
                capture,
                audio_transcript=audio_transcript,
                target_fps=target_fps,
                max_frames=max_frames,
                run_injection=run_injection,
                run_cross_modal=run_cross_modal,
                run_caption=run_caption,
                run_vision_deepfake=run_vision_deepfake,
                run_avsync=run_avsync,
                video_path=handle.name,
                log_path=log_path,
                max_frame_width=max_frame_width,
            )
        finally:
            capture.release()

    def analyze_webcam(
        self,
        camera_index: int = 0,
        duration_sec: float = 10.0,
        target_fps: float = 5.0,
        run_injection: bool = True,
        run_cross_modal: bool = True,
        run_caption: bool = True,
        run_vision_deepfake: bool = True,
        run_avsync: bool = True,
        log_path: Optional[Path] = None,
        max_frame_width: int = 640,
    ) -> Tuple[List[VideoFrameResult], Dict[str, Any]]:
        capture = cv2.VideoCapture(camera_index)
        try:
            max_frames = int(duration_sec * target_fps) if duration_sec > 0 else None
            return self.analyze_capture(
                capture,
                audio_transcript="",
                target_fps=target_fps,
                max_frames=max_frames,
                run_injection=run_injection,
                run_cross_modal=run_cross_modal,
                run_caption=run_caption,
                run_vision_deepfake=run_vision_deepfake,
                run_avsync=run_avsync,
                video_path=None,
                log_path=log_path,
                max_frame_width=max_frame_width,
            )
        finally:
            capture.release()
