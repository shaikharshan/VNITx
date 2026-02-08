import asyncio
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile

from src.video.video_processor import VideoAnalyzer

app = FastAPI(title="Video Prompt Detection API")
_ANALYZER: VideoAnalyzer | None = None


@app.on_event("startup")
def load_analyzer() -> None:
    global _ANALYZER
    if _ANALYZER is None:
        _ANALYZER = VideoAnalyzer()


@app.get("/")
def health_check() -> dict:
    return {"status": "ok", "engine": "video"}


@app.post("/analyze_video")
async def analyze_video(
    video: UploadFile = File(...),
    audio_transcript: str = Form(""),
    target_fps: float = Form(5.0),
    max_frames: int | None = Form(None),
    run_injection: bool = Form(True),
    run_cross_modal: bool = Form(True),
    run_caption: bool = Form(True),
    run_vision_deepfake: bool = Form(True),
    run_avsync: bool = Form(True),
    log_frames: bool = Form(True),
) -> dict:
    if _ANALYZER is None:
        load_analyzer()
    analyzer = _ANALYZER
    video_bytes = await video.read()
    log_path = None
    if log_frames:
        log_path = f"/tmp/video_frame_log_{int(asyncio.get_event_loop().time()*1000)}.jsonl"

    frames, summary = analyzer.analyze_video_bytes(
        video_bytes,
        audio_transcript=audio_transcript,
        target_fps=target_fps,
        max_frames=max_frames,
        run_injection=run_injection,
        run_cross_modal=run_cross_modal,
        run_caption=run_caption,
        run_vision_deepfake=run_vision_deepfake,
        run_avsync=run_avsync,
        log_path=Path(log_path) if log_path else None,
    )

    top_risky = sorted(frames, key=lambda f: f.final_score, reverse=True)[:5]

    def _action_from_score(score: float) -> str:
        if score >= 0.7:
            return "BLOCK"
        if score >= 0.5:
            return "FLAG"
        return "ALLOW"

    def flatten(frame):
        action = _action_from_score(frame.final_score)
        return {
            "frame_index": frame.frame_index,
            "timestamp_sec": frame.timestamp_sec,
            "final_score": frame.final_score,
            "action": action,
            "deepfake_score": frame.deepfake_score,
            "deepfake_label": frame.deepfake_label,
            "deepfake_is_fake": frame.deepfake_is_fake,
            "injection_risk": frame.injection.get("risk_score", 0.0),
            "injection_reason": frame.injection.get("reason", ""),
            "cross_modal_score": frame.cross_modal.get("consistency_score", 0.0),
            "ocr_vs_image_score": frame.ocr_vs_image.get("consistency_score", 0.0),
            "caption_alignment_score": frame.caption_alignment.get("alignment_score", 0.0),
            "caption": frame.caption_alignment.get("caption", ""),
            "ocr_text": frame.ocr_text,
        }

    action = _action_from_score(summary.get("max_final_score", 0.0))
    explanations = [
        f"avg_deepfake={summary.get('avg_deepfake_score', 0.0)}",
        f"avsync={summary.get('avsync_score', 0.0)}",
        f"max_final={summary.get('max_final_score', 0.0)}",
    ]

    return {
        "summary": summary,
        "timeline": [f.__dict__ for f in frames],
        "timeline_flat": [flatten(f) for f in frames],
        "top_risky_frames": [f.__dict__ for f in top_risky],
        "top_risky_frames_flat": [flatten(f) for f in top_risky],
        "action": action,
        "explanations": explanations,
        "log_path": log_path,
    }


@app.post("/analyze_webcam")
async def analyze_webcam(
    camera_index: int = Form(0),
    duration_sec: float = Form(10.0),
    target_fps: float = Form(5.0),
    run_injection: bool = Form(True),
    run_cross_modal: bool = Form(True),
    run_caption: bool = Form(True),
    run_vision_deepfake: bool = Form(True),
    run_avsync: bool = Form(True),
    log_frames: bool = Form(True),
) -> dict:
    if _ANALYZER is None:
        load_analyzer()
    analyzer = _ANALYZER
    log_path = None
    if log_frames:
        log_path = f"/tmp/webcam_frame_log_{int(asyncio.get_event_loop().time()*1000)}.jsonl"

    frames, summary = analyzer.analyze_webcam(
        camera_index=camera_index,
        duration_sec=duration_sec,
        target_fps=target_fps,
        run_injection=run_injection,
        run_cross_modal=run_cross_modal,
        run_caption=run_caption,
        run_vision_deepfake=run_vision_deepfake,
        run_avsync=run_avsync,
        log_path=Path(log_path) if log_path else None,
    )
    top_risky = sorted(frames, key=lambda f: f.final_score, reverse=True)[:5]

    def _action_from_score(score: float) -> str:
        if score >= 0.7:
            return "BLOCK"
        if score >= 0.5:
            return "FLAG"
        return "ALLOW"

    def flatten(frame):
        action = _action_from_score(frame.final_score)
        return {
            "frame_index": frame.frame_index,
            "timestamp_sec": frame.timestamp_sec,
            "final_score": frame.final_score,
            "action": action,
            "deepfake_score": frame.deepfake_score,
            "deepfake_label": frame.deepfake_label,
            "deepfake_is_fake": frame.deepfake_is_fake,
            "injection_risk": frame.injection.get("risk_score", 0.0),
            "injection_reason": frame.injection.get("reason", ""),
            "cross_modal_score": frame.cross_modal.get("consistency_score", 0.0),
            "ocr_vs_image_score": frame.ocr_vs_image.get("consistency_score", 0.0),
            "caption_alignment_score": frame.caption_alignment.get("alignment_score", 0.0),
            "caption": frame.caption_alignment.get("caption", ""),
            "ocr_text": frame.ocr_text,
        }

    action = _action_from_score(summary.get("max_final_score", 0.0))
    explanations = [
        f"avg_deepfake={summary.get('avg_deepfake_score', 0.0)}",
        f"avsync={summary.get('avsync_score', 0.0)}",
        f"max_final={summary.get('max_final_score', 0.0)}",
    ]

    return {
        "summary": summary,
        "timeline": [f.__dict__ for f in frames],
        "timeline_flat": [flatten(f) for f in frames],
        "top_risky_frames": [f.__dict__ for f in top_risky],
        "top_risky_frames_flat": [flatten(f) for f in top_risky],
        "action": action,
        "explanations": explanations,
        "log_path": log_path,
    }
