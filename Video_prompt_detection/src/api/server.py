from fastapi import FastAPI, File, Form, UploadFile

from src.engines.visual_engine import VisualSecurityEngine

app = FastAPI(title="Visual Security Engine API")
_ENGINE: VisualSecurityEngine | None = None


@app.on_event("startup")
def load_engine() -> None:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = VisualSecurityEngine()


@app.get("/")
def health_check() -> dict:
    return {"status": "ok"}


@app.post("/analyze")
async def analyze_image(
    image: UploadFile = File(...),
    audio_transcript: str = Form(""),
    run_ocr: bool = Form(True),
    run_injection: bool = Form(True),
    run_cross_modal: bool = Form(True),
) -> dict:
    if _ENGINE is None:
        load_engine()
    engine = _ENGINE
    image_bytes = await image.read()
    if run_injection:
        run_ocr = True

    text_payload = None
    if run_ocr:
        text_payload = engine.extract_text(image_bytes)

    if run_injection:
        injection_result = engine.detect_injection_from_text(
            text_payload["normalized_text"] if text_payload else ""
        )
    else:
        injection_result = {"skipped": True}

    if run_cross_modal and audio_transcript.strip():
        cross_modal_result = engine.check_cross_modal(image_bytes, audio_transcript)
    elif run_cross_modal:
        cross_modal_result = {"is_mismatch": True, "consistency_score": 0.0}
    else:
        cross_modal_result = {"skipped": True}

    return {
        "ocr": text_payload or {"skipped": True},
        "injection": injection_result,
        "cross_modal": cross_modal_result,
    }
