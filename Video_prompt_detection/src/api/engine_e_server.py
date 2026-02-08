from fastapi import FastAPI, File, Form, UploadFile

from src.engines.visual_engine import CrossModalEngine

app = FastAPI(title="Engine E (Cross-Modal) API")
_ENGINE: CrossModalEngine | None = None


@app.on_event("startup")
def load_engine() -> None:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = CrossModalEngine()


@app.get("/")
def health_check() -> dict:
    return {"status": "ok", "engine": "e"}


@app.post("/analyze_e")
async def analyze_engine_e(
    image: UploadFile = File(...),
    audio_transcript: str = Form(""),
    ocr_text: str = Form(""),
    run_caption: bool = Form(True),
) -> dict:
    if _ENGINE is None:
        load_engine()
    engine = _ENGINE
    image_bytes = await image.read()
    cross_modal_result = engine.check_cross_modal(image_bytes, audio_transcript)
    ocr_vs_image = engine.check_ocr_vs_image(image_bytes, ocr_text) if ocr_text else {
        "is_mismatch": False,
        "consistency_score": 0.0,
    }
    caption_alignment = (
        engine.check_caption_alignment(image_bytes, ocr_text) if run_caption else {"caption": "", "alignment_score": 0.0}
    )
    return {
        "cross_modal": cross_modal_result,
        "ocr_vs_image": ocr_vs_image,
        "caption_alignment": caption_alignment,
    }
