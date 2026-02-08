import os

import httpx
from fastapi import FastAPI, File, Form, UploadFile

app = FastAPI(title="Visual Security Engine Gateway API")


def _engine_d_url() -> str:
    return os.environ.get("ENGINE_D_URL", "http://localhost:8001").rstrip("/")


def _engine_e_url() -> str:
    return os.environ.get("ENGINE_E_URL", "http://localhost:8002").rstrip("/")


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _action_from_score(score: float) -> str:
    if score >= 0.7:
        return "BLOCK"
    if score >= 0.5:
        return "FLAG"
    return "ALLOW"


@app.get("/")
def health_check() -> dict:
    return {"status": "ok", "engine": "gateway"}


@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    audio_transcript: str = Form(""),
    run_caption: bool = Form(True),
    deep: bool = Form(True),
) -> dict:
    image_bytes = await image.read()

    async with httpx.AsyncClient(timeout=300) as client:
        resp_d = await client.post(
            f"{_engine_d_url()}/analyze_d",
            files={"image": (image.filename, image_bytes, image.content_type or "image/jpeg")},
            data={"deep": str(deep).lower()},
        )
        resp_d.raise_for_status()
        payload_d = resp_d.json()

        ocr_text = payload_d.get("ocr", {}).get("normalized_text", "")
        resp_e = await client.post(
            f"{_engine_e_url()}/analyze_e",
            files={"image": (image.filename, image_bytes, image.content_type or "image/jpeg")},
            data={
                "audio_transcript": audio_transcript,
                "ocr_text": ocr_text,
                "run_caption": str(run_caption).lower(),
            },
        )
        resp_e.raise_for_status()
        payload_e = resp_e.json()

    injection = payload_d.get("injection", {})
    ocr_conf = float(payload_d.get("ocr", {}).get("ocr_confidence", 0.5))
    cross_modal = payload_e.get("cross_modal", {})
    ocr_vs_image = payload_e.get("ocr_vs_image", {})
    caption_align = payload_e.get("caption_alignment", {})

    injection_risk = float(injection.get("risk_score", 0.0))
    audio_align = float(cross_modal.get("consistency_score", 0.0))
    ocr_img_align = float(ocr_vs_image.get("consistency_score", 0.0))
    caption_align_score = float(caption_align.get("alignment_score", 0.0))

    final_score = _clamp(
        0.45 * injection_risk
        + 0.15 * (1.0 - ocr_conf)
        + 0.2 * (1.0 - audio_align)
        + 0.1 * (1.0 - ocr_img_align)
        + 0.1 * (1.0 - caption_align_score)
    )
    action = _action_from_score(final_score)
    explanations = [
        f"injection_risk={round(injection_risk,3)}",
        f"ocr_confidence={round(ocr_conf,3)}",
        f"audio_align={round(audio_align,3)}",
        f"ocr_vs_image={round(ocr_img_align,3)}",
        f"caption_align={round(caption_align_score,3)}",
    ]

    return {
        "ocr": payload_d.get("ocr", {}),
        "injection": injection,
        "cross_modal": cross_modal,
        "ocr_vs_image": ocr_vs_image,
        "caption_alignment": caption_align,
        "final_score": round(final_score, 3),
        "action": action,
        "explanations": explanations,
    }
