from typing import List

from fastapi import FastAPI, File, Form, UploadFile

from src.engines.visual_engine import PromptInjectionEngine, THREAT_DICTIONARY

app = FastAPI(title="Engine D (Prompt Injection) API")
_ENGINE: PromptInjectionEngine | None = None


@app.on_event("startup")
def load_engine() -> None:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = PromptInjectionEngine(use_onnx=True)


@app.get("/")
def health_check() -> dict:
    return {"status": "ok", "engine": "d"}


@app.post("/analyze_d")
async def analyze_engine_d(
    image: UploadFile = File(...),
    deep: bool = Form(True),
) -> dict:
    if _ENGINE is None:
        load_engine()
    engine = _ENGINE
    image_bytes = await image.read()
    text_payload = engine.extract_text(image_bytes)
    normalized_text = text_payload["normalized_text"]
    matched = [phrase for phrase in THREAT_DICTIONARY if phrase in normalized_text]
    scores = [score for _, score in text_payload.get("scored", [])]
    ocr_confidence = float(sum(scores) / len(scores)) if scores else 0.5
    if deep:
        injection_result = engine.detect_injection_from_text(normalized_text, matched_phrases=matched)
    else:
        injection_result = {
            "is_threat": bool(matched),
            "risk_score": 0.9 if matched else 0.1,
            "reason": "FastPathRegex",
        }
    return {
        "ocr": {**text_payload, "ocr_confidence": round(ocr_confidence, 3)},
        "injection": injection_result,
    }


@app.post("/analyze_d_batch")
async def analyze_engine_d_batch(
    images: List[UploadFile] = File(...),
    deep: bool = Form(True),
) -> dict:
    if _ENGINE is None:
        load_engine()
    engine = _ENGINE
    normalized_batch: List[str] = []
    ocr_payloads: List[dict] = []
    matched_batch: List[List[str]] = []

    for img in images:
        image_bytes = await img.read()
        payload = engine.extract_text(image_bytes)
        scores = [score for _, score in payload.get("scored", [])]
        payload["ocr_confidence"] = round(float(sum(scores) / len(scores)) if scores else 0.5, 3)
        ocr_payloads.append(payload)
        normalized_text = payload["normalized_text"]
        normalized_batch.append(normalized_text)
        matched_batch.append([phrase for phrase in THREAT_DICTIONARY if phrase in normalized_text])

    results: List[dict] = []
    if deep:
        # Batch the DeBERTa pipeline to utilize parallelism.
        classifier = engine._get_injection_classifier()
        classifications = classifier(normalized_batch, top_k=1)
        for idx, classification in enumerate(classifications):
            label = str(classification.get("label", "")).upper()
            score = float(classification.get("score", 0.0))
            is_injection = "1" in label or "INJECTION" in label
            risk_score = score if is_injection else 1.0 - score
            reason = f"Model={label or 'UNKNOWN'}; model_score={score:.3f}"
            if matched_batch[idx]:
                reason += f"; matched_phrases={', '.join(sorted(set(matched_batch[idx])))}"
            results.append(
                {"is_threat": bool(is_injection), "risk_score": round(risk_score, 3), "reason": reason}
            )
    else:
        for matched in matched_batch:
            results.append(
                {
                    "is_threat": bool(matched),
                    "risk_score": 0.9 if matched else 0.1,
                    "reason": "FastPathRegex",
                }
            )

    return {"ocr": ocr_payloads, "injection": results}
