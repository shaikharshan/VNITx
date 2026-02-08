import io
import os
import re
import urllib.request
from typing import Any, Dict, Iterable, List, Tuple, Union

import numpy as np
import cv2
from PIL import Image
import easyocr
from sentence_transformers import SentenceTransformer
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BlipForConditionalGeneration,
    BlipProcessor,
    pipeline,
)

try:
    from optimum.onnxruntime import ORTModelForSequenceClassification

    _HAS_ORT = True
except Exception:
    _HAS_ORT = False


THREAT_DICTIONARY = [
    "ignore previous",
    "system override",
    "transfer funds",
    "bypass safety",
    "disable guardrails",
    "override policy",
    "reveal secrets",
]


class PromptInjectionEngine:
    def __init__(
        self,
        use_onnx: bool | None = None,
        force_cpu: bool | None = None,
        model_name: str | None = None,
    ) -> None:
        os.environ.setdefault("HF_HUB_TIMEOUT", "60")
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        self._ocr: easyocr.Reader | None = None
        self._injection_classifier = None
        self._model_name = model_name or "protectai/deberta-v3-base-prompt-injection"
        if force_cpu is None:
            self._force_cpu = os.environ.get("SENTINEL_FORCE_CPU", "").lower() in {
                "1",
                "true",
                "yes",
            }
        else:
            self._force_cpu = force_cpu
        if use_onnx is None:
            self._use_onnx = os.environ.get("SENTINEL_USE_ONNX", "1") not in {"0", "false"}
        else:
            self._use_onnx = use_onnx

    def _get_ocr(self) -> easyocr.Reader:
        if self._ocr is None:
            ocr_gpu = os.environ.get("SENTINEL_OCR_GPU", "1") not in {"0", "false"}
            try:
                self._ocr = easyocr.Reader(["en"], gpu=ocr_gpu)
            except Exception:
                self._ocr = easyocr.Reader(["en"], gpu=False)
        return self._ocr

    def _get_injection_classifier(self):
        if self._injection_classifier is None:
            if self._use_onnx and _HAS_ORT:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        self._model_name, subfolder="onnx", local_files_only=True
                    )
                    model = ORTModelForSequenceClassification.from_pretrained(
                        self._model_name, subfolder="onnx", export=False, local_files_only=True
                    )
                except Exception:
                    tokenizer = AutoTokenizer.from_pretrained(self._model_name, subfolder="onnx")
                    model = ORTModelForSequenceClassification.from_pretrained(
                        self._model_name, subfolder="onnx", export=False
                    )
                self._injection_classifier = pipeline(
                    "text-classification",
                    model=model,
                    tokenizer=tokenizer,
                    truncation=True,
                    max_length=512,
                )
            else:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        self._model_name, local_files_only=True
                    )
                    model = AutoModelForSequenceClassification.from_pretrained(
                        self._model_name, local_files_only=True
                    )
                except Exception:
                    tokenizer = AutoTokenizer.from_pretrained(self._model_name)
                    model = AutoModelForSequenceClassification.from_pretrained(self._model_name)
                device = torch.device(
                    "cpu"
                    if self._force_cpu or not torch.backends.mps.is_available()
                    else "mps"
                )
                self._injection_classifier = pipeline(
                    "text-classification",
                    model=model,
                    tokenizer=tokenizer,
                    truncation=True,
                    max_length=512,
                    device=device,
                )
        return self._injection_classifier

    @staticmethod
    def _normalize_text(text: str) -> str:
        lowered = text.lower()
        cleaned = re.sub(r"[^a-z0-9]+", " ", lowered)
        tokens = cleaned.split()

        def merge_single_letter_runs(items: Iterable[str]) -> List[str]:
            merged: List[str] = []
            run: List[str] = []
            for token in items:
                if len(token) == 1:
                    run.append(token)
                    continue
                if run:
                    merged.append("".join(run))
                    run = []
                merged.append(token)
            if run:
                merged.append("".join(run))
            return merged

        merged_tokens = merge_single_letter_runs(tokens)
        return " ".join(merged_tokens)

    @staticmethod
    def _load_image_for_ocr(image: Union[str, bytes]) -> Union[str, np.ndarray]:
        if isinstance(image, str):
            return image
        pil_image = Image.open(io.BytesIO(image)).convert("RGB")
        return np.array(pil_image)

    @staticmethod
    def _enhance_for_hidden_text(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)

    @staticmethod
    def _load_image_for_clip(image: Union[str, bytes]) -> Image.Image:
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        return Image.open(io.BytesIO(image)).convert("RGB")

    @staticmethod
    def _extract_ocr_text(ocr_result: List[Any]) -> Tuple[str, List[Tuple[str, float]]]:
        fragments: List[str] = []
        scored: List[Tuple[str, float]] = []
        # EasyOCR returns: [([bbox], text, confidence), ...]
        for line in ocr_result or []:
            if not line or len(line) < 2:
                continue
            text = str(line[1])
            score = float(line[2]) if len(line) > 2 and isinstance(line[2], (float, int)) else None
            if text:
                fragments.append(text)
                if score is not None:
                    scored.append((text, score))
        return " ".join(fragments), scored

    def detect_injection(self, image: Union[str, bytes]) -> Dict[str, Any]:
        text_payload = self.extract_text(image)
        return self.detect_injection_from_text(
            text_payload["normalized_text"],
            matched_phrases=[
                phrase for phrase in THREAT_DICTIONARY if phrase in text_payload["normalized_text"]
            ],
        )

    def detect_injection_from_text(
        self, normalized_text: str, matched_phrases: List[str] | None = None
    ) -> Dict[str, Any]:
        if not normalized_text:
            return {
                "is_threat": False,
                "risk_score": 0.0,
                "reason": "No readable text detected in image.",
            }

        matched = matched_phrases or [
            phrase for phrase in THREAT_DICTIONARY if phrase in normalized_text
        ]

        try:
            classifier = self._get_injection_classifier()
            classification = classifier(normalized_text, top_k=1)[0]
            label = str(classification.get("label", "")).upper()
            score = float(classification.get("score", 0.0))
            is_injection = "1" in label or "INJECTION" in label
            risk_score = score if is_injection else 1.0 - score
            reason_parts = [
                f"Model={label or 'UNKNOWN'}",
                f"model_score={score:.3f}",
            ]
        except Exception:
            is_injection = bool(matched)
            risk_score = 0.9 if matched else 0.1
            reason_parts = ["Model=FALLBACK", "model_score=0.0"]
        if matched:
            reason_parts.append(f"matched_phrases={', '.join(sorted(set(matched)))}")

        return {
            "is_threat": bool(is_injection),
            "risk_score": round(risk_score, 3),
            "reason": "; ".join(reason_parts),
        }

    def extract_text(self, image: Union[str, bytes]) -> Dict[str, Any]:
        ocr_input = self._load_image_for_ocr(image)
        reader = self._get_ocr()
        if isinstance(ocr_input, str):
            ocr_result = reader.readtext(ocr_input)
            raw_text, scored = self._extract_ocr_text(ocr_result)
            normalized = self._normalize_text(raw_text)
        else:
            base_result = reader.readtext(ocr_input)
            enhanced_image = self._enhance_for_hidden_text(ocr_input)
            enhanced_result = reader.readtext(enhanced_image)
            raw_text_base, scored_base = self._extract_ocr_text(base_result)
            raw_text_enh, scored_enh = self._extract_ocr_text(enhanced_result)
            raw_text = " ".join([raw_text_base, raw_text_enh]).strip()
            scored = scored_base + scored_enh
            normalized = self._normalize_text(raw_text)
        return {
            "raw_text": raw_text,
            "normalized_text": normalized,
            "scored": scored,
        }


class CrossModalEngine:
    def __init__(self, clip_model: str | None = None, caption_model: str | None = None) -> None:
        self._clip = SentenceTransformer(
            clip_model or os.environ.get("SENTINEL_CLIP_MODEL", "clip-ViT-B-32")
        )
        self._captioner = None
        self._caption_model = caption_model or os.environ.get(
            "SENTINEL_BLIP_MODEL", "Salesforce/blip-image-captioning-base"
        )

    @staticmethod
    def _load_image_for_clip(image: Union[str, bytes]) -> Image.Image:
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        return Image.open(io.BytesIO(image)).convert("RGB")

    def _get_captioner(self):
        if self._captioner is None:
            # Use BLIP processor + model directly to avoid pipeline task mismatches.
            processor = BlipProcessor.from_pretrained(self._caption_model)
            model = BlipForConditionalGeneration.from_pretrained(self._caption_model)
            device = os.environ.get("SENTINEL_BLIP_DEVICE", "cpu")
            model.to(device)
            self._captioner = (processor, model, device)
        return self._captioner

    def check_cross_modal(self, image: Union[str, bytes], audio_transcript: str) -> Dict[str, Any]:
        if not audio_transcript:
            return {"is_mismatch": True, "consistency_score": 0.0}

        pil_image = self._load_image_for_clip(image)
        image_emb = self._clip.encode([pil_image], normalize_embeddings=True)
        text_emb = self._clip.encode([audio_transcript], normalize_embeddings=True)
        similarity = float(np.dot(image_emb[0], text_emb[0]))

        return {
            "is_mismatch": similarity < 0.18,
            "consistency_score": round(similarity, 4),
        }

    def check_ocr_vs_image(self, image: Union[str, bytes], ocr_text: str) -> Dict[str, Any]:
        if not ocr_text:
            return {"is_mismatch": False, "consistency_score": 0.0}
        pil_image = self._load_image_for_clip(image)
        image_emb = self._clip.encode([pil_image], normalize_embeddings=True)
        text_emb = self._clip.encode([ocr_text], normalize_embeddings=True)
        similarity = float(np.dot(image_emb[0], text_emb[0]))
        return {
            "is_mismatch": similarity < 0.18,
            "consistency_score": round(similarity, 4),
        }

    def check_caption_alignment(self, image: Union[str, bytes], ocr_text: str) -> Dict[str, Any]:
        if not ocr_text:
            return {"caption": "", "alignment_score": 0.0}
        pil_image = self._load_image_for_clip(image)
        processor, model, device = self._get_captioner()
        inputs = processor(images=pil_image, return_tensors="pt").to(device)
        output_ids = model.generate(**inputs, max_new_tokens=30)
        caption = processor.decode(output_ids[0], skip_special_tokens=True)
        text_emb = self._clip.encode([ocr_text], normalize_embeddings=True)
        caption_emb = self._clip.encode([caption], normalize_embeddings=True)
        similarity = float(np.dot(text_emb[0], caption_emb[0]))
        return {"caption": caption, "alignment_score": round(similarity, 4)}


class VisualSecurityEngine:
    def __init__(
        self,
        use_onnx: bool | None = None,
        force_cpu: bool | None = None,
        clip_model: str | None = None,
    ) -> None:
        self.engine_d = PromptInjectionEngine(use_onnx=use_onnx, force_cpu=force_cpu)
        self.engine_e = CrossModalEngine(clip_model=clip_model)

    def extract_text(self, image: Union[str, bytes]) -> Dict[str, Any]:
        return self.engine_d.extract_text(image)

    def detect_injection(self, image: Union[str, bytes]) -> Dict[str, Any]:
        return self.engine_d.detect_injection(image)

    def detect_injection_from_text(
        self, normalized_text: str, matched_phrases: List[str] | None = None
    ) -> Dict[str, Any]:
        return self.engine_d.detect_injection_from_text(normalized_text, matched_phrases)

    def check_cross_modal(self, image: Union[str, bytes], audio_transcript: str) -> Dict[str, Any]:
        return self.engine_e.check_cross_modal(image, audio_transcript)

    def check_ocr_vs_image(self, image: Union[str, bytes], ocr_text: str) -> Dict[str, Any]:
        return self.engine_e.check_ocr_vs_image(image, ocr_text)

    def check_caption_alignment(self, image: Union[str, bytes], ocr_text: str) -> Dict[str, Any]:
        return self.engine_e.check_caption_alignment(image, ocr_text)


def _download_demo_image() -> bytes:
    demo_urls = [
        "https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/7/74/A-Cat.jpg",
    ]
    headers = {"User-Agent": "Mozilla/5.0 (Sentinel-X demo)"}
    last_error: Exception | None = None
    for url in demo_urls:
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=20) as response:
                return response.read()
        except Exception as exc:  # pragma: no cover - best effort demo download
            last_error = exc
            continue
    raise RuntimeError(f"Failed to download demo image: {last_error}")


if __name__ == "__main__":
    demo_bytes = _download_demo_image()

    engine = VisualSecurityEngine()
    injection_result = engine.detect_injection(demo_bytes)
    cross_modal_result = engine.check_cross_modal(demo_bytes, "a cat sitting on a ledge")

    print("Injection detection:", injection_result)
    print("Cross-modal consistency:", cross_modal_result)
