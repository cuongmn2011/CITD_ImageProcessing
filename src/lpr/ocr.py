"""OCR backends and license-plate text normalization."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

_ALLOWED_CHARACTERS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_PLATE_PATTERN = re.compile(r"^[0-9]{2}[A-Z]{1,2}[0-9]{4,6}$")


@dataclass(frozen=True, slots=True)
class OCRResult:
    text: str
    confidence: float
    backend: str
    raw_text: str = ""
    variant: str | None = None

    @property
    def valid_plate_format(self) -> bool:
        return bool(_PLATE_PATTERN.fullmatch(self.text))


class OCRBackend(Protocol):
    name: str

    def recognize(self, image: np.ndarray) -> OCRResult:
        """Recognize text from one plate image."""


def normalize_text(text: str) -> str:
    """Normalize OCR output to uppercase alphanumeric characters."""
    return "".join(character for character in text.upper() if character in _ALLOWED_CHARACTERS)


def best_result(results: list[OCRResult]) -> OCRResult | None:
    """Prefer valid plate-shaped output, then confidence, without inventing text."""
    if not results:
        return None
    valid = [result for result in results if result.valid_plate_format]
    candidates = valid or results
    return max(candidates, key=lambda result: result.confidence)


def _validate_image(image: np.ndarray) -> None:
    if image is None or image.size == 0:
        raise ValueError("Image must be non-empty")


class TesseractBackend:
    name = "tesseract"

    def __init__(self, language: str = "eng", page_segmentation_mode: int = 7) -> None:
        self.language = language
        self.page_segmentation_mode = page_segmentation_mode
        try:
            import pytesseract
        except ImportError as error:
            raise RuntimeError("Install OCR support with: uv sync --extra ocr") from error
        self._pytesseract = pytesseract

    def recognize(self, image: np.ndarray) -> OCRResult:
        _validate_image(image)
        try:
            data = self._pytesseract.image_to_data(
                image,
                lang=self.language,
                config=(
                    f"--psm {self.page_segmentation_mode} "
                    f"-c tessedit_char_whitelist={_ALLOWED_CHARACTERS}"
                ),
                output_type=self._pytesseract.Output.DICT,
            )
        except self._pytesseract.TesseractNotFoundError as error:
            raise RuntimeError("Tesseract binary is not installed or not on PATH") from error
        texts: list[str] = []
        confidences: list[float] = []
        for raw_text, raw_confidence in zip(data.get("text", []), data.get("conf", [])):
            text = str(raw_text).strip()
            try:
                confidence = float(raw_confidence)
            except (TypeError, ValueError):
                continue
            if text and confidence >= 0:
                texts.append(text)
                confidences.append(confidence / 100.0)
        raw_text = " ".join(texts)
        return OCRResult(
            normalize_text(raw_text),
            float(np.mean(confidences)) if confidences else 0.0,
            self.name,
            raw_text,
        )


class EasyOCRBackend:
    name = "easyocr"

    def __init__(self, languages: list[str] | tuple[str, ...] = ("en",), gpu: bool = False) -> None:
        try:
            import easyocr
        except ImportError as error:
            raise RuntimeError("Install OCR support with: uv sync --extra ocr") from error
        self._reader = easyocr.Reader(list(languages), gpu=gpu)

    def recognize(self, image: np.ndarray) -> OCRResult:
        _validate_image(image)
        detections = self._reader.readtext(
            image,
            detail=1,
            paragraph=False,
            allowlist=_ALLOWED_CHARACTERS,
        )
        ordered = sorted(detections, key=lambda item: min(point[0] for point in item[0]))
        texts = [
            str(item[1]).strip() for item in ordered if len(item) >= 3 and str(item[1]).strip()
        ]
        confidences = [
            float(item[2]) for item in ordered if len(item) >= 3 and str(item[1]).strip()
        ]
        raw_text = " ".join(texts)
        return OCRResult(
            normalize_text(raw_text),
            float(np.mean(confidences)) if confidences else 0.0,
            self.name,
            raw_text,
        )


class PaddleOCRBackend:
    """Adapter for PaddleOCR's current ``predict`` API."""

    name = "paddleocr"

    def __init__(self, language: str = "en") -> None:
        try:
            from paddleocr import PaddleOCR
        except ImportError as error:
            raise RuntimeError("Install PaddleOCR separately for this optional backend") from error
        self._ocr = PaddleOCR(
            lang=language,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    def recognize(self, image: np.ndarray) -> OCRResult:
        _validate_image(image)
        prediction = next(iter(self._ocr.predict(image)), None)
        if prediction is None:
            return OCRResult("", 0.0, self.name)
        texts, scores = _paddle_text_and_scores(prediction)
        raw_text = " ".join(texts)
        return OCRResult(
            normalize_text(raw_text), float(np.mean(scores)) if scores else 0.0, self.name, raw_text
        )


def _paddle_text_and_scores(prediction: Any) -> tuple[list[str], list[float]]:
    """Handle PaddleOCR result objects and dicts without binding pipeline internals."""

    def get(name: str, default: Any) -> Any:
        if isinstance(prediction, dict):
            return prediction.get(name, default)
        return getattr(prediction, name, default)

    raw_texts = list(get("rec_texts", []) or [])
    raw_scores = list(get("rec_scores", []) or [])
    pairs: list[tuple[str, float]] = []
    for index, raw_text in enumerate(raw_texts):
        if raw_text is None:
            continue
        text = str(raw_text).strip()
        if not text:
            continue
        try:
            score = float(raw_scores[index]) if index < len(raw_scores) else 0.0
        except (TypeError, ValueError):
            score = 0.0
        pairs.append((text, score))
    return [text for text, _ in pairs], [score for _, score in pairs]
