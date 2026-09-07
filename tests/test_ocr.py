import sys

import numpy as np

from lpr.ocr import (
    OCRResult,
    TesseractBackend,
    _paddle_text_and_scores,
    best_result,
    normalize_text,
)


def test_normalize_text_removes_separators_and_noise() -> None:
    assert normalize_text("  29A-123.45\n") == "29A12345"
    assert normalize_text("51f@12 o45") == "51F12O45"


def test_best_result_prefers_valid_plate_format() -> None:
    results = [
        OCRResult("29A12345", 0.72, "easyocr"),
        OCRResult("ABC", 0.99, "tesseract"),
    ]
    assert best_result(results) == results[0]


def test_best_result_returns_none_for_empty_input() -> None:
    assert best_result([]) is None


def test_valid_plate_format_is_exposed() -> None:
    assert OCRResult("51F12345", 0.8, "test").valid_plate_format
    assert not OCRResult("ABC", 0.8, "test").valid_plate_format


def test_best_result_rejects_invalid_ocr_text() -> None:
    assert best_result([OCRResult("LEE", 0.99, "tesseract")]) is None


def test_tesseract_defaults_to_two_line_plate_segmentation(monkeypatch) -> None:
    calls: dict[str, str] = {}

    class FakePytesseract:
        class Output:
            DICT = object()

        class TesseractNotFoundError(Exception):
            pass

        @staticmethod
        def image_to_data(image, *, lang, config, output_type):
            calls["config"] = config
            return {"text": ["80A", "005.38"], "conf": ["80", "90"]}

    monkeypatch.setitem(sys.modules, "pytesseract", FakePytesseract)
    result = TesseractBackend().recognize(np.zeros((20, 40, 3), dtype=np.uint8))

    assert "--psm 6" in calls["config"]
    assert result.text == "80A00538"


def test_paddle_text_and_scores_stay_aligned() -> None:
    texts, scores = _paddle_text_and_scores({"rec_texts": ["", "ABC"], "rec_scores": [0.1, 0.9]})
    assert texts == ["ABC"]
    assert scores == [0.9]
