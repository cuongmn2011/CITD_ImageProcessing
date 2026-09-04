
from lpr.ocr import OCRResult, best_result, normalize_text


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


