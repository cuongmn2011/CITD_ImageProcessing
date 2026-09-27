from runpy import run_path
from types import SimpleNamespace

import cv2
import numpy as np
from fastapi.testclient import TestClient

from lpr.ocr import OCRResult

create_app = run_path("scripts/ocr-demo.py")["create_app"]


class FakeBackend:
    def recognize(self, image: np.ndarray) -> OCRResult:
        return OCRResult("51G48154", 0.99, "fake", "51G 48154")


def _png_bytes() -> bytes:
    ok, encoded = cv2.imencode(".png", np.zeros((10, 30, 3), dtype=np.uint8))
    assert ok
    return encoded.tobytes()


def test_index_serves_upload_page() -> None:
    client = TestClient(create_app(FakeBackend()))

    response = client.get("/")

    assert response.status_code == 200
    assert "/api/ocr" in response.text


def test_ocr_endpoint_returns_recognized_text() -> None:
    client = TestClient(create_app(FakeBackend()))

    response = client.post("/api/ocr", content=_png_bytes())

    assert response.status_code == 200
    assert response.json() == {"text": "51G48154", "confidence": 0.99, "raw_text": "51G 48154"}


def test_ocr_endpoint_rejects_non_image_and_empty_bodies() -> None:
    client = TestClient(create_app(SimpleNamespace()))

    assert client.post("/api/ocr", content=b"not an image").status_code == 400
    assert client.post("/api/ocr", content=b"").status_code == 400
