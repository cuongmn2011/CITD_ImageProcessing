from runpy import run_path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from lpr.detector import PlateDetection
from lpr.ocr import OCRResult
from lpr.pipeline import LicensePlateRecognizer

create_app = run_path("scripts/pipeline-demo.py")["create_app"]


class FakeDetector:
    def detect(self, image: np.ndarray) -> list[PlateDetection]:
        return [PlateDetection((1, 1, 5, 5), 0.9)]

    @staticmethod
    def crop(image: np.ndarray, detection: PlateDetection, padding: float = 0.08) -> np.ndarray:
        return image


class FakeBackend:
    name = "fake"

    def recognize(self, image: np.ndarray) -> OCRResult:
        return OCRResult("51G48154", 0.98, self.name)


def _recognizer() -> LicensePlateRecognizer:
    return LicensePlateRecognizer(FakeDetector(), [FakeBackend()])


def _jpeg_bytes() -> bytes:
    ok, encoded = cv2.imencode(".jpg", np.zeros((20, 40, 3), dtype=np.uint8))
    assert ok
    return encoded.tobytes()


def test_index_serves_upload_page() -> None:
    client = TestClient(create_app(_recognizer()))

    response = client.get("/")

    assert response.status_code == 200
    assert "/api/detect" in response.text


def test_detect_endpoint_returns_plates_and_annotated_image() -> None:
    client = TestClient(create_app(_recognizer()))

    response = client.post("/api/detect", content=_jpeg_bytes())

    assert response.status_code == 200
    payload = response.json()
    assert payload["plates"] == [
        {
            "bbox": [1, 1, 5, 5],
            "detection_confidence": 0.9,
            "text": "51G48154",
            "ocr_confidence": 0.98,
        }
    ]
    assert payload["annotated_image"]


def test_detect_endpoint_rejects_non_image_and_empty_bodies() -> None:
    client = TestClient(create_app(_recognizer()))

    assert client.post("/api/detect", content=b"not an image").status_code == 400
    assert client.post("/api/detect", content=b"").status_code == 400
