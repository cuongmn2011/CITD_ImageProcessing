import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from lpr.detector import PlateDetection
from lpr.ocr import OCRResult
from lpr.server import create_app, frame_result_payload
from lpr.stream import RealtimePlateRecognizer


class FakeDetector:
    def track(self, image: np.ndarray) -> list[PlateDetection]:
        return [PlateDetection((2, 2, 20, 12), 0.9, track_id=4)]

    @staticmethod
    def crop(image: np.ndarray, detection: PlateDetection, padding: float = 0.08) -> np.ndarray:
        x1, y1, x2, y2 = detection.bbox
        return image[y1:y2, x1:x2]


class FakeBackend:
    name = "fake"

    def recognize(self, image: np.ndarray) -> OCRResult:
        return OCRResult("30A12345", 0.9, self.name)


def _recognizer() -> RealtimePlateRecognizer:
    return RealtimePlateRecognizer(
        FakeDetector(), [FakeBackend()], variants=("gray",), stable_votes=1
    )


def test_health_reports_loaded_injected_runtime() -> None:
    with TestClient(create_app(_recognizer())) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["model_loaded"] is True
    assert response.json()["ocr_backends"] == ["paddleocr"]


def test_websocket_processes_metadata_and_binary_jpeg() -> None:
    with TestClient(create_app(_recognizer())) as client:
        with client.websocket_connect("/ws/stream") as websocket:
            websocket.send_json({"type": "config", "source_width": 40, "source_height": 20})
            assert websocket.receive_json() == {"type": "ready"}

            image = np.zeros((20, 40, 3), dtype=np.uint8)
            ok, encoded = cv2.imencode(".jpg", image)
            assert ok
            websocket.send_json(
                {
                    "type": "frame_meta",
                    "frame_id": 3,
                    "source_time_ms": 100.0,
                    "width": 40,
                    "height": 20,
                }
            )
            websocket.send_bytes(encoded.tobytes())
            result = websocket.receive_json()

    assert result["type"] == "result"
    assert result["frame_id"] == 3
    assert result["plates"][0]["text"] == "30A12345"
    assert result["plates"][0]["bbox_norm"] == [0.05, 0.1, 0.5, 0.6]


def test_frame_result_payload_rejects_invalid_dimensions() -> None:
    with TestClient(create_app(_recognizer())):
        result = _recognizer().process_frame(np.zeros((20, 40, 3), dtype=np.uint8), 1)

    try:
        frame_result_payload(result, 0, 20)
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("Expected invalid dimensions to fail")


def test_websocket_rejects_disallowed_origin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LPR_ALLOWED_ORIGINS", "http://localhost:5173")
    with TestClient(create_app(_recognizer())) as client:
        with pytest.raises(WebSocketDisconnect) as error:
            with client.websocket_connect(
                "/ws/stream", headers={"origin": "https://evil.example"}
            ):
                pass

    assert error.value.code == 1008


def test_websocket_rejects_oversized_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LPR_MAX_FRAME_BYTES", "3")
    with TestClient(create_app(_recognizer())) as client:
        with client.websocket_connect("/ws/stream") as websocket:
            websocket.send_json(
                {"type": "frame_meta", "frame_id": 0, "width": 40, "height": 20}
            )
            websocket.send_bytes(b"1234")
            assert "exceeds" in websocket.receive_json()["message"]
