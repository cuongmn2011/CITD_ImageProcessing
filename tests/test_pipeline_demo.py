import threading
import time
from runpy import run_path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from lpr.detector import PlateDetection
from lpr.ocr import OCRResult
from lpr.pipeline import LicensePlateRecognizer
from lpr.stream import RealtimePlateRecognizer

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


class FakeTrackingDetector(FakeDetector):
    def track(self, image: np.ndarray) -> list[PlateDetection]:
        return [PlateDetection((1, 1, 20, 10), 0.9, 0, 3)]


def _video_factory(started: threading.Event | None = None, release: threading.Event | None = None):
    def make(variants: tuple[str, ...]) -> RealtimePlateRecognizer:
        if started is not None and release is not None:
            started.set()
            release.wait(timeout=10)
        return RealtimePlateRecognizer(FakeTrackingDetector(), [FakeBackend()], variants=variants)

    return make


def _video_bytes(tmp_path) -> bytes:
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    assert writer.isOpened()
    for index in range(8):
        writer.write(np.full((48, 64, 3), index * 25, dtype=np.uint8))
    writer.release()
    return path.read_bytes()


def _wait_done(client: TestClient, job_id: str) -> dict:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        job = client.get(f"/api/video/{job_id}").json()
        if job["state"] != "running":
            return job
        time.sleep(0.05)
    raise AssertionError("video job did not finish")


def test_video_job_reports_tracks_and_scores(tmp_path) -> None:
    jobs_dir = tmp_path / "jobs"
    client = TestClient(create_app(_recognizer(), _video_factory(), jobs_dir))

    response = client.post(
        "/api/video?stride=1&variants=raw&filename=clip.mp4", content=_video_bytes(tmp_path)
    )
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    job = _wait_done(client, job_id)

    assert job["state"] == "done", job["error"]
    assert job["frames_processed"] == 8
    assert [(track["track_id"], track["text"]) for track in job["tracks"]] == [(3, "51G48154")]
    assert client.get(f"/api/video/{job_id}/video").status_code == 200
    crop = client.get(f"/api/video/{job_id}/crop/3.jpg")
    assert crop.status_code == 200 and crop.headers["content-type"] == "image/jpeg"

    score = client.post(
        f"/api/video/{job_id}/score", json={"truths": "51G-481.54\n30A11111"}
    ).json()
    assert (score["correct"], score["wrong"], score["missed"]) == (1, 0, 1)
    assert (jobs_dir / job_id / "score.json").is_file()

    reloaded = TestClient(create_app(_recognizer(), _video_factory(), jobs_dir))
    assert reloaded.get(f"/api/video/{job_id}").json()["state"] == "done"


def test_video_upload_rejects_second_job_while_running(tmp_path) -> None:
    started, release = threading.Event(), threading.Event()
    client = TestClient(create_app(_recognizer(), _video_factory(started, release), tmp_path))
    video = _video_bytes(tmp_path)

    first = client.post("/api/video?filename=a.mp4", content=video)
    assert first.status_code == 200
    assert started.wait(timeout=10)
    try:
        assert client.post("/api/video?filename=b.mp4", content=video).status_code == 409
        assert client.post("/api/detect", content=_jpeg_bytes()).status_code == 409
    finally:
        release.set()
    assert _wait_done(client, first.json()["job_id"])["state"] == "done"


def test_video_upload_validates_input(tmp_path) -> None:
    client = TestClient(create_app(_recognizer(), _video_factory(), tmp_path))

    assert client.post("/api/video?variants=bogus", content=b"x").status_code == 400
    assert client.post("/api/video?stride=0", content=b"x").status_code == 400
    assert client.post("/api/video", content=b"").status_code == 400
    assert client.get("/api/video/missing").status_code == 404


def test_video_mode_disabled_without_factory(tmp_path) -> None:
    client = TestClient(create_app(_recognizer(), None, tmp_path))

    assert client.post("/api/video", content=b"x").status_code == 503
