import os

import numpy as np
import pytest

from lpr.detector import PlateDetection
from lpr.ocr import OCRResult
from lpr.pipeline import LicensePlateRecognizer, PlateRecognition, annotate_image


class FakeDetector:
    def detect(self, image: np.ndarray) -> list[PlateDetection]:
        return [PlateDetection((5, 5, 35, 15), 0.9)]

    @staticmethod
    def crop(image: np.ndarray, detection: PlateDetection, padding: float = 0.08) -> np.ndarray:
        x1, y1, x2, y2 = detection.bbox
        return image[y1:y2, x1:x2]


class FakeBackend:
    name = "fake"

    def recognize(self, image: np.ndarray) -> OCRResult:
        return OCRResult("29A12345", 0.8, self.name)


def test_pipeline_detects_and_recognizes_each_plate() -> None:
    image = np.zeros((20, 40, 3), dtype=np.uint8)
    pipeline = LicensePlateRecognizer(FakeDetector(), [FakeBackend()], variants=("gray",))
    results = pipeline.recognize_image(image)
    assert len(results) == 1
    assert results[0].ocr is not None
    assert results[0].ocr.text == "29A12345"
    assert results[0].ocr.variant == "gray"


def test_pipeline_rejects_empty_backend_list() -> None:
    try:
        LicensePlateRecognizer(FakeDetector(), [])
    except ValueError as error:
        assert "backend" in str(error)
    else:
        raise AssertionError("Expected ValueError")


def test_pipeline_rejects_invalid_variants() -> None:
    with pytest.raises(ValueError, match="Unknown preprocessing"):
        LicensePlateRecognizer(FakeDetector(), [FakeBackend()], variants=("nope",))


def test_video_guards_run_before_opening_resources(tmp_path) -> None:
    pipeline = LicensePlateRecognizer(FakeDetector(), [FakeBackend()], variants=("gray",))
    video_path = tmp_path / "input.mp4"
    with pytest.raises(ValueError, match="different"):
        pipeline.recognize_video(video_path, video_path)
    with pytest.raises(ValueError, match="positive"):
        pipeline.recognize_video(video_path, tmp_path / "output.mp4", max_frames=0)


def test_video_rejects_hardlink_aliases(tmp_path) -> None:
    pipeline = LicensePlateRecognizer(FakeDetector(), [FakeBackend()], variants=("gray",))
    video_path = tmp_path / "input.mp4"
    alias_path = tmp_path / "alias.mp4"
    video_path.write_bytes(b"not a video")
    os.link(video_path, alias_path)
    with pytest.raises(ValueError, match="different"):
        pipeline.recognize_video(video_path, alias_path)


def test_annotate_image_returns_copy_with_box() -> None:
    image = np.zeros((30, 40, 3), dtype=np.uint8)
    recognition = PlateRecognition(
        PlateDetection((5, 5, 20, 20), 0.9), OCRResult("29A12345", 0.8, "fake")
    )
    annotated = annotate_image(image, [recognition])
    assert annotated.shape == image.shape
    assert not np.array_equal(annotated, image)
    assert image.sum() == 0
    assert annotated[5, 5].tolist() == [0, 220, 0]
