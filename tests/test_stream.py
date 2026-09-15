import numpy as np

from lpr.detector import PlateDetection
from lpr.ocr import OCRResult
from lpr.stream import RealtimePlateRecognizer


class FakeDetector:
    def __init__(self) -> None:
        self.calls = 0

    def detect(self, image: np.ndarray) -> list[PlateDetection]:
        return [PlateDetection((2, 2, 30, 14), 0.9)]

    def track(self, image: np.ndarray) -> list[PlateDetection]:
        self.calls += 1
        return [PlateDetection((2, 2, 30, 14), 0.9, track_id=7)]

    @staticmethod
    def crop(image: np.ndarray, detection: PlateDetection, padding: float = 0.08) -> np.ndarray:
        x1, y1, x2, y2 = detection.bbox
        return image[y1:y2, x1:x2]


class CountingBackend:
    name = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def recognize(self, image: np.ndarray) -> OCRResult:
        self.calls += 1
        return OCRResult("30A12345", 0.9, self.name)


class SequenceBackend:
    name = "fake"

    def __init__(self, results: list[OCRResult]) -> None:
        self.results = iter(results)
        self.calls = 0

    def recognize(self, image: np.ndarray) -> OCRResult:
        self.calls += 1
        return next(self.results)


def test_realtime_retries_unstable_ocr_before_stable_refresh() -> None:
    detector = FakeDetector()
    backend = SequenceBackend(
        [
            OCRResult("not-a-plate", 0.9, "fake"),
            OCRResult("30A12345", 0.9, "fake"),
        ]
    )
    recognizer = RealtimePlateRecognizer(
        detector,
        [backend],
        variants=("gray",),
        stable_votes=1,
        ocr_refresh_frames=20,
        ocr_retry_frames=2,
    )
    image = np.zeros((20, 40, 3), dtype=np.uint8)

    first = recognizer.process_frame(image, 0)
    recognizer.process_frame(image, 1)
    second = recognizer.process_frame(image, 2)

    assert first.plates[0].status == "candidate"
    assert second.plates[0].status == "stable"
    assert second.plates[0].recognition.ocr is not None
    assert second.plates[0].recognition.ocr.text == "30A12345"
    assert backend.calls == 2


def test_realtime_uses_tracking_and_stabilizes_text() -> None:
    detector = FakeDetector()
    backend = CountingBackend()
    recognizer = RealtimePlateRecognizer(
        detector,
        [backend],
        variants=("gray",),
        stable_votes=2,
        ocr_refresh_frames=1,
    )
    image = np.zeros((20, 40, 3), dtype=np.uint8)

    first = recognizer.process_frame(image, 0, 0.0)
    second = recognizer.process_frame(image, 1, 33.3)

    assert first.plates[0].status == "candidate"
    assert second.plates[0].status == "stable"
    assert second.plates[0].track_id == 7
    assert second.plates[0].recognition.ocr is not None
    assert second.plates[0].recognition.ocr.text == "30A12345"
    assert backend.calls == 2


def test_realtime_caches_stable_ocr_until_refresh_interval() -> None:
    detector = FakeDetector()
    backend = CountingBackend()
    recognizer = RealtimePlateRecognizer(
        detector,
        [backend],
        variants=("gray",),
        stable_votes=1,
        ocr_refresh_frames=10,
    )
    image = np.zeros((20, 40, 3), dtype=np.uint8)

    recognizer.process_frame(image, 0)
    recognizer.process_frame(image, 1)
    recognizer.process_frame(image, 9)
    assert backend.calls == 1

    recognizer.process_frame(image, 10)
    assert backend.calls == 2


def test_realtime_reset_forgets_track_state() -> None:
    detector = FakeDetector()
    backend = CountingBackend()
    recognizer = RealtimePlateRecognizer(detector, [backend], variants=("gray",))
    image = np.zeros((20, 40, 3), dtype=np.uint8)

    recognizer.process_frame(image, 0)
    recognizer.reset()
    recognizer.process_frame(image, 1)

    assert backend.calls == 2
