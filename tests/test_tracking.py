import numpy as np
import pytest

from lpr.detector import PlateDetection
from lpr.stream import RealtimePlateRecognizer
from lpr.tracking import CentroidTracker, CentroidTrackingDetector

# Real YOLO boxes from a street video, frames 122-133: a motorbike plate moving up at about a
# third of its width per frame, and a car plate further up. Frames 124 and 130 repeat the
# frame before; the car plate was missed on frame 132.
MOTORBIKE = [
    (1011, 1003, 1077, 1056), (1002, 985, 1070, 1034), (1002, 985, 1070, 1034),
    (991, 940, 1056, 992), (985, 920, 1050, 972), (979, 900, 1040, 951),
    (975, 880, 1035, 929), (969, 868, 1029, 911), (969, 868, 1029, 911),
    (954, 826, 1019, 874), (954, 807, 1013, 855), (951, 794, 1008, 840),
]
CAR = [
    (1106, 182, 1184, 209), (1108, 176, 1184, 204), (1108, 176, 1184, 204),
    (1113, 165, 1185, 189), (1110, 160, 1183, 183), (1114, 153, 1181, 179),
    (1115, 146, 1180, 171), (1116, 143, 1182, 167), (1116, 143, 1182, 167),
    (1116, 132, 1186, 158), None, (1117, 123, 1183, 147),
]


def _plate(bbox: tuple[int, int, int, int], confidence: float = 0.6) -> PlateDetection:
    return PlateDetection(bbox, confidence)


def test_follows_real_plates_through_repeated_and_missed_frames() -> None:
    tracker = CentroidTracker()
    ids: dict[str, set[int]] = {"motorbike": set(), "car": set()}

    for motorbike, car in zip(MOTORBIKE, CAR):
        detections = [_plate(motorbike)] + ([_plate(car)] if car else [])
        tracked = tracker.update(detections)
        ids["motorbike"].add(tracked[0].track_id)
        if car:
            ids["car"].add(tracked[1].track_id)

    assert ids == {"motorbike": {1}, "car": {2}}


def test_keeps_detection_order_and_fields() -> None:
    tracked = CentroidTracker().update([_plate((0, 0, 50, 20), 0.7), _plate((500, 0, 550, 20))])

    assert [(item.bbox, item.confidence, item.track_id) for item in tracked] == [
        ((0, 0, 50, 20), 0.7, 1),
        ((500, 0, 550, 20), 0.6, 2),
    ]


def test_starts_a_new_track_on_a_far_jump_or_a_size_change() -> None:
    tracker = CentroidTracker()
    first = tracker.update([_plate((0, 0, 50, 20))])[0].track_id
    jumped = tracker.update([_plate((200, 0, 250, 20))])[0].track_id  # 4 widths away
    resized = tracker.update([_plate((200, 0, 320, 48))])[0].track_id  # 2.4x wider

    assert len({first, jumped, resized}) == 3


def test_predicts_motion_across_a_missed_frame() -> None:
    tracker = CentroidTracker()
    for x in (0, 50, 100):  # one plate width per frame
        tracker.update([_plate((x, 0, x + 50, 20))])
    tracker.update([])

    # Two widths from the last sighting, but where its velocity says it should be.
    assert tracker.update([_plate((200, 0, 250, 20))])[0].track_id == 1


def test_drops_a_track_after_max_age_updates_unseen() -> None:
    kept = CentroidTracker(max_age=2)
    dropped = CentroidTracker(max_age=2)
    for tracker, misses in ((kept, 1), (dropped, 2)):
        tracker.update([_plate((0, 0, 50, 20))])
        for _ in range(misses):
            tracker.update([])

    assert kept.update([_plate((0, 0, 50, 20))])[0].track_id == 1
    assert dropped.update([_plate((0, 0, 50, 20))])[0].track_id == 2


def test_rejects_invalid_settings() -> None:
    with pytest.raises(ValueError):
        CentroidTracker(max_distance=0)


class FakeBackend:
    name = "fake"


class FakeDetector:
    def __init__(self) -> None:
        self.x = 0

    def detect(self, image: np.ndarray) -> list[PlateDetection]:
        self.x += 10
        return [_plate((self.x, 5, self.x + 30, 15))]

    @staticmethod
    def crop(image: np.ndarray, detection: PlateDetection, padding: float = 0.08) -> np.ndarray:
        x1, y1, x2, y2 = detection.bbox
        return image[y1:y2, x1:x2]


def test_tracking_detector_ids_detections_and_resets_with_the_recognizer() -> None:
    detector = CentroidTrackingDetector(FakeDetector())
    image = np.zeros((40, 200, 3), dtype=np.uint8)

    assert [detector.track(image)[0].track_id for _ in range(3)] == [1, 1, 1]
    assert detector.detect(image)[0].track_id is None
    assert detector.crop(image, _plate((10, 5, 40, 15))).shape == (10, 30, 3)

    assert detector.tracker.update([_plate((500, 0, 530, 10))])[0].track_id == 2
    RealtimePlateRecognizer(detector, [FakeBackend()]).reset()
    assert detector.tracker.update([_plate((0, 0, 30, 10))])[0].track_id == 1
