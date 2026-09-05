from types import SimpleNamespace

from lpr.detector import detections_from_result


class Scalar:
    def __init__(self, value: float) -> None:
        self.value = value

    def item(self) -> float:
        return self.value


class FakeBox:
    xyxy = [SimpleNamespace(tolist=lambda: [1.2, -2.0, 14.8, 9.9])]
    conf = [Scalar(0.91)]
    cls = [Scalar(0)]
    id = [Scalar(7)]


class EmptyBox:
    xyxy = [SimpleNamespace(tolist=lambda: [5, 5, 5, 9])]
    conf = [Scalar(0.5)]
    cls = [Scalar(0)]
    id = None


class OutsideBox:
    xyxy = [SimpleNamespace(tolist=lambda: [-20, -10, -5, -1])]
    conf = [Scalar(0.5)]
    cls = [Scalar(0)]
    id = None


def test_detections_are_clamped_and_normalized() -> None:
    result = SimpleNamespace(boxes=[FakeBox()])
    detections = detections_from_result(result, (10, 15, 3))
    assert detections[0].bbox == (1, 0, 15, 10)
    assert detections[0].confidence == 0.91
    assert detections[0].track_id == 7


def test_invalid_boxes_are_ignored() -> None:
    result = SimpleNamespace(boxes=[EmptyBox()])
    assert detections_from_result(result, (10, 15, 3)) == []


def test_boxes_outside_image_are_ignored() -> None:
    result = SimpleNamespace(boxes=[OutsideBox()])
    assert detections_from_result(result, (10, 15, 3)) == []


class TinyBox:
    xyxy = [SimpleNamespace(tolist=lambda: [10.1, 4.2, 10.9, 4.8])]
    conf = [Scalar(0.8)]
    cls = [Scalar(0)]
    id = None


def test_subpixel_boxes_are_rounded_outward() -> None:
    result = SimpleNamespace(boxes=[TinyBox()])
    detections = detections_from_result(result, (10, 15, 3))
    assert detections[0].bbox == (10, 4, 11, 5)


def test_missing_boxes_returns_empty_list() -> None:
    assert detections_from_result(SimpleNamespace(boxes=None), (10, 15, 3)) == []
