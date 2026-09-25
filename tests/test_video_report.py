import cv2
import numpy as np
import pytest

from lpr.detector import PlateDetection
from lpr.ocr import OCRResult
from lpr.stream import RealtimePlateRecognizer
from lpr.video_report import (
    TrackSummary,
    merge_fragments,
    process_video,
    score_against_ground_truth,
)


class FakeTrackingDetector:
    def __init__(self, track_id: int | None = 7) -> None:
        self.track_id = track_id
        self.calls = 0

    def detect(self, image: np.ndarray) -> list[PlateDetection]:
        return self.track(image)

    def track(self, image: np.ndarray) -> list[PlateDetection]:
        self.calls += 1
        return [PlateDetection((5, 5, 40, 20), 0.9, 0, self.track_id)]

    @staticmethod
    def crop(image: np.ndarray, detection: PlateDetection, padding: float = 0.08) -> np.ndarray:
        x1, y1, x2, y2 = detection.bbox
        return image[y1:y2, x1:x2].copy()


class FakeBackend:
    name = "fake"

    def recognize(self, image: np.ndarray) -> OCRResult:
        return OCRResult("51G48154", 0.95, self.name)


def _write_video(path, frames: int = 10) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    assert writer.isOpened()
    for index in range(frames):
        writer.write(np.full((48, 64, 3), (index * 20) % 256, dtype=np.uint8))
    writer.release()


def _recognizer(detector: FakeTrackingDetector | None = None) -> RealtimePlateRecognizer:
    return RealtimePlateRecognizer(
        detector or FakeTrackingDetector(), [FakeBackend()], variants=("raw",)
    )


def test_process_video_summarizes_one_stable_track(tmp_path) -> None:
    source = tmp_path / "in.mp4"
    _write_video(source)
    progress: list[int] = []

    report = process_video(
        _recognizer(),
        source,
        tmp_path / "out.webm",
        on_progress=lambda done, total, tracks: progress.append(done),
    )

    assert report.frames_processed == 10
    assert report.output_path.is_file()
    assert progress == list(range(1, 11))
    assert len(report.tracks) == 1
    track = report.tracks[0]
    assert (track.track_id, track.text, track.stable) == (7, "51G48154", True)
    assert (track.first_frame, track.last_frame) == (0, 9)
    assert track.crop is not None and track.crop.shape[:2] == (15, 35)


def test_process_video_honors_stride(tmp_path) -> None:
    source = tmp_path / "in.mp4"
    _write_video(source)

    report = process_video(_recognizer(), source, tmp_path / "out.mp4", stride=2)

    assert report.frames_processed == 5
    assert report.frames_total == 10


def test_process_video_skips_untracked_detections(tmp_path) -> None:
    source = tmp_path / "in.mp4"
    _write_video(source)

    report = process_video(
        _recognizer(FakeTrackingDetector(track_id=None)), source, tmp_path / "out.mp4"
    )

    assert report.tracks == ()


def test_process_video_scales_the_annotated_copy_down(tmp_path) -> None:
    source = tmp_path / "in.mp4"
    _write_video(source)
    shown: list[tuple[int, ...]] = []

    report = process_video(
        _recognizer(),
        source,
        tmp_path / "out.mp4",
        max_output_width=32,
        on_frame=lambda image, index: shown.append(image.shape),
    )

    capture = cv2.VideoCapture(str(report.output_path))
    size = (capture.get(cv2.CAP_PROP_FRAME_WIDTH), capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    assert size == (32, 24)
    assert shown[0] == (24, 32, 3)
    # Recognition still ran on the full-size frame: the crop keeps full resolution.
    assert report.tracks[0].crop.shape[:2] == (15, 35)


def test_process_video_rejects_invalid_stride(tmp_path) -> None:
    with pytest.raises(ValueError, match="stride"):
        process_video(_recognizer(), tmp_path / "in.mp4", tmp_path / "out.mp4", stride=0)


def test_score_matches_one_to_one_and_classifies_reads() -> None:
    result = score_against_ground_truth(
        predictions=["51G48154", "605140494", "68", "51G48154"],
        truths=["51G-481.54", "60F140494", "86H79591"],
    )

    assert [detail["status"] for detail in result["details"]] == ["correct", "wrong", "missed"]
    assert (result["correct"], result["wrong"], result["missed"]) == (1, 1, 1)
    assert result["extra"] == ["68"]
    assert result["samples"] == 3
    assert result["exact_accuracy"] == pytest.approx(1 / 3)
    assert result["cer"] == pytest.approx((0 + 1 + 8) / (8 + 9 + 8))


def test_score_requires_ground_truth() -> None:
    with pytest.raises(ValueError, match="at least one plate"):
        score_against_ground_truth(["51G48154"], ["", "  "])


def test_process_video_limits_to_a_segment_and_streams_frames(tmp_path) -> None:
    source = tmp_path / "in.mp4"
    _write_video(source, frames=20)  # 10 fps, 2 s
    frames: list[int] = []

    report = process_video(
        _recognizer(),
        source,
        tmp_path / "out.mp4",
        start_seconds=0.5,
        duration_seconds=0.8,
        on_frame=lambda image, index: frames.append(index),
    )

    assert frames == list(range(5, 13))
    assert (report.frames_total, report.frames_processed) == (8, 8)
    assert report.tracks[0].first_time_ms == pytest.approx(500.0)


def test_process_video_stops_early_and_keeps_partial_results(tmp_path) -> None:
    source = tmp_path / "in.mp4"
    _write_video(source)
    seen: list[int] = []

    report = process_video(
        _recognizer(),
        source,
        tmp_path / "out.mp4",
        on_frame=lambda image, index: seen.append(index),
        should_stop=lambda: len(seen) >= 3,
    )

    assert report.stopped is True
    assert report.frames_processed == 3
    assert len(report.tracks) == 1


def test_process_video_rejects_start_past_end(tmp_path) -> None:
    source = tmp_path / "in.mp4"
    _write_video(source)

    with pytest.raises(ValueError, match="past the end"):
        process_video(_recognizer(), source, tmp_path / "out.mp4", start_seconds=5)


def _fragment(
    track_id: int,
    first: int,
    last: int,
    text: str = "",
    *,
    stable: bool = False,
    votes: int = 1,
    confidence: float = 0.9,
    detection_confidence: float = 0.8,
) -> TrackSummary:
    """A raw track at 10 fps, seen on every frame from ``first`` to ``last``, whose
    reading came from ``votes`` OCR reads."""
    read = {text: votes} if text and votes else {}
    return TrackSummary(
        track_id,
        first,
        last,
        first * 100.0,
        last * 100.0,
        text=text,
        ocr_confidence=confidence,
        stable=stable,
        detection_confidence=detection_confidence,
        frames_seen=last - first + 1,
        track_ids=(track_id,),
        votes=read,
        vote_confidence={text: confidence for text in read},
        crop=np.full((2, 2), track_id, dtype=np.uint8),
    )


def test_merge_joins_a_split_vehicle_and_keeps_its_best_reading() -> None:
    # Pieces of one car from a real run: a read cut off at the frame edge, a misread,
    # and a second, looser box on the same plate while the main track was still alive.
    vehicles = merge_fragments(
        [
            _fragment(1, 0, 1, "4A0781", votes=0),
            _fragment(4, 3, 12, "24A07816", votes=3, detection_confidence=0.7),
            _fragment(9, 14, 17, "22A07816", detection_confidence=0.9),
            _fragment(12, 16, 20, "24A07816"),
        ]
    )

    assert len(vehicles) == 1
    vehicle = vehicles[0]
    assert (vehicle.track_id, vehicle.track_ids) == (1, (1, 4, 9, 12))
    assert (vehicle.first_frame, vehicle.last_frame, vehicle.frames_seen) == (0, 20, 21)
    assert (vehicle.text, vehicle.stable) == ("24A07816", True)
    assert vehicle.votes == {"24A07816": 4, "22A07816": 1}
    # The crop comes from the fragment with the most confident detection.
    assert vehicle.detection_confidence == 0.9 and vehicle.crop[0, 0] == 9


def test_merge_votes_over_all_reads_rather_than_one_locked_fragment() -> None:
    # A real pickup, plate 24C09238: one fragment locked in on a misread with 3 reads,
    # five short fragments read it right once each but were too short to lock in.
    fragments = [_fragment(94, 0, 9, "24G09238", votes=3, stable=True)] + [
        _fragment(track_id, first, first + 1, "24C09238")
        for track_id, first in [(101, 15), (105, 20), (109, 27), (113, 32), (118, 39)]
    ]

    vehicles = merge_fragments(fragments)

    assert [(vehicle.text, vehicle.stable) for vehicle in vehicles] == [("24C09238", True)]


def test_merge_breaks_a_tied_vote_by_confidence_not_by_frames_shown() -> None:
    # One continuous track locked in on 24G09238 early and showed it for 40 frames; later,
    # closer reads said 24C09238 as often and more confidently.
    fragments = [
        _fragment(12, 0, 39, "24G09238", stable=True, votes=3, confidence=0.9),
        _fragment(20, 41, 44, "24C09238", votes=3, confidence=0.98),
    ]

    assert [vehicle.text for vehicle in merge_fragments(fragments)] == ["24C09238"]


def test_merge_keeps_distinct_vehicles_apart() -> None:
    # Readings from a real run: six vehicles, 0.2 s apart, some read as short noise.
    readings = ["3T4073", "51F22029", "605140494", "6", "86", "676103786"]
    fragments = [
        _fragment(index + 1, index * 15, index * 15 + 13, text)
        for index, text in enumerate(readings)
    ]

    assert [vehicle.text for vehicle in merge_fragments(fragments)] == readings


def test_merge_needs_a_short_gap_and_a_reading() -> None:
    far_apart = [_fragment(1, 0, 9, "51G48154"), _fragment(2, 40, 49, "51G48154")]
    unread = [_fragment(1, 0, 9, "51G48154"), _fragment(2, 10, 19)]

    assert len(merge_fragments(far_apart)) == 2
    assert [vehicle.text for vehicle in merge_fragments(unread)] == ["51G48154", ""]


class SwitchingTrackDetector(FakeTrackingDetector):
    """Loses the vehicle halfway and picks it up again under a new id."""

    def track(self, image: np.ndarray) -> list[PlateDetection]:
        self.track_id = 7 if self.calls < 5 else 8
        return super().track(image)


def test_process_video_reports_one_vehicle_for_a_split_track(tmp_path) -> None:
    source = tmp_path / "in.mp4"
    _write_video(source)

    report = process_video(_recognizer(SwitchingTrackDetector()), source, tmp_path / "out.mp4")

    assert [fragment.track_id for fragment in report.fragments] == [7, 8]
    assert len(report.tracks) == 1
    vehicle = report.tracks[0]
    assert (vehicle.track_ids, vehicle.text) == ((7, 8), "51G48154")
    assert (vehicle.first_frame, vehicle.last_frame) == (0, 9)
    # Neither half had the 3 reads to lock in; together they do (OCR on frames 0, 3, 5, 8).
    assert not any(fragment.stable for fragment in report.fragments)
    assert (vehicle.votes, vehicle.stable) == ({"51G48154": 4}, True)
