import cv2
import numpy as np
import pytest

from lpr.detector import PlateDetection
from lpr.ocr import OCRResult
from lpr.stream import RealtimePlateRecognizer
from lpr.video_report import process_video, score_against_ground_truth


class FakeTrackingDetector:
    def __init__(self, track_id: int | None = 7) -> None:
        self.track_id = track_id

    def detect(self, image: np.ndarray) -> list[PlateDetection]:
        return self.track(image)

    def track(self, image: np.ndarray) -> list[PlateDetection]:
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
