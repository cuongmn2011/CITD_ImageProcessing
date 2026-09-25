"""Offline video processing with per-vehicle plate summaries and ground-truth scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from .metrics import edit_distance, evaluate_ocr_pairs
from .ocr import OCRResult, best_result, normalize_text
from .pipeline import annotate_image
from .stream import FrameResult, RealtimePlateRecognizer


@dataclass(slots=True)
class TrackSummary:
    """Everything seen for one tracked vehicle over a video."""

    track_id: int
    first_frame: int
    last_frame: int
    first_time_ms: float
    last_time_ms: float
    text: str = ""
    ocr_confidence: float = 0.0
    stable: bool = False
    detection_confidence: float = 0.0
    crop: np.ndarray | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "first_time_ms": self.first_time_ms,
            "last_time_ms": self.last_time_ms,
            "text": self.text,
            "ocr_confidence": self.ocr_confidence,
            "stable": self.stable,
            "detection_confidence": self.detection_confidence,
        }


@dataclass(frozen=True, slots=True)
class VideoReport:
    frames_total: int
    frames_processed: int
    output_path: Path
    tracks: tuple[TrackSummary, ...]


ProgressCallback = Callable[[int, int, list[TrackSummary]], None]


class _TrackAggregator:
    """Keep per-track state after the realtime recognizer has evicted it."""

    def __init__(self) -> None:
        self._tracks: dict[int, TrackSummary] = {}
        self._candidates: dict[int, dict[str, OCRResult]] = {}

    def update(
        self,
        frame: np.ndarray,
        frame_id: int,
        time_ms: float,
        result: FrameResult,
        crop: Callable[[np.ndarray, Any], np.ndarray],
    ) -> None:
        for plate in result.plates:
            # Untracked boxes get a fresh key every frame, so they cannot be followed.
            if plate.track_id is None:
                continue
            detection = plate.recognition.detection
            summary = self._tracks.get(plate.track_id)
            if summary is None:
                summary = TrackSummary(plate.track_id, frame_id, frame_id, time_ms, time_ms)
                self._tracks[plate.track_id] = summary
            summary.last_frame = frame_id
            summary.last_time_ms = time_ms
            if detection.confidence > summary.detection_confidence:
                summary.detection_confidence = detection.confidence
                summary.crop = crop(frame, detection)
            ocr = plate.recognition.ocr
            if ocr is None or not ocr.text:
                continue
            if plate.stable:
                summary.stable = True
                summary.text = ocr.text
                summary.ocr_confidence = ocr.confidence
                continue
            candidates = self._candidates.setdefault(plate.track_id, {})
            previous = candidates.get(ocr.text)
            if previous is None or ocr.confidence > previous.confidence:
                candidates[ocr.text] = ocr
            if not summary.stable:
                chosen = best_result(list(candidates.values())) or max(
                    candidates.values(), key=lambda candidate: candidate.confidence
                )
                summary.text = chosen.text
                summary.ocr_confidence = chosen.confidence

    def summaries(self) -> list[TrackSummary]:
        return sorted(self._tracks.values(), key=lambda summary: summary.first_frame)


def _open_writer(path: Path, fps: float, size: tuple[int, int]) -> tuple[cv2.VideoWriter, Path]:
    """Prefer browser-playable VP8 WebM; fall back to mp4v MP4 when it is unavailable."""
    attempts = [(path, "VP80")] if path.suffix.lower() == ".webm" else []
    attempts.append((path.with_suffix(".mp4"), "mp4v"))
    for candidate, fourcc in attempts:
        writer = cv2.VideoWriter(str(candidate), cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if writer.isOpened():
            return writer, candidate
        writer.release()
    raise ValueError(f"Could not create output video: {path}")


def process_video(
    recognizer: RealtimePlateRecognizer,
    input_path: str | Path,
    output_path: str | Path,
    *,
    stride: int = 1,
    on_progress: ProgressCallback | None = None,
) -> VideoReport:
    """Track and read plates across a video, writing an annotated copy.

    Only every ``stride``-th frame is processed and written, so the output plays at
    ``fps / stride``. Fast vehicles appear in few frames; a large stride can skip them.
    """
    if stride < 1:
        raise ValueError("stride must be at least 1")
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Could not open video: {input_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise ValueError(f"Video has invalid dimensions: {input_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0
    frames_total = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        writer, destination = _open_writer(destination, fps / stride, (width, height))
    except ValueError:
        capture.release()
        raise

    recognizer.reset()
    aggregator = _TrackAggregator()

    def crop(frame: np.ndarray, detection: Any) -> np.ndarray:
        return recognizer.detector.crop(frame, detection, padding=recognizer.crop_padding)

    frame_index = 0
    processed = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % stride == 0:
                time_ms = frame_index / fps * 1000.0
                result = recognizer.process_frame(frame, frame_index, time_ms)
                aggregator.update(frame, frame_index, time_ms, result, crop)
                recognitions = [plate.recognition for plate in result.plates]
                writer.write(annotate_image(frame, recognitions))
                processed += 1
                if on_progress is not None:
                    on_progress(frame_index + 1, frames_total, aggregator.summaries())
            frame_index += 1
    finally:
        capture.release()
        writer.release()
    return VideoReport(
        frames_total=frames_total or frame_index,
        frames_processed=processed,
        output_path=destination,
        tracks=tuple(aggregator.summaries()),
    )


def score_against_ground_truth(
    predictions: list[str], truths: list[str]
) -> dict[str, Any]:
    """Match predicted plates one-to-one with true plates and score them.

    Identical predictions are counted once, since one vehicle split into two tracks
    is not a second read. A truth is only paired with a prediction within
    ``len(truth) // 2`` edits; beyond that it counts as missed rather than misread.
    """
    truth_texts = [text for text in (normalize_text(value) for value in truths) if text]
    if not truth_texts:
        raise ValueError("Ground truth must contain at least one plate")
    predicted = list(dict.fromkeys(t for t in (normalize_text(v) for v in predictions) if t))

    pairs = sorted(
        (edit_distance(truth, prediction), truth_index, prediction_index)
        for truth_index, truth in enumerate(truth_texts)
        for prediction_index, prediction in enumerate(predicted)
    )
    matched_truth: dict[int, int] = {}
    used_predictions: set[int] = set()
    for distance, truth_index, prediction_index in pairs:
        if truth_index in matched_truth or prediction_index in used_predictions:
            continue
        if distance > len(truth_texts[truth_index]) // 2:
            continue
        matched_truth[truth_index] = prediction_index
        used_predictions.add(prediction_index)

    details = []
    for truth_index, truth in enumerate(truth_texts):
        prediction_index = matched_truth.get(truth_index)
        prediction = predicted[prediction_index] if prediction_index is not None else ""
        if not prediction:
            status = "missed"
        elif prediction == truth:
            status = "correct"
        else:
            status = "wrong"
        details.append({"ground_truth": truth, "prediction": prediction, "status": status})

    metrics = evaluate_ocr_pairs(details)
    return {
        **metrics,
        "correct": sum(detail["status"] == "correct" for detail in details),
        "wrong": sum(detail["status"] == "wrong" for detail in details),
        "missed": sum(detail["status"] == "missed" for detail in details),
        "details": details,
        "extra": [text for index, text in enumerate(predicted) if index not in used_predictions],
    }
