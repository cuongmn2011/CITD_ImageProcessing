"""Offline video processing with per-vehicle plate summaries and ground-truth scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import cv2
import numpy as np

from .metrics import edit_distance, evaluate_ocr_pairs
from .ocr import OCRResult, best_result, normalize_text
from .pipeline import annotate_image
from .stream import FrameResult, RealtimePlateRecognizer


@dataclass(slots=True)
class TrackSummary:
    """Everything seen for one tracked vehicle over a video.

    ``track_ids`` lists the tracker ids joined into this vehicle; a raw track has one.
    ``votes`` counts the plate-shaped OCR reads per text, with their best confidence in
    ``vote_confidence``.
    """

    track_id: int
    first_frame: int
    last_frame: int
    first_time_ms: float
    last_time_ms: float
    text: str = ""
    ocr_confidence: float = 0.0
    stable: bool = False
    detection_confidence: float = 0.0
    frames_seen: int = 0
    track_ids: tuple[int, ...] = ()
    votes: dict[str, int] = field(default_factory=dict)
    vote_confidence: dict[str, float] = field(default_factory=dict, repr=False)
    crop: np.ndarray | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "track_ids": list(self.track_ids),
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "first_time_ms": self.first_time_ms,
            "last_time_ms": self.last_time_ms,
            "text": self.text,
            "ocr_confidence": self.ocr_confidence,
            "stable": self.stable,
            "detection_confidence": self.detection_confidence,
            "frames_seen": self.frames_seen,
            "votes": dict(self.votes),
        }


@dataclass(frozen=True, slots=True)
class VideoReport:
    """``tracks`` holds one entry per vehicle; ``fragments`` the raw tracker tracks."""

    frames_total: int
    frames_processed: int
    output_path: Path
    tracks: tuple[TrackSummary, ...]
    stopped: bool = False
    fragments: tuple[TrackSummary, ...] = ()


ProgressCallback = Callable[[int, int, list[TrackSummary]], None]
FrameCallback = Callable[[np.ndarray, int], None]
PlatesCallback = Callable[[int, float, FrameResult], None]


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
                summary = TrackSummary(
                    plate.track_id, frame_id, frame_id, time_ms, time_ms,
                    track_ids=(plate.track_id,),
                )
                self._tracks[plate.track_id] = summary
            summary.last_frame = frame_id
            summary.last_time_ms = time_ms
            summary.frames_seen += 1
            if detection.confidence > summary.detection_confidence:
                summary.detection_confidence = detection.confidence
                summary.crop = crop(frame, detection)
            read = plate.read
            if read is not None and read.valid_plate_format:
                summary.votes[read.text] = summary.votes.get(read.text, 0) + 1
                summary.vote_confidence[read.text] = max(
                    summary.vote_confidence.get(read.text, 0.0), read.confidence
                )
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


def _pooled_votes(members: list[TrackSummary]) -> tuple[dict[str, int], dict[str, float]]:
    votes: dict[str, int] = {}
    confidence: dict[str, float] = {}
    for member in members:
        for text, count in member.votes.items():
            votes[text] = votes.get(text, 0) + count
            confidence[text] = max(confidence.get(text, 0.0), member.vote_confidence[text])
    return votes, confidence


def _best_reading(members: list[TrackSummary], stable_votes: int) -> tuple[str, bool, float]:
    """Pick one reading for a vehicle by a vote over the OCR reads of all its fragments.

    A fragment's own lock-in does not count: a vehicle split into short fragments gets a
    read or two per fragment, too few for any fragment to lock in, and a lone longer
    fragment could otherwise outvote them with a misread. Ties go to the more confident
    reading, then to the one shown for more frames; frames shown come last because a
    track keeps showing whatever it locked in first.
    """
    votes, vote_confidence = _pooled_votes(members)
    support = {text: (count, vote_confidence[text], 0) for text, count in votes.items()}
    for member in members:
        if member.text:
            count, confidence, frames = support.get(member.text, (0, 0.0, 0))
            support[member.text] = (
                count, max(confidence, member.ocr_confidence), frames + member.frames_seen
            )
    if not support:
        return "", False, 0.0
    text = max(support, key=support.__getitem__)
    count, confidence, _ = support[text]
    return text, count >= stable_votes, confidence


def _same_plate(left: str, right: str, max_edits: int) -> bool:
    # Short reads are mostly noise ("6", "86"), so they must match exactly; a plate cut
    # off at the frame edge ("4A0781" for "24A07816") still joins its full reading.
    allowed = min(max_edits, max(len(left), len(right)) // 4)
    return edit_distance(left, right) <= allowed


def merge_fragments(
    fragments: Iterable[TrackSummary],
    *,
    max_gap_ms: float = 2000.0,
    max_edits: int = 2,
    stable_votes: int = 3,
) -> list[TrackSummary]:
    """Join tracks the tracker split off one vehicle, keeping one reading per vehicle.

    A small, fast plate makes the tracker lose a vehicle and restart it under a new id;
    a loose second box on the same plate gets its own id too. A fragment joins an earlier
    vehicle when it starts within ``max_gap_ms`` of that vehicle's last sighting and their
    readings differ by at most ``max_edits`` edits (fewer for short readings). Unread
    fragments carry no evidence and stay separate. Two different vehicles with
    near-identical plates seconds apart would be joined. A vehicle counts as stable once
    its reading has ``stable_votes`` reads.
    """
    vehicles: list[list[TrackSummary]] = []
    recent: list[list[TrackSummary]] = []
    for fragment in sorted(fragments, key=lambda item: (item.first_frame, item.track_id)):
        # Fragments arrive by start time, so a vehicle out of reach now stays out of reach.
        recent = [
            members
            for members in recent
            if fragment.first_time_ms - max(member.last_time_ms for member in members)
            <= max_gap_ms
        ]
        match: list[TrackSummary] | None = None
        if fragment.text:
            best_distance = max_edits + 1
            for members in recent:
                text = _best_reading(members, stable_votes)[0]
                if not text or not _same_plate(fragment.text, text, max_edits):
                    continue
                distance = edit_distance(fragment.text, text)
                if distance < best_distance:
                    match, best_distance = members, distance
        if match is None:
            match = []
            vehicles.append(match)
            recent.append(match)
        match.append(fragment)
    return [_combine(members, stable_votes) for members in vehicles]


def _combine(members: list[TrackSummary], stable_votes: int) -> TrackSummary:
    text, stable, confidence = _best_reading(members, stable_votes)
    best_view = max(members, key=lambda member: member.detection_confidence)
    votes, vote_confidence = _pooled_votes(members)
    return TrackSummary(
        track_id=members[0].track_id,
        first_frame=min(member.first_frame for member in members),
        last_frame=max(member.last_frame for member in members),
        first_time_ms=min(member.first_time_ms for member in members),
        last_time_ms=max(member.last_time_ms for member in members),
        text=text,
        ocr_confidence=confidence,
        stable=stable,
        detection_confidence=best_view.detection_confidence,
        frames_seen=sum(member.frames_seen for member in members),
        track_ids=tuple(track_id for member in members for track_id in member.track_ids),
        votes=votes,
        vote_confidence=vote_confidence,
        crop=best_view.crop,
    )


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
    start_seconds: float = 0.0,
    duration_seconds: float | None = None,
    on_progress: ProgressCallback | None = None,
    on_frame: FrameCallback | None = None,
    on_plates: PlatesCallback | None = None,
    should_stop: Callable[[], bool] | None = None,
    max_output_width: int | None = 1280,
) -> VideoReport:
    """Track and read plates across a video, writing an annotated copy.

    Only every ``stride``-th frame is processed and written, so the output plays at
    ``fps / stride``. Fast vehicles appear in few frames; a large stride can skip them.
    ``start_seconds``/``duration_seconds`` limit the run to one segment; timestamps in
    the report stay relative to the start of the source video. ``on_frame`` receives each
    annotated frame; ``on_plates`` receives each frame's raw boxes (frame index, timestamp,
    the frame's ``FrameResult``) before they are baked into that frame's pixels, for a
    caller that draws its own overlay instead of using the annotated video. ``should_stop``
    ends the run early and still returns what was processed. Tracks split off one vehicle
    are joined by ``merge_fragments``, both in the progress callback and in the report.
    Recognition always runs on full-size frames; the annotated copy is scaled down to
    ``max_output_width`` (``None`` keeps it full size), because encoding 1080p WebM cost
    about as much per frame as plate detection.
    """
    if stride < 1:
        raise ValueError("stride must be at least 1")
    if max_output_width is not None and max_output_width < 2:
        raise ValueError("max_output_width must be at least 2")
    if start_seconds < 0 or (duration_seconds is not None and duration_seconds <= 0):
        raise ValueError("start_seconds must be >= 0 and duration_seconds > 0")
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
    source_frames = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    start_frame = int(round(start_seconds * fps))
    if source_frames and start_frame >= source_frames:
        capture.release()
        raise ValueError("start_seconds is past the end of the video")
    end_frame = (
        start_frame + max(1, int(round(duration_seconds * fps)))
        if duration_seconds is not None
        else None
    )
    if source_frames:
        end_frame = min(end_frame, source_frames) if end_frame is not None else source_frames
    frames_total = end_frame - start_frame if end_frame is not None else 0
    if start_frame:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    output_size = (width, height)
    if max_output_width is not None and width > max_output_width:
        # Codecs want even dimensions.
        scaled_height = round(height * max_output_width / width / 2) * 2
        output_size = (max_output_width // 2 * 2, max(2, scaled_height))
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        writer, destination = _open_writer(destination, fps / stride, output_size)
    except ValueError:
        capture.release()
        raise

    recognizer.reset()
    aggregator = _TrackAggregator()
    stable_votes = recognizer.stable_votes

    def crop(frame: np.ndarray, detection: Any) -> np.ndarray:
        return recognizer.detector.crop(frame, detection, padding=recognizer.crop_padding)

    frame_index = start_frame
    processed = 0
    stopped = False
    try:
        while end_frame is None or frame_index < end_frame:
            if should_stop is not None and should_stop():
                stopped = True
                break
            ok, frame = capture.read()
            if not ok:
                break
            if (frame_index - start_frame) % stride == 0:
                time_ms = frame_index / fps * 1000.0
                result = recognizer.process_frame(frame, frame_index, time_ms)
                aggregator.update(frame, frame_index, time_ms, result, crop)
                if on_plates is not None:
                    on_plates(frame_index, time_ms, result)
                recognitions = [plate.recognition for plate in result.plates]
                annotated = annotate_image(frame, recognitions)
                if output_size != (width, height):
                    annotated = cv2.resize(annotated, output_size, interpolation=cv2.INTER_AREA)
                writer.write(annotated)
                processed += 1
                if on_frame is not None:
                    on_frame(annotated, frame_index)
                if on_progress is not None:
                    on_progress(
                        frame_index + 1 - start_frame,
                        frames_total,
                        merge_fragments(aggregator.summaries(), stable_votes=stable_votes),
                    )
            frame_index += 1
    finally:
        capture.release()
        writer.release()
    fragments = aggregator.summaries()
    return VideoReport(
        frames_total=frames_total or frame_index - start_frame,
        frames_processed=processed,
        output_path=destination,
        tracks=tuple(merge_fragments(fragments, stable_votes=stable_votes)),
        stopped=stopped,
        fragments=tuple(fragments),
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
