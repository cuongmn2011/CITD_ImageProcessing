"""Realtime frame inference with tracking, OCR gating, and temporal voting."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Iterable, Literal, Protocol

import numpy as np

from .detector import PlateDetection
from .ocr import OCRBackend, OCRResult, best_result
from .pipeline import PlateRecognition
from .preprocessing import PREPROCESS_VARIANTS, PreprocessVariant, generate_variants


class StreamingDetector(Protocol):
    def detect(self, image: np.ndarray) -> list[PlateDetection]:
        """Run stateless detection."""

    @staticmethod
    def crop(image: np.ndarray, detection: PlateDetection, padding: float = 0.08) -> np.ndarray:
        """Crop one detection."""


@dataclass(frozen=True, slots=True)
class StreamPlate:
    """One plate result enriched with temporal state for a live UI.

    ``recognition.ocr`` is the reading to show, often cached from earlier frames; ``read``
    is the OCR result computed on this frame, or ``None`` when OCR did not run.
    """

    recognition: PlateRecognition
    track_id: int | None
    stable: bool
    status: Literal["stable", "candidate", "detected"]
    read: OCRResult | None = None


@dataclass(frozen=True, slots=True)
class FrameResult:
    """Result emitted immediately after processing one frame."""

    frame_id: int
    source_time_ms: float | None
    plates: tuple[StreamPlate, ...]
    latency_ms: float


@dataclass(slots=True)
class _TrackState:
    last_seen_frame: int
    last_ocr_frame: int = -1
    stable_text: str = ""
    stable_result: OCRResult | None = None
    vote_history: deque[str] = field(default_factory=deque)
    latest_by_text: dict[str, OCRResult] = field(default_factory=dict)


class RealtimePlateRecognizer:
    """Process frames without repeating OCR for already-stable tracks."""

    def __init__(
        self,
        detector: StreamingDetector,
        backends: Iterable[OCRBackend],
        variants: Iterable[PreprocessVariant] = ("otsu",),
        crop_padding: float = 0.08,
        ocr_refresh_frames: int = 12,
        ocr_retry_frames: int = 3,
        stable_votes: int = 3,
        track_ttl_frames: int = 30,
    ) -> None:
        self.detector = detector
        self.backends = tuple(backends)
        self.variants = tuple(variants)
        self.crop_padding = crop_padding
        self.ocr_refresh_frames = ocr_refresh_frames
        self.ocr_retry_frames = min(ocr_retry_frames, ocr_refresh_frames)
        self.stable_votes = stable_votes
        self.track_ttl_frames = track_ttl_frames
        self._tracks: dict[int, _TrackState] = {}
        if not self.backends:
            raise ValueError("At least one OCR backend is required")
        if not self.variants:
            raise ValueError("At least one preprocessing variant is required")
        invalid = set(self.variants) - set(PREPROCESS_VARIANTS)
        if invalid:
            raise ValueError(f"Unknown preprocessing variants: {sorted(invalid)}")
        if not 0 <= crop_padding <= 1:
            raise ValueError("crop_padding must be between 0 and 1")
        if (
            ocr_refresh_frames < 1
            or ocr_retry_frames < 1
            or stable_votes < 1
            or track_ttl_frames < 1
        ):
            raise ValueError("Realtime timing values must be positive")
    def reset(self) -> None:
        """Forget tracker-side OCR state at the beginning of a new session."""
        self._tracks.clear()

    def process_frame(
        self,
        image: np.ndarray,
        frame_id: int,
        source_time_ms: float | None = None,
    ) -> FrameResult:
        """Process one BGR/RGB frame and return a structured live result."""
        if image is None or image.size == 0:
            raise ValueError("Image must be non-empty")
        if frame_id < 0:
            raise ValueError("frame_id must be non-negative")
        started = perf_counter()
        track_method = getattr(self.detector, "track", None)
        detections = (
            track_method(image) if callable(track_method) else self.detector.detect(image)
        )
        plates: list[StreamPlate] = []
        active_keys: set[int] = set()
        for fallback_key, detection in enumerate(detections):
            key = detection.track_id if detection.track_id is not None else -(fallback_key + 1)
            active_keys.add(key)
            state = self._tracks.setdefault(key, _TrackState(last_seen_frame=frame_id))
            state.last_seen_frame = frame_id
            recognition, read = self._recognize_detection(image, detection, state, frame_id)
            stable = bool(state.stable_text)
            status = "stable" if stable else ("candidate" if recognition.ocr else "detected")
            plates.append(StreamPlate(recognition, detection.track_id, stable, status, read))
        self._evict_stale(frame_id, active_keys)
        return FrameResult(
            frame_id=frame_id,
            source_time_ms=source_time_ms,
            plates=tuple(plates),
            latency_ms=(perf_counter() - started) * 1000.0,
        )

    def _recognize_detection(
        self,
        image: np.ndarray,
        detection: PlateDetection,
        state: _TrackState,
        frame_id: int,
    ) -> tuple[PlateRecognition, OCRResult | None]:
        """Return the reading to show and, when OCR ran on this frame, its fresh result."""
        has_stable_text = bool(state.stable_text)
        refresh_after = self.ocr_refresh_frames if has_stable_text else self.ocr_retry_frames
        should_refresh = (
            state.last_ocr_frame < 0
            or frame_id - state.last_ocr_frame >= refresh_after
        )
        read: OCRResult | None = None
        if should_refresh:
            crop = self.detector.crop(image, detection, padding=self.crop_padding)
            read = self._recognize_crop(crop)
            state.last_ocr_frame = frame_id
            if read is not None:
                if read.valid_plate_format:
                    state.vote_history.append(read.text)
                    state.latest_by_text[read.text] = read
                    while len(state.vote_history) > self.stable_votes * 2:
                        state.vote_history.popleft()
                    winner, votes = Counter(state.vote_history).most_common(1)[0]
                    if votes >= self.stable_votes:
                        state.stable_text = winner
                        # The winning reading, not this read when it lost the vote.
                        state.stable_result = state.latest_by_text[winner]
                    elif not has_stable_text:
                        state.stable_result = read
                elif not has_stable_text:
                    state.stable_result = read
        return PlateRecognition(detection, state.stable_result), read

    def _recognize_crop(self, crop: np.ndarray) -> OCRResult | None:
        variants = generate_variants(crop, self.variants)
        candidates: list[OCRResult] = []
        for variant in self.variants:
            for backend in self.backends:
                result = backend.recognize(variants[variant])
                candidates.append(replace(result, variant=variant))
        selected = best_result(candidates)
        if selected is not None:
            return selected
        return max(candidates, key=lambda candidate: candidate.confidence, default=None)

    def _evict_stale(self, frame_id: int, active_keys: set[int]) -> None:
        stale = [
            key
            for key, state in self._tracks.items()
            if key not in active_keys and frame_id - state.last_seen_frame > self.track_ttl_frames
        ]
        for key in stale:
            del self._tracks[key]
