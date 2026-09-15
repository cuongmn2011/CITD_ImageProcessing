"""Realtime frame inference with tracking, OCR gating, and temporal voting."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Iterable, Protocol

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
    """One plate result enriched with temporal state for a live UI."""

    recognition: PlateRecognition
    track_id: int | None
    stable: bool
    status: str


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
    votes: Counter[str] | None = None
    observations: int = 0

    def __post_init__(self) -> None:
        self.votes = Counter()


class RealtimePlateRecognizer:
    """Process frames without repeating OCR for already-stable tracks."""

    def __init__(
        self,
        detector: StreamingDetector,
        backends: Iterable[OCRBackend],
        variants: Iterable[PreprocessVariant] = ("otsu",),
        crop_padding: float = 0.08,
        ocr_refresh_frames: int = 12,
        stable_votes: int = 3,
        track_ttl_frames: int = 30,
    ) -> None:
        self.detector = detector
        self.backends = tuple(backends)
        self.variants = tuple(variants)
        self.crop_padding = crop_padding
        self.ocr_refresh_frames = ocr_refresh_frames
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
        if ocr_refresh_frames < 1 or stable_votes < 1 or track_ttl_frames < 1:
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
            recognition = self._recognize_detection(image, detection, state, frame_id)
            stable = bool(state.stable_text)
            status = "stable" if stable else ("candidate" if recognition.ocr else "detected")
            plates.append(StreamPlate(recognition, detection.track_id, stable, status))
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
    ) -> PlateRecognition:
        cached = state.stable_result
        should_refresh = (
            cached is None
            or state.last_ocr_frame < 0
            or frame_id - state.last_ocr_frame >= self.ocr_refresh_frames
        )
        if should_refresh:
            crop = self.detector.crop(image, detection, padding=self.crop_padding)
            result = self._recognize_crop(crop)
            state.last_ocr_frame = frame_id
            if result is not None:
                state.observations += 1
                if result.valid_plate_format:
                    assert state.votes is not None
                    state.votes[result.text] += 1
                    winner, votes = state.votes.most_common(1)[0]
                    if votes >= self.stable_votes:
                        state.stable_text = winner
                        state.stable_result = result
                elif state.stable_result is None:
                    state.stable_result = result
        if state.stable_result is not None:
            result = state.stable_result
        else:
            result = OCRResult("", 0.0, "none")
        return PlateRecognition(detection, result)

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
