"""Plate tracking by the distance between predicted and detected box centres.

Ultralytics' ByteTrack matches boxes by overlap multiplied by detection confidence and
confirms a new track only when that product reaches 0.3. On a street video, a motorbike
plate moved about a third of its width per frame (overlap 0.35-0.5) at 0.5-0.7
confidence, so its track was never confirmed and the plate was never returned, except on
repeated frames where it stood still. Measured in plate widths instead, the same plate
moved 0.1-0.8 widths per frame while plates of different vehicles were 5+ widths apart.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np

from .detector import PlateDetection

Box = tuple[int, int, int, int]


class PlateDetector(Protocol):
    def detect(self, image: np.ndarray) -> list[PlateDetection]:
        """Run stateless detection."""

    def crop(
        self, image: np.ndarray, detection: PlateDetection, padding: float = 0.08
    ) -> np.ndarray:
        """Crop one detection."""


@dataclass(slots=True)
class _Track:
    track_id: int
    bbox: Box
    last_step: int
    velocity: tuple[float, float] | None = None


def _centre(bbox: Box) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2


def _width(bbox: Box) -> int:
    return max(1, bbox[2] - bbox[0])


class CentroidTracker:
    """Give plate detections ids that persist while the plate moves.

    Each track is moved forward at its recent velocity and matched to the nearest
    detection whose centre lies within ``max_distance`` plate widths of that prediction
    and whose width is within a factor ``max_size_change`` of the track's. Closest pairs
    are matched first. A track unmatched for more than ``max_age`` updates is dropped;
    a detection left unmatched starts a new track at once.
    """

    def __init__(
        self, max_distance: float = 1.5, max_size_change: float = 2.0, max_age: int = 10
    ) -> None:
        if max_distance <= 0 or max_size_change < 1 or max_age < 1:
            raise ValueError("max_distance > 0, max_size_change >= 1 and max_age >= 1 required")
        self.max_distance = max_distance
        self.max_size_change = max_size_change
        self.max_age = max_age
        self._tracks: list[_Track] = []
        self._next_id = 1
        self._step = 0

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1
        self._step = 0

    def update(self, detections: list[PlateDetection]) -> list[PlateDetection]:
        """Match one frame's detections and return them with ``track_id`` set."""
        self._step += 1
        self._tracks = [
            track for track in self._tracks if self._step - track.last_step <= self.max_age
        ]
        pairs: list[tuple[float, int, int]] = []
        for track_index, track in enumerate(self._tracks):
            predicted = self._predict(track)
            for detection_index, detection in enumerate(detections):
                small, large = sorted((_width(track.bbox), _width(detection.bbox)))
                if large > self.max_size_change * small:
                    continue
                distance = math.dist(predicted, _centre(detection.bbox)) / large
                if distance <= self.max_distance:
                    pairs.append((distance, track_index, detection_index))

        matched: dict[int, int] = {}  # detection index -> track index
        taken: set[int] = set()
        for _, track_index, detection_index in sorted(pairs):
            if detection_index not in matched and track_index not in taken:
                matched[detection_index] = track_index
                taken.add(track_index)

        tracked: list[PlateDetection] = []
        for detection_index, detection in enumerate(detections):
            track_index = matched.get(detection_index)
            if track_index is None:
                track = _Track(self._next_id, detection.bbox, self._step)
                self._next_id += 1
                self._tracks.append(track)
            else:
                track = self._tracks[track_index]
                self._move(track, detection.bbox)
            tracked.append(replace(detection, track_id=track.track_id))
        return tracked

    def _predict(self, track: _Track) -> tuple[float, float]:
        x, y = _centre(track.bbox)
        if track.velocity is None:
            return x, y
        steps = self._step - track.last_step
        return x + track.velocity[0] * steps, y + track.velocity[1] * steps

    def _move(self, track: _Track, bbox: Box) -> None:
        steps = self._step - track.last_step
        (old_x, old_y), (new_x, new_y) = _centre(track.bbox), _centre(bbox)
        velocity = ((new_x - old_x) / steps, (new_y - old_y) / steps)
        if track.velocity is not None:
            # Averaging smooths the stall and double step around a repeated frame.
            velocity = (
                (velocity[0] + track.velocity[0]) / 2,
                (velocity[1] + track.velocity[1]) / 2,
            )
        track.velocity = velocity
        track.bbox = bbox
        track.last_step = self._step


class CentroidTrackingDetector:
    """A plate detector whose ``track`` runs plain detection through a ``CentroidTracker``.

    A drop-in for ``YoloPlateDetector`` in ``RealtimePlateRecognizer``. Every detection gets
    an id on the frame it appears in; ByteTrack returns a new plate only once confirmed.
    """

    def __init__(self, detector: PlateDetector, tracker: CentroidTracker | None = None) -> None:
        self.detector = detector
        self.tracker = tracker or CentroidTracker()

    def detect(self, image: np.ndarray) -> list[PlateDetection]:
        return self.detector.detect(image)

    def track(self, image: np.ndarray) -> list[PlateDetection]:
        return self.tracker.update(self.detector.detect(image))

    def crop(
        self, image: np.ndarray, detection: PlateDetection, padding: float = 0.08
    ) -> np.ndarray:
        return self.detector.crop(image, detection, padding=padding)

    def reset(self) -> None:
        self.tracker.reset()
