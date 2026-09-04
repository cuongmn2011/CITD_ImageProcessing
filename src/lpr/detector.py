"""YOLO adapter with a small, testable detection contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .preprocessing import crop_box


@dataclass(frozen=True, slots=True)
class PlateDetection:
    """One detected license plate in image coordinates."""

    bbox: tuple[int, int, int, int]
    confidence: float
    class_id: int = 0
    track_id: int | None = None


def detections_from_result(result: Any, image_shape: tuple[int, ...]) -> list[PlateDetection]:
    """Convert one Ultralytics result into stable application objects."""
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    height, width = image_shape[:2]
    detections: list[PlateDetection] = []
    for box in boxes:
        coordinates = np.asarray(box.xyxy[0].tolist(), dtype=float)
        if coordinates.size != 4 or not np.isfinite(coordinates).all():
            continue
        x1, y1, x2, y2 = coordinates
        x1, x2 = sorted((max(0, int(x1)), min(width, int(x2))))
        y1, y2 = sorted((max(0, int(y1)), min(height, int(y2))))
        if x2 <= x1 or y2 <= y1:
            continue
        confidence = float(box.conf[0].item()) if getattr(box, "conf", None) is not None else 0.0
        class_id = int(box.cls[0].item()) if getattr(box, "cls", None) is not None else 0
        track_id = None
        if getattr(box, "id", None) is not None:
            track_id = int(box.id[0].item())
        detections.append(PlateDetection((x1, y1, x2, y2), confidence, class_id, track_id))
    return detections


class YoloPlateDetector:
    """Lazy-loading Ultralytics YOLO detector for one plate class."""

    def __init__(
        self,
        model_path: str | Path,
        confidence: float = 0.4,
        iou: float = 0.7,
        device: str | int | None = None,
    ) -> None:
        if not 0 < confidence <= 1:
            raise ValueError("confidence must be in (0, 1]")
        if not 0 < iou <= 1:
            raise ValueError("iou must be in (0, 1]")
        self.model_path = Path(model_path)
        self.confidence = confidence
        self.iou = iou
        self.device = device
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError("Install YOLO support with: uv sync --extra vision") from error
        self._model = YOLO(str(self.model_path))

    def detect(self, image: np.ndarray) -> list[PlateDetection]:
        """Run detection on one BGR/RGB image."""
        if image is None or image.size == 0:
            raise ValueError("Image must be non-empty")
        kwargs: dict[str, Any] = {
            "conf": self.confidence,
            "iou": self.iou,
            "verbose": False,
        }
        if self.device is not None:
            kwargs["device"] = self.device
        result = self._model.predict(source=image, **kwargs)[0]
        return detections_from_result(result, image.shape)

    @staticmethod
    def crop(image: np.ndarray, detection: PlateDetection, padding: float = 0.08) -> np.ndarray:
        """Crop a detected plate for OCR."""
        return crop_box(image, detection.bbox, padding=padding)
