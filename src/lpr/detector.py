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
        left, top = int(np.floor(x1)), int(np.floor(y1))
        right, bottom = int(np.ceil(x2)), int(np.ceil(y2))
        x1, x2 = max(0, left), min(width, right)
        y1, y2 = max(0, top), min(height, bottom)
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
    """Ultralytics YOLO detector with stateless and persistent tracking modes."""

    def __init__(
        self,
        model_path: str | Path,
        confidence: float = 0.4,
        iou: float = 0.7,
        device: str | int | None = None,
        imgsz: int = 640,
        tracker: str = "bytetrack.yaml",
    ) -> None:
        if not 0 < confidence <= 1:
            raise ValueError("confidence must be in (0, 1]")
        if not 0 < iou <= 1:
            raise ValueError("iou must be in (0, 1]")
        if imgsz <= 0:
            raise ValueError("imgsz must be positive")
        self.model_path = Path(model_path)
        self.confidence = confidence
        self.iou = iou
        self.device = device
        self.imgsz = imgsz
        self.tracker = tracker
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError("Install YOLO support with: uv sync --extra vision") from error
        self._model = YOLO(str(self.model_path))

    def warmup(self, channels: int = 3) -> None:
        """Run one inference so the first live frame avoids model startup cost."""
        if channels not in {1, 3}:
            raise ValueError("channels must be 1 or 3")
        image = np.zeros((self.imgsz, self.imgsz, channels), dtype=np.uint8)
        self.detect(image)

    def _inference_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "conf": self.confidence,
            "iou": self.iou,
            "imgsz": self.imgsz,
            "verbose": False,
        }
        if self.device is not None:
            kwargs["device"] = self.device
        return kwargs

    def detect(self, image: np.ndarray) -> list[PlateDetection]:
        """Run stateless detection on one BGR/RGB image."""
        if image is None or image.size == 0:
            raise ValueError("Image must be non-empty")
        result = self._model.predict(source=image, **self._inference_kwargs())[0]
        return detections_from_result(result, image.shape)

    def track(self, image: np.ndarray) -> list[PlateDetection]:
        """Track plates across frames using Ultralytics' persistent tracker."""
        if image is None or image.size == 0:
            raise ValueError("Image must be non-empty")
        kwargs = self._inference_kwargs()
        kwargs.update({"persist": True, "tracker": self.tracker})
        result = self._model.track(source=image, **kwargs)[0]
        return detections_from_result(result, image.shape)

    @staticmethod
    def crop(image: np.ndarray, detection: PlateDetection, padding: float = 0.08) -> np.ndarray:
        """Crop a detected plate for OCR."""
        return crop_box(image, detection.bbox, padding=padding)
