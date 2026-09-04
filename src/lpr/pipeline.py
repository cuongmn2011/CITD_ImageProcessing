"""End-to-end YOLO plate detection and OCR orchestration."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Protocol

import cv2
import numpy as np

from .detector import PlateDetection
from .ocr import OCRBackend, OCRResult, best_result
from .preprocessing import PreprocessVariant, preprocess_plate


class Detector(Protocol):
    def detect(self, image: np.ndarray) -> list[PlateDetection]:
        """Detect plates in one image."""

    @staticmethod
    def crop(image: np.ndarray, detection: PlateDetection, padding: float = 0.08) -> np.ndarray:
        """Crop one detection."""


@dataclass(frozen=True, slots=True)
class PlateRecognition:
    detection: PlateDetection
    ocr: OCRResult | None


class LicensePlateRecognizer:
    """Compose a detector with one or more OCR engines and preprocessing variants."""

    def __init__(
        self,
        detector: Detector,
        backends: Iterable[OCRBackend],
        variants: Iterable[PreprocessVariant] = ("otsu", "clahe"),
        crop_padding: float = 0.08,
    ) -> None:
        self.detector = detector
        self.backends = tuple(backends)
        self.variants = tuple(variants)
        self.crop_padding = crop_padding
        if not self.backends:
            raise ValueError("At least one OCR backend is required")
        if not self.variants:
            raise ValueError("At least one preprocessing variant is required")
        if not 0 <= crop_padding <= 1:
            raise ValueError("crop_padding must be between 0 and 1")

    def recognize_image(self, image: np.ndarray) -> list[PlateRecognition]:
        if image is None or image.size == 0:
            raise ValueError("Image must be non-empty")
        recognitions: list[PlateRecognition] = []
        for detection in self.detector.detect(image):
            crop = self.detector.crop(image, detection, padding=self.crop_padding)
            candidates: list[OCRResult] = []
            for variant in self.variants:
                processed = preprocess_plate(crop, variant)
                for backend in self.backends:
                    result = backend.recognize(processed)
                    candidates.append(replace(result, variant=variant))
            recognitions.append(PlateRecognition(detection, best_result(candidates)))
        return recognitions

    def recognize_video(
        self,
        input_path: str | Path,
        output_path: str | Path,
        max_frames: int | None = None,
    ) -> int:
        """Process a video and write annotated frames; return processed frame count."""
        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {input_path}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            raise ValueError(f"Could not create output video: {output_path}")

        processed_frames = 0
        try:
            while max_frames is None or processed_frames < max_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                annotated = annotate_image(frame, self.recognize_image(frame))
                writer.write(annotated)
                processed_frames += 1
        finally:
            capture.release()
            writer.release()
        return processed_frames


def annotate_image(image: np.ndarray, recognitions: Iterable[PlateRecognition]) -> np.ndarray:
    """Draw plate boxes and recognized text without mutating the input image."""
    if image is None or image.size == 0:
        raise ValueError("Image must be non-empty")
    annotated = image.copy()
    height, width = annotated.shape[:2]
    for recognition in recognitions:
        x1, y1, x2, y2 = recognition.detection.bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 0), 2)
        if recognition.ocr is None or not recognition.ocr.text:
            continue
        label = f"{recognition.ocr.text} ({recognition.ocr.confidence:.2f})"
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        label_top = max(0, y1 - text_height - baseline - 4)
        label_right = min(width, x1 + text_width + 6)
        cv2.rectangle(annotated, (x1, label_top), (label_right, max(y1, text_height + baseline)), (0, 220, 0), -1)
        cv2.putText(annotated, label, (x1 + 3, max(text_height + 1, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return annotated
