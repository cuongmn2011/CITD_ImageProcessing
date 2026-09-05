"""End-to-end YOLO plate detection and OCR orchestration."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Protocol

import cv2
import numpy as np

from .detector import PlateDetection
from .ocr import OCRBackend, OCRResult, best_result
from .preprocessing import PREPROCESS_VARIANTS, PreprocessVariant, generate_variants


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
        invalid_variants = set(self.variants) - set(PREPROCESS_VARIANTS)
        if invalid_variants:
            raise ValueError(f"Unknown preprocessing variants: {sorted(invalid_variants)}")
        if not 0 <= crop_padding <= 1:
            raise ValueError("crop_padding must be between 0 and 1")

    def recognize_image(self, image: np.ndarray) -> list[PlateRecognition]:
        if image is None or image.size == 0:
            raise ValueError("Image must be non-empty")
        recognitions: list[PlateRecognition] = []
        for detection in self.detector.detect(image):
            crop = self.detector.crop(image, detection, padding=self.crop_padding)
            variants = generate_variants(crop)
            candidates: list[OCRResult] = []
            for variant in self.variants:
                for backend in self.backends:
                    result = backend.recognize(variants[variant])
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
        source = Path(input_path).resolve()
        destination = Path(output_path).resolve()
        if source == destination:
            raise ValueError("Input and output video paths must be different")
        try:
            if os.path.exists(destination) and os.path.samefile(source, destination):
                raise ValueError("Input and output video paths must be different")
        except FileNotFoundError:
            pass
        if max_frames is not None and max_frames <= 0:
            raise ValueError("max_frames must be positive when provided")

        destination.parent.mkdir(parents=True, exist_ok=True)
        capture = cv2.VideoCapture(str(source))
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
        writer = cv2.VideoWriter(
            str(destination),
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
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        )
        label_top = max(0, y1 - text_height - baseline - 4)
        label_right = min(width, x1 + text_width + 6)
        cv2.rectangle(
            annotated,
            (x1, label_top),
            (label_right, max(y1, text_height + baseline)),
            (0, 220, 0),
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 3, max(text_height + 1, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return annotated
