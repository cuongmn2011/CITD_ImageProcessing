"""Image preparation utilities for license-plate OCR."""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np

PreprocessVariant = Literal["raw", "gray", "otsu", "adaptive", "clahe"]
PREPROCESS_VARIANTS: tuple[PreprocessVariant, ...] = ("raw", "gray", "otsu", "adaptive", "clahe")


def _as_points(
    points: np.ndarray | list[list[float]] | tuple[tuple[float, float], ...],
) -> np.ndarray:
    array = np.asarray(points, dtype=np.float32)
    if array.shape != (4, 2):
        raise ValueError(f"Expected four 2D points, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("Corner points must be finite")
    if np.unique(array, axis=0).shape[0] != 4:
        raise ValueError("Corner points must be distinct")
    return array


def order_quad_points(
    points: np.ndarray | list[list[float]] | tuple[tuple[float, float], ...],
) -> np.ndarray:
    """Return corners in top-left, top-right, bottom-right, bottom-left order."""
    pts = _as_points(points)
    top_two, bottom_two = (
        pts[np.argsort(pts[:, 1], kind="stable")[:2]],
        pts[np.argsort(pts[:, 1], kind="stable")[2:]],
    )
    top_left, top_right = top_two[np.argsort(top_two[:, 0], kind="stable")]
    bottom_left, bottom_right = bottom_two[np.argsort(bottom_two[:, 0], kind="stable")]
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def rectify_plate(
    image: np.ndarray,
    corners: np.ndarray | list[list[float]] | tuple[tuple[float, float], ...],
) -> np.ndarray:
    """Perspective-correct a quadrilateral plate crop into a flat rectangle."""
    if image is None or image.size == 0:
        raise ValueError("Image must be non-empty")

    top_left, top_right, bottom_right, bottom_left = order_quad_points(corners)
    width = max(
        np.linalg.norm(bottom_right - bottom_left),
        np.linalg.norm(top_right - top_left),
    )
    height = max(
        np.linalg.norm(top_right - bottom_right),
        np.linalg.norm(top_left - bottom_left),
    )
    output_width, output_height = int(round(width)), int(round(height))
    if output_width < 2 or output_height < 2:
        raise ValueError("Quadrilateral must have non-zero width and height")

    destination = np.array(
        [
            [0, 0],
            [output_width - 1, 0],
            [output_width - 1, output_height - 1],
            [0, output_height - 1],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(
        np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32),
        destination,
    )
    return cv2.warpPerspective(image, transform, (output_width, output_height))


def crop_box(
    image: np.ndarray, box: tuple[float, float, float, float], padding: float = 0.08
) -> np.ndarray:
    """Crop and clamp an xyxy box, adding proportional padding around the plate."""
    if image is None or image.size == 0:
        raise ValueError("Image must be non-empty")
    if len(box) != 4 or not np.isfinite(box).all():
        raise ValueError("Box must contain four finite coordinates")
    if padding < 0:
        raise ValueError("Padding cannot be negative")

    x1, y1, x2, y2 = map(float, box)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Box must have positive width and height")
    height, width = image.shape[:2]
    pad_x, pad_y = (x2 - x1) * padding, (y2 - y1) * padding
    left = max(0, int(np.floor(x1 - pad_x)))
    top = max(0, int(np.floor(y1 - pad_y)))
    right = min(width, int(np.ceil(x2 + pad_x)))
    bottom = min(height, int(np.ceil(y2 + pad_y)))
    return image[top:bottom, left:right].copy()


def resize_for_ocr(image: np.ndarray, target_height: int = 64, max_width: int = 512) -> np.ndarray:
    """Upscale/downscale while preserving the plate aspect ratio."""
    if image is None or image.size == 0:
        raise ValueError("Image must be non-empty")
    if target_height < 1 or max_width < 1:
        raise ValueError("Resize limits must be positive")
    height, width = image.shape[:2]
    scale = target_height / height
    output_width = max(1, int(round(width * scale)))
    if output_width > max_width:
        scale = max_width / width
        output_width = max_width
        output_height = max(1, int(round(height * scale)))
    else:
        output_height = target_height
    interpolation = cv2.INTER_CUBIC if scale >= 1 else cv2.INTER_AREA
    return cv2.resize(image, (output_width, output_height), interpolation=interpolation)


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim != 3:
        raise ValueError(f"Expected a 2D or 3D image, got {image.ndim} dimensions")
    channels = image.shape[2]
    if channels == 1:
        return image[:, :, 0]
    if channels == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if channels == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"Unsupported image channel count: {channels}")


def _preprocess_resized(image: np.ndarray, variant: PreprocessVariant) -> np.ndarray:
    if variant == "raw":
        return image
    gray = _to_gray(image)
    if variant == "gray":
        return gray
    if variant == "otsu":
        return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    if variant == "adaptive":
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def preprocess_plate(image: np.ndarray, variant: PreprocessVariant = "otsu") -> np.ndarray:
    """Apply one deterministic OCR preprocessing variant."""
    if image is None or image.size == 0:
        raise ValueError("Image must be non-empty")
    if variant not in PREPROCESS_VARIANTS:
        raise ValueError(f"Unknown preprocessing variant: {variant}")
    return _preprocess_resized(resize_for_ocr(image), variant)


def generate_variants(image: np.ndarray) -> dict[PreprocessVariant, np.ndarray]:
    """Generate all supported variants after one shared resize operation."""
    resized = resize_for_ocr(image)
    return {variant: _preprocess_resized(resized, variant) for variant in PREPROCESS_VARIANTS}
