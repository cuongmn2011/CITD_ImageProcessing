import numpy as np
import pytest

from lpr.preprocessing import (
    crop_box,
    generate_variants,
    order_quad_points,
    preprocess_plate,
    rectify_plate,
    resize_for_ocr,
)


def test_order_quad_points_returns_clockwise_rectangle() -> None:
    points = [[90, 80], [10, 10], [100, 20], [0, 70]]
    assert np.array_equal(
        order_quad_points(points),
        np.array([[10, 10], [100, 20], [90, 80], [0, 70]], dtype=np.float32),
    )


def test_order_quad_points_rejects_duplicate_points() -> None:
    with pytest.raises(ValueError, match="distinct"):
        order_quad_points([[0, 0], [0, 0], [1, 1], [1, 0]])


def test_rectify_plate_produces_expected_dimensions() -> None:
    image = np.zeros((100, 180, 3), dtype=np.uint8)
    corners = [[20, 20], [140, 10], [150, 50], [10, 60]]
    rectified = rectify_plate(image, corners)
    assert rectified.shape[:2] == (41, 140)


def test_crop_box_clamps_and_copies() -> None:
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    crop = crop_box(image, (-5, 2, 10, 10), padding=0)
    assert crop.shape[:2] == (8, 10)
    crop[:] = 255
    assert image[2, 0].tolist() == [0, 0, 0]


def test_resize_and_variants_keep_non_empty_output() -> None:
    image = np.full((10, 30, 3), 120, dtype=np.uint8)
    resized = resize_for_ocr(image, target_height=64)
    assert resized.shape[:2] == (64, 192)
    variants = generate_variants(image)
    assert set(variants) == {"raw", "gray", "otsu", "adaptive", "clahe"}
    assert all(value.size > 0 for value in variants.values())
    assert preprocess_plate(image, "gray").ndim == 2


def test_single_channel_images_are_supported() -> None:
    image = np.full((10, 30, 1), 120, dtype=np.uint8)
    assert preprocess_plate(image, "gray").shape == (64, 192)


def test_order_quad_points_handles_tied_coordinate_scores() -> None:
    points = [[5, 0], [10, 5], [5, 10], [0, 5]]
    assert np.array_equal(
        order_quad_points(points), np.array([[5, 0], [10, 5], [5, 10], [0, 5]], dtype=np.float32)
    )


def test_order_quad_points_handles_strong_skew() -> None:
    points = [[100, 100], [200, 100], [200, 200], [0, 150]]
    assert np.array_equal(
        order_quad_points(points),
        np.array([[100, 100], [200, 100], [200, 200], [0, 150]], dtype=np.float32),
    )
