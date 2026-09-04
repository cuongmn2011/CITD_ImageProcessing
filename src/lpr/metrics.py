"""Metrics for OCR and end-to-end plate recognition experiments."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from .ocr import normalize_text


def edit_distance(left: str, right: str) -> int:
    """Compute Levenshtein distance with O(min(len(left), len(right))) memory."""
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            current.append(min(
                current[-1] + 1,
                previous[right_index] + 1,
                previous[right_index - 1] + (left_char != right_char),
            ))
        previous = current
    return previous[-1]


def evaluate_ocr_pairs(rows: Iterable[Mapping[str, str]]) -> dict[str, float | int]:
    """Return exact accuracy, character accuracy, CER, and sample count."""
    exact = 0
    errors = 0
    total_characters = 0
    count = 0
    for row in rows:
        ground_truth = normalize_text(str(row.get("ground_truth", "")))
        prediction = normalize_text(str(row.get("prediction", "")))
        count += 1
        exact += prediction == ground_truth
        errors += edit_distance(ground_truth, prediction)
        total_characters += len(ground_truth)
    return {
        "samples": count,
        "exact_accuracy": exact / count if count else 0.0,
        "character_accuracy": max(0.0, 1.0 - errors / total_characters) if total_characters else 0.0,
        "cer": errors / total_characters if total_characters else 0.0,
    }
