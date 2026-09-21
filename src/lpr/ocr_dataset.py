"""Build a PaddleX OCR recognition dataset from filename-labeled plate crops.

The PBL4_Deep-Learning GitHub dataset (see docs/ocr-dataset-selection.md) names
each plate-crop image after its ground-truth text, for example
``29A87180_1212_0.jpg``. ``parse_filename_labeled_samples`` extracts those
pairs, and ``write_paddlex_rec_dataset`` writes them into the
``images/`` + ``train.txt``/``val.txt`` layout PaddleX's recognition trainer
expects (``image_path\\tlabel`` per line).
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

from .ocr import normalize_text

DEFAULT_VAL_RATIO = 0.1
_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


class OcrDatasetError(RuntimeError):
    """Raised when plate-crop samples cannot be converted into an OCR dataset."""


@dataclass(frozen=True, slots=True)
class OcrSample:
    """A (plate-crop, ground-truth-text) pair."""

    image_path: Path
    text: str


def parse_filename_labeled_samples(directory: Path) -> list[OcrSample]:
    """Read (crop, ground-truth-text) pairs encoded in image filenames.

    The substring before the first underscore in each filename is treated as
    the plate text, for example ``29A87180_1212_0.jpg`` -> ``29A87180``.
    """
    directory = Path(directory)
    samples: list[OcrSample] = []
    for image_path in sorted(directory.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in _IMAGE_SUFFIXES:
            continue
        raw_label = image_path.stem.split("_", 1)[0]
        text = normalize_text(raw_label)
        if text:
            samples.append(OcrSample(image_path, text))
    if not samples:
        raise OcrDatasetError(f"No filename-labeled samples found under {directory}")
    return samples


def _is_validation_sample(sample: OcrSample, val_ratio: float) -> bool:
    # Hash by plate text, not image path: several crops share the same plate
    # (e.g. ``29A87180_1212_0.jpg``, ``29A87180_1212_1.jpg``), and splitting by
    # path would leak near-duplicate crops of the same plate across train/val.
    digest = hashlib.sha256(sample.text.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return bucket < val_ratio


def write_paddlex_rec_dataset(
    samples: list[OcrSample],
    out_dir: Path,
    *,
    val_ratio: float = DEFAULT_VAL_RATIO,
) -> Path:
    """Write a PaddleX text-recognition dataset: images/ + train.txt/val.txt."""
    if not samples:
        raise OcrDatasetError("Cannot write an OCR dataset with no samples")
    out_dir = Path(out_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    train_lines: list[str] = []
    val_lines: list[str] = []
    used_names: set[str] = set()
    for sample in samples:
        digest = hashlib.sha256(str(sample.image_path).encode("utf-8")).hexdigest()[:12]
        name = f"{digest}_{sample.image_path.name}"
        if name in used_names:
            raise OcrDatasetError(f"Duplicate output image name: {name}")
        used_names.add(name)
        shutil.copyfile(sample.image_path, images_dir / name)
        line = f"images/{name}\t{sample.text}"
        if _is_validation_sample(sample, val_ratio):
            val_lines.append(line)
        else:
            train_lines.append(line)

    if not train_lines:
        raise OcrDatasetError("Split produced an empty training set; lower val_ratio")
    if not val_lines:
        raise OcrDatasetError("Split produced an empty validation set; raise val_ratio")

    (out_dir / "train.txt").write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    (out_dir / "val.txt").write_text("\n".join(val_lines) + "\n", encoding="utf-8")
    return out_dir
