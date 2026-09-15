"""Convert plate-crop sources into a merged OCR recognition dataset.

Two source formats are supported: the Roboflow character-annotation export,
which labels each digit/letter on a plate crop as its own YOLO bounding box
(reconstructed into text via ``convert_yolo_char_labels``), and the
PBL4_Deep-Learning GitHub supplement, which encodes ground truth directly in
each crop's filename (``parse_filename_labeled_samples``). Both produce
``OcrSample`` records that ``write_paddlex_rec_dataset`` merges into the
``images/`` + ``train.txt``/``val.txt`` layout PaddleX's recognition trainer
expects (``image_path\\tlabel`` per line). See docs/ocr-dataset-selection.md
for provenance and license notes on each source.
"""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .ocr import normalize_text

DEFAULT_Y_THRESHOLD = 0.08
DEFAULT_VAL_RATIO = 0.1
_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


class OcrDatasetError(RuntimeError):
    """Raised when character labels cannot be converted into an OCR dataset."""


@dataclass(frozen=True, slots=True)
class CharacterBox:
    """One character's normalized YOLO bounding box on a plate crop."""

    char: str
    cx: float
    cy: float
    w: float
    h: float


@dataclass(frozen=True, slots=True)
class OcrSample:
    """A reconstructed (plate-crop, ground-truth-text) pair."""

    image_path: Path
    text: str
    source: str = "unknown"


def load_class_names(data_yaml: Path) -> dict[int, str]:
    """Read the ``names:`` character-class mapping from a YOLO data.yaml."""
    import yaml

    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise OcrDatasetError(f"YOLO data YAML must contain a mapping: {data_yaml}")
    names = payload.get("names")
    if isinstance(names, Mapping):
        return {int(index): str(name) for index, name in names.items()}
    if isinstance(names, list):
        return {index: str(name) for index, name in enumerate(names)}
    raise OcrDatasetError(f"YOLO data YAML has no usable names mapping: {data_yaml}")


def group_lines(
    boxes: list[CharacterBox], *, y_threshold: float = DEFAULT_Y_THRESHOLD
) -> list[list[CharacterBox]]:
    """Cluster character boxes into text lines by vertical (y) position.

    Boxes are sorted by center-y; a new line starts whenever the gap to the
    previous box's center-y exceeds ``y_threshold``. This handles both
    one-line car plates (a single cluster) and two-line motorbike plates.
    """
    if not boxes:
        return []
    ordered = sorted(boxes, key=lambda box: box.cy)
    lines: list[list[CharacterBox]] = [[ordered[0]]]
    for box in ordered[1:]:
        if box.cy - lines[-1][-1].cy > y_threshold:
            lines.append([box])
        else:
            lines[-1].append(box)
    return lines


def boxes_to_text(lines: list[list[CharacterBox]]) -> str:
    """Sort each line left-to-right and join lines top-to-bottom."""
    return "".join(
        "".join(box.char for box in sorted(line, key=lambda box: box.cx)) for line in lines
    )


def _parse_label_file(path: Path, class_names: dict[int, str]) -> list[CharacterBox]:
    boxes: list[CharacterBox] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise OcrDatasetError(f"Malformed YOLO label line in {path}: {raw_line!r}")
        class_id = int(parts[0])
        char = class_names.get(class_id)
        if char is None:
            raise OcrDatasetError(f"Unknown character class id {class_id} in {path}")
        cx, cy, w, h = (float(value) for value in parts[1:])
        boxes.append(CharacterBox(char, cx, cy, w, h))
    return boxes


def convert_yolo_char_labels(
    export_dir: Path,
    class_names: dict[int, str],
    *,
    y_threshold: float = DEFAULT_Y_THRESHOLD,
) -> list[OcrSample]:
    """Reconstruct (crop, ground-truth-text) pairs from a character-box export."""
    export_dir = Path(export_dir)
    samples: list[OcrSample] = []
    for split in ("train", "val", "valid", "test"):
        images_dir = export_dir / split / "images"
        labels_dir = export_dir / split / "labels"
        if not images_dir.is_dir() or not labels_dir.is_dir():
            continue
        for image_path in sorted(images_dir.iterdir()):
            if not image_path.is_file():
                continue
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                continue
            boxes = _parse_label_file(label_path, class_names)
            text = boxes_to_text(group_lines(boxes, y_threshold=y_threshold))
            if text:
                samples.append(OcrSample(image_path, text, source="roboflow-chars"))
    if not samples:
        raise OcrDatasetError(f"No character-labeled samples found under {export_dir}")
    return samples


def parse_filename_labeled_samples(directory: Path) -> list[OcrSample]:
    """Read (crop, ground-truth-text) pairs encoded in image filenames.

    Some plate datasets (e.g. the PBL4_Deep-Learning Vietnamese supplement)
    name each crop after its plate text, for example ``29A87180_1212_0.jpg``.
    The substring before the first underscore is treated as the label.
    """
    directory = Path(directory)
    samples: list[OcrSample] = []
    for image_path in sorted(directory.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in _IMAGE_SUFFIXES:
            continue
        raw_label = image_path.stem.split("_", 1)[0]
        text = normalize_text(raw_label)
        if text:
            samples.append(OcrSample(image_path, text, source="github-filename"))
    if not samples:
        raise OcrDatasetError(f"No filename-labeled samples found under {directory}")
    return samples


def _is_validation_sample(sample: OcrSample, val_ratio: float) -> bool:
    digest = hashlib.sha256(str(sample.image_path).encode("utf-8")).hexdigest()
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
