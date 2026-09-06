"""Lazy Roboflow dataset download and YOLO export validation."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_DATASET_SPEC = "cuong-ta-ulxex/vietnamese-car-license-plate/1"
DEFAULT_DATASET_CACHE_ROOT = Path("data/processed")
DEFAULT_DATASET_LOCATION = DEFAULT_DATASET_CACHE_ROOT / "license-plates"
DEFAULT_DATASET_FORMAT = "yolov8"
MANIFEST_FILENAME = ".dataset-manifest.json"


class DatasetPreparationError(RuntimeError):
    """Raised when a dataset cannot be downloaded or validated."""


@dataclass(frozen=True, slots=True)
class RoboflowDatasetSpec:
    """A public Roboflow Universe dataset version."""

    workspace: str
    project: str
    version: int

    @classmethod
    def parse(cls, value: str) -> "RoboflowDatasetSpec":
        parts = value.strip("/").split("/")
        if len(parts) != 3 or not all(parts):
            raise ValueError("Dataset must use workspace/project/version format")
        try:
            version = int(parts[2])
        except ValueError as error:
            raise ValueError("Dataset version must be a positive integer") from error
        if version <= 0:
            raise ValueError("Dataset version must be a positive integer")
        return cls(parts[0], parts[1], version)

    def __str__(self) -> str:
        return f"{self.workspace}/{self.project}/{self.version}"


@dataclass(frozen=True, slots=True)
class PreparedDataset:
    """Validated local paths for a downloaded dataset."""

    location: Path
    data_yaml: Path
    manifest: Path
    spec: RoboflowDatasetSpec
    model_format: str


def _find_data_yaml(location: Path) -> Path:
    candidates = [location / "data.yaml", location / "data.yml"]
    candidates.extend(sorted(location.glob("*/data.yaml")))
    candidates.extend(sorted(location.glob("*/data.yml")))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise DatasetPreparationError(f"Roboflow export has no data.yaml under {location}")


def _image_files(directory: Path) -> list[Path]:
    suffixes = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
    return [
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in suffixes
    ]


def validate_yolo_export(location: str | Path) -> Path:
    """Validate the directory shape emitted by a Roboflow YOLO export."""
    root = Path(location).expanduser().resolve()
    if not root.is_dir():
        raise DatasetPreparationError(f"Dataset location does not exist: {root}")

    data_yaml = _find_data_yaml(root)
    try:
        import yaml

        payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    except ImportError as error:
        raise DatasetPreparationError(
            "Install dataset support with: uv sync --extra dataset"
        ) from error
    except yaml.YAMLError as error:
        raise DatasetPreparationError(f"Invalid YOLO data YAML: {data_yaml}") from error
    if not isinstance(payload, Mapping):
        raise DatasetPreparationError(f"YOLO data YAML must contain a mapping: {data_yaml}")

    train_entry = payload.get("train")
    validation_entry = payload.get("val", payload.get("valid"))
    if not isinstance(train_entry, str) or not isinstance(validation_entry, str):
        raise DatasetPreparationError("YOLO data YAML must define train and val paths")

    export_root = root
    if (
        not (export_root / "train" / "images").is_dir()
        and (data_yaml.parent / "train" / "images").is_dir()
    ):
        export_root = data_yaml.parent
    yaml_root = Path(str(payload.get("path", ".")))
    if not yaml_root.is_absolute():
        yaml_root = (data_yaml.parent / yaml_root).resolve()
    train_images = (yaml_root / train_entry).resolve()
    validation_images = (yaml_root / validation_entry).resolve()
    expected_train = (export_root / "train" / "images").resolve()
    validation_split = next(
        (
            split
            for split in ("val", "valid")
            if (export_root / split / "images").is_dir()
        ),
        None,
    )
    expected_validation = (
        (export_root / validation_split / "images").resolve()
        if validation_split is not None
        else None
    )
    train_matches = train_images == expected_train
    validation_matches = validation_images == expected_validation
    paths_need_normalization = (
        payload.get("path") != str(export_root)
        or payload.get("train") != "train/images"
        or payload.get("val") != f"{validation_split}/images"
        or "valid" in payload
    )
    if not train_matches or not validation_matches:
        train_tail_matches = Path(train_entry).parts[-2:] == ("train", "images")
        validation_tail_matches = Path(validation_entry).parts[-2:] in {
            ("val", "images"),
            ("valid", "images"),
        }
        if (
            expected_validation is None
            or not train_tail_matches
            or not validation_tail_matches
        ):
            raise DatasetPreparationError(
                "YOLO data YAML paths do not match the exported train/validation directories"
            )
    if paths_need_normalization:
        if expected_validation is None:
            raise DatasetPreparationError("YOLO export has no validation images directory")
        payload["path"] = str(export_root)
        payload["train"] = "train/images"
        payload["val"] = f"{validation_split}/images"
        payload.pop("valid", None)
        data_yaml.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        train_images = expected_train
        validation_images = expected_validation
    if expected_validation is None:
        raise DatasetPreparationError("YOLO export has no validation images directory")

    train_labels = expected_train.parent / "labels"
    validation_labels = expected_validation.parent / "labels"
    if not train_labels.is_dir() or not validation_labels.is_dir():
        raise DatasetPreparationError("YOLO export is missing train or validation labels")
    if not _image_files(expected_train) or not list(train_labels.glob("*.txt")):
        raise DatasetPreparationError("YOLO export train split is empty")
    if not _image_files(expected_validation) or not list(validation_labels.glob("*.txt")):
        raise DatasetPreparationError("YOLO export validation split is empty")
    return data_yaml


def _read_manifest(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return dict(payload) if isinstance(payload, Mapping) else None


def _cached_dataset(
    location: Path,
    spec: RoboflowDatasetSpec,
    model_format: str,
) -> PreparedDataset | None:
    manifest_path = location / MANIFEST_FILENAME
    manifest = _read_manifest(manifest_path)
    if manifest is None:
        return None
    if manifest.get("dataset") != str(spec) or manifest.get("format") != model_format:
        return None
    raw_data_yaml = manifest.get("data_yaml")
    if not isinstance(raw_data_yaml, str) or Path(raw_data_yaml).is_absolute():
        return None
    data_yaml = (location / raw_data_yaml).resolve()
    try:
        data_yaml.relative_to(location)
    except ValueError:
        return None
    if not data_yaml.is_file():
        return None
    validated_data_yaml = validate_yolo_export(location).resolve()
    if validated_data_yaml != data_yaml:
        return None
    return PreparedDataset(location, validated_data_yaml, manifest_path, spec, model_format)


def _validate_destination(destination: Path, *, force: bool) -> None:
    current_directory = Path.cwd().resolve()
    cache_root = (current_directory / DEFAULT_DATASET_CACHE_ROOT).resolve()
    if destination in {Path(destination.anchor), current_directory, cache_root}:
        raise DatasetPreparationError("Dataset location must be a dedicated cache child directory")
    try:
        current_directory.relative_to(destination)
    except ValueError:
        pass
    else:
        raise DatasetPreparationError(
            "Dataset location cannot be the project directory or its parent"
        )
    if force:
        try:
            destination.relative_to(cache_root)
        except ValueError as error:
            raise DatasetPreparationError(
                f"Forced replacement is limited to a child of {cache_root}"
            ) from error


def _download_with_roboflow(
    spec: RoboflowDatasetSpec,
    location: Path,
    api_key: str,
    model_format: str,
) -> Path:
    try:
        import roboflow
    except ImportError as error:
        raise DatasetPreparationError(
            "Install dataset support with: uv sync --extra dataset"
        ) from error

    try:
        rf = roboflow.Roboflow(api_key=api_key)
        project = rf.workspace(spec.workspace).project(spec.project)
        dataset = project.version(spec.version).download(
            model_format=model_format,
            location=str(location),
            overwrite=True,
        )
    except Exception as error:
        raise DatasetPreparationError(f"Roboflow download failed for {spec}: {error}") from error

    downloaded_location = Path(getattr(dataset, "location", location)).expanduser().resolve()
    if not downloaded_location.is_dir():
        raise DatasetPreparationError(
            f"Roboflow SDK returned an invalid dataset location: {downloaded_location}"
        )
    return downloaded_location


def ensure_roboflow_dataset(
    spec: str | RoboflowDatasetSpec = DEFAULT_DATASET_SPEC,
    location: str | Path = DEFAULT_DATASET_LOCATION,
    *,
    api_key: str | None = None,
    model_format: str = DEFAULT_DATASET_FORMAT,
    force: bool = False,
) -> PreparedDataset:
    """Return a cached dataset, downloading it only when it is missing."""
    parsed_spec = RoboflowDatasetSpec.parse(spec) if isinstance(spec, str) else spec
    destination = Path(location).expanduser().resolve()
    _validate_destination(destination, force=force)
    try:
        cached = _cached_dataset(destination, parsed_spec, model_format)
    except DatasetPreparationError:
        if not force:
            raise
        cached = None
    if cached is not None and not force:
        return cached

    key = api_key or os.getenv("ROBOFLOW_API_KEY")
    if not key:
        raise DatasetPreparationError(
            "ROBOFLOW_API_KEY is required to download the dataset; "
            "set it or pass api_key explicitly"
        )
    if destination.exists() and any(destination.iterdir()) and not force:
        raise DatasetPreparationError(
            f"Dataset directory is not a valid cache: {destination}. Use --force to replace it."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="roboflow-", dir=destination.parent) as temporary:
        temporary_location = Path(temporary) / "export"
        downloaded_location = _download_with_roboflow(
            parsed_spec, temporary_location, key, model_format
        )
        validate_yolo_export(downloaded_location)
        if destination.is_dir():
            shutil.rmtree(destination)
        elif destination.exists() or destination.is_symlink():
            destination.unlink()
        shutil.move(str(downloaded_location), str(destination))

    data_yaml = validate_yolo_export(destination)
    manifest_path = destination / MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(
            {
                "dataset": str(parsed_spec),
                "format": model_format,
                "data_yaml": str(data_yaml.relative_to(destination)),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return PreparedDataset(destination, data_yaml, manifest_path, parsed_spec, model_format)
