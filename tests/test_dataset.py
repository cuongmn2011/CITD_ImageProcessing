import json
import sys
from types import SimpleNamespace

import pytest

from lpr.dataset import (
    DatasetPreparationError,
    RoboflowDatasetSpec,
    ensure_roboflow_dataset,
    validate_yolo_export,
)


def _write_yolo_export(root) -> None:
    for split in ("train", "valid"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
    (root / "data.yaml").write_text(
        "path: .\ntrain: train/images\nval: valid/images\nnames: [plate]\n",
        encoding="utf-8",
    )
    (root / "train" / "images" / "plate.jpg").write_bytes(b"image")
    (root / "train" / "labels" / "plate.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    (root / "valid" / "images" / "plate.jpg").write_bytes(b"image")
    (root / "valid" / "labels" / "plate.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")


def test_dataset_spec_parses_and_rejects_invalid_values() -> None:
    assert str(RoboflowDatasetSpec.parse("workspace/project/3")) == "workspace/project/3"
    with pytest.raises(ValueError, match="workspace/project/version"):
        RoboflowDatasetSpec.parse("workspace/project")
    with pytest.raises(ValueError, match="positive"):
        RoboflowDatasetSpec.parse("workspace/project/0")


def test_validate_yolo_export_requires_train_and_validation_splits(tmp_path) -> None:
    _write_yolo_export(tmp_path)
    assert validate_yolo_export(tmp_path) == (tmp_path / "data.yaml").resolve()
    (tmp_path / "valid" / "labels" / "plate.txt").unlink()
    with pytest.raises(DatasetPreparationError, match="validation split"):
        validate_yolo_export(tmp_path)


def test_validate_yolo_export_normalizes_parent_relative_roboflow_paths(tmp_path) -> None:
    _write_yolo_export(tmp_path)
    (tmp_path / "data.yaml").write_text(
        "train: ../train/images\nval: ../valid/images\nnames: [plate]\n",
        encoding="utf-8",
    )

    assert validate_yolo_export(tmp_path) == (tmp_path / "data.yaml").resolve()

    normalized = (tmp_path / "data.yaml").read_text(encoding="utf-8")
    assert f"path: {tmp_path.resolve()}" in normalized
    assert "train: train/images" in normalized
    assert "val: valid/images" in normalized


def test_validate_yolo_export_rewrites_relative_root_for_ultralytics(tmp_path) -> None:
    _write_yolo_export(tmp_path)

    assert validate_yolo_export(tmp_path) == (tmp_path / "data.yaml").resolve()

    normalized = (tmp_path / "data.yaml").read_text(encoding="utf-8")
    assert f"path: {tmp_path.resolve()}" in normalized


def test_force_rejects_project_directory(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(DatasetPreparationError, match="dedicated cache child"):
        ensure_roboflow_dataset("workspace/project/1", tmp_path, api_key="secret", force=True)
    with pytest.raises(DatasetPreparationError, match="Forced replacement"):
        ensure_roboflow_dataset(
            "workspace/project/1", tmp_path.parent / "external-cache", api_key="secret", force=True
        )
    with pytest.raises(DatasetPreparationError, match="child of"):
        ensure_roboflow_dataset(
            "workspace/project/1", tmp_path / "scripts", api_key="secret", force=True
        )


def test_ensure_reuses_valid_cache_without_api_key(tmp_path) -> None:
    _write_yolo_export(tmp_path)
    manifest = tmp_path / ".dataset-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset": "workspace/project/1",
                "format": "yolov8",
                "data_yaml": "data.yaml",
            }
        ),
        encoding="utf-8",
    )
    prepared = ensure_roboflow_dataset("workspace/project/1", tmp_path)
    assert prepared.data_yaml == (tmp_path / "data.yaml").resolve()


def test_ensure_downloads_and_writes_manifest(tmp_path, monkeypatch) -> None:
    class FakeVersion:
        def download(self, *, model_format, location, overwrite):
            assert model_format == "yolov8"
            assert overwrite is True
            export = tmp_path / "sdk-export"
            _write_yolo_export(export)
            return SimpleNamespace(location=str(export))

    class FakeProject:
        def version(self, version):
            assert version == 1
            return FakeVersion()

    class FakeRoboflow:
        def __init__(self, *, api_key):
            assert api_key == "secret"

        def workspace(self, workspace):
            assert workspace == "workspace"
            return SimpleNamespace(project=lambda project: FakeProject())

    monkeypatch.setitem(sys.modules, "roboflow", SimpleNamespace(Roboflow=FakeRoboflow))
    prepared = ensure_roboflow_dataset(
        "workspace/project/1", tmp_path / "prepared", api_key="secret"
    )
    assert prepared.data_yaml.is_file()
    assert (
        json.loads(prepared.manifest.read_text(encoding="utf-8"))["dataset"]
        == "workspace/project/1"
    )
