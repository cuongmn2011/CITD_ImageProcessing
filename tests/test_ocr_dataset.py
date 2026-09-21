import pytest

from lpr.ocr_dataset import (
    OcrDatasetError,
    OcrSample,
    parse_filename_labeled_samples,
    write_paddlex_rec_dataset,
)


def test_write_paddlex_rec_dataset_splits_train_and_val(tmp_path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    samples = []
    for index in range(20):
        image_path = source_dir / f"plate-{index}.jpg"
        image_path.write_bytes(b"image")
        samples.append(OcrSample(image_path, f"29A{index:04d}"))

    out_dir = write_paddlex_rec_dataset(samples, tmp_path / "out", val_ratio=0.5)
    train_lines = (out_dir / "train.txt").read_text(encoding="utf-8").splitlines()
    val_lines = (out_dir / "val.txt").read_text(encoding="utf-8").splitlines()
    assert len(train_lines) + len(val_lines) == len(samples)
    assert train_lines and val_lines
    for line in train_lines + val_lines:
        image_ref, text = line.split("\t")
        assert (out_dir / image_ref).is_file()
        assert text.startswith("29A")


def test_parse_filename_labeled_samples_extracts_prefix_before_underscore(tmp_path) -> None:
    (tmp_path / "29A87180_1212_0.jpg").write_bytes(b"image")
    (tmp_path / "30A99279_1306_0.jpg").write_bytes(b"image")

    samples = parse_filename_labeled_samples(tmp_path)

    texts = {sample.image_path.name: sample.text for sample in samples}
    assert texts == {"29A87180_1212_0.jpg": "29A87180", "30A99279_1306_0.jpg": "30A99279"}


def test_parse_filename_labeled_samples_skips_non_plate_names(tmp_path) -> None:
    (tmp_path / "29A87180_1212_0.jpg").write_bytes(b"image")
    (tmp_path / "___.jpg").write_bytes(b"image")

    samples = parse_filename_labeled_samples(tmp_path)

    assert [sample.image_path.name for sample in samples] == ["29A87180_1212_0.jpg"]


def test_parse_filename_labeled_samples_recurses_into_subdirectories(tmp_path) -> None:
    nested = tmp_path / "batch1"
    nested.mkdir()
    (nested / "51F12345_0001_0.jpg").write_bytes(b"image")

    samples = parse_filename_labeled_samples(tmp_path)

    assert [sample.text for sample in samples] == ["51F12345"]


def test_parse_filename_labeled_samples_rejects_empty_directory(tmp_path) -> None:
    with pytest.raises(OcrDatasetError, match="No filename-labeled samples"):
        parse_filename_labeled_samples(tmp_path)
