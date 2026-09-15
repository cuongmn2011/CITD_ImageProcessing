import pytest

from lpr.ocr_dataset import (
    CharacterBox,
    OcrDatasetError,
    OcrSample,
    boxes_to_text,
    convert_yolo_char_labels,
    group_lines,
    load_class_names,
    parse_filename_labeled_samples,
    write_paddlex_rec_dataset,
)

CLASS_NAMES = {0: "2", 1: "9", 2: "A", 3: "1", 4: "T"}


def _write_char_export(root) -> None:
    (root / "train" / "images").mkdir(parents=True)
    (root / "train" / "labels").mkdir(parents=True)
    (root / "valid" / "images").mkdir(parents=True)
    (root / "valid" / "labels").mkdir(parents=True)
    (root / "data.yaml").write_text(
        "names:\n  0: '2'\n  1: '9'\n  2: 'A'\n  3: '1'\n  4: 'T'\n",
        encoding="utf-8",
    )
    # One-line plate "29A" as three character boxes at the same height.
    (root / "train" / "images" / "car.jpg").write_bytes(b"image")
    (root / "train" / "labels" / "car.txt").write_text(
        "0 0.10 0.50 0.05 0.10\n1 0.20 0.50 0.05 0.10\n2 0.30 0.50 0.05 0.10\n",
        encoding="utf-8",
    )
    # Two-line motorbike plate: top line "29T" (y=0.2), bottom line "1A" (y=0.8).
    (root / "valid" / "images" / "moto.jpg").write_bytes(b"image")
    (root / "valid" / "labels" / "moto.txt").write_text(
        "0 0.10 0.20 0.05 0.10\n4 0.30 0.20 0.05 0.10\n1 0.20 0.20 0.05 0.10\n"
        "3 0.20 0.80 0.05 0.10\n2 0.10 0.80 0.05 0.10\n",
        encoding="utf-8",
    )


def test_group_lines_clusters_by_vertical_gap() -> None:
    boxes = [
        CharacterBox("2", 0.1, 0.2, 0.05, 0.1),
        CharacterBox("9", 0.2, 0.21, 0.05, 0.1),
        CharacterBox("1", 0.2, 0.8, 0.05, 0.1),
    ]
    lines = group_lines(boxes, y_threshold=0.08)
    assert [len(line) for line in lines] == [2, 1]


def test_group_lines_keeps_single_line_plate_together() -> None:
    boxes = [
        CharacterBox(char, x, 0.5, 0.05, 0.1) for char, x in [("2", 0.1), ("9", 0.2), ("A", 0.3)]
    ]
    lines = group_lines(boxes, y_threshold=0.08)
    assert len(lines) == 1


def test_boxes_to_text_sorts_left_to_right_then_top_to_bottom() -> None:
    lines = [
        [CharacterBox("9", 0.3, 0.2, 0.05, 0.1), CharacterBox("2", 0.1, 0.2, 0.05, 0.1)],
        [CharacterBox("A", 0.2, 0.8, 0.05, 0.1)],
    ]
    assert boxes_to_text(lines) == "29A"


def test_load_class_names_reads_mapping(tmp_path) -> None:
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("names:\n  0: '0'\n  1: 'A'\n", encoding="utf-8")
    assert load_class_names(data_yaml) == {0: "0", 1: "A"}


def test_convert_yolo_char_labels_reconstructs_text(tmp_path) -> None:
    _write_char_export(tmp_path)
    samples = convert_yolo_char_labels(tmp_path, CLASS_NAMES, y_threshold=0.08)
    texts = {sample.image_path.name: sample.text for sample in samples}
    assert texts == {"car.jpg": "29A", "moto.jpg": "29TA1"}
    assert {sample.source for sample in samples} == {"roboflow-chars"}


def test_convert_yolo_char_labels_rejects_unknown_class(tmp_path) -> None:
    _write_char_export(tmp_path)
    (tmp_path / "train" / "labels" / "car.txt").write_text(
        "99 0.5 0.5 0.1 0.1\n", encoding="utf-8"
    )
    with pytest.raises(OcrDatasetError, match="Unknown character class"):
        convert_yolo_char_labels(tmp_path, CLASS_NAMES)


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
    assert {sample.source for sample in samples} == {"github-filename"}


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
