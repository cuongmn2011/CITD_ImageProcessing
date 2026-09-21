import subprocess
import sys


def test_prepare_ocr_dataset_cli_exposes_val_ratio_option() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/prepare-ocr-dataset.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--val-ratio" in completed.stdout


def test_prepare_ocr_dataset_cli_exposes_repo_options() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/prepare-ocr-dataset.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--repo" in completed.stdout
    assert "--ref" in completed.stdout
    assert "--subdir" in completed.stdout


def test_prepare_ocr_dataset_cli_exposes_extra_source_options() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/prepare-ocr-dataset.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--extra-repo" in completed.stdout
    assert "--extra-ref" in completed.stdout
    assert "--extra-subdir" in completed.stdout
    assert "--no-extra-source" in completed.stdout


def test_train_ocr_cli_exposes_config_option() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/train-ocr.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--config" in completed.stdout
