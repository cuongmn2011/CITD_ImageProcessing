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


def test_prepare_ocr_dataset_cli_exposes_github_supplement_options() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/prepare-ocr-dataset.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--no-github-supplement" in completed.stdout
    assert "--github-repo" in completed.stdout


def test_train_ocr_cli_exposes_config_option() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/train-ocr.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--config" in completed.stdout
