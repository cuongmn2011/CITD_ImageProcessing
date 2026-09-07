import subprocess
import sys
from runpy import run_path


def test_bootstrap_kaggle_help_is_available() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/bootstrap-kaggle.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--mode" in completed.stdout
    assert "train" in completed.stdout
    assert "inference" in completed.stdout


def test_bootstrap_maps_thop_distribution_to_importable_module() -> None:
    bootstrap = run_path("scripts/bootstrap-kaggle.py", run_name="bootstrap_kaggle")

    assert ("thop", "ultralytics-thop") in bootstrap["VISION_PACKAGES"]
