import subprocess
import sys


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
