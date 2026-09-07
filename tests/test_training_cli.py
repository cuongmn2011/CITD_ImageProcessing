import subprocess
import sys


def test_training_cli_exposes_workers_option() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/train-yolo.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--workers" in completed.stdout
