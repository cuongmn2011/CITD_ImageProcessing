import json
import subprocess
import sys
import zipfile


def test_package_run_selects_completed_run_and_writes_manifest(tmp_path) -> None:
    runs_root = tmp_path / "runs" / "detect"
    (runs_root / "train").mkdir(parents=True)
    run_dir = runs_root / "train-2"
    weights = run_dir / "weights"
    weights.mkdir(parents=True)
    (weights / "best.pt").write_bytes(b"best")
    (weights / "last.pt").write_bytes(b"last")
    (run_dir / "results.csv").write_text("epoch,metric\n1,0.5\n", encoding="utf-8")
    output = tmp_path / "artifacts" / "training.zip"

    subprocess.run(
        [
            sys.executable,
            "scripts/package-run.py",
            "--runs-root",
            str(runs_root),
            "--output",
            str(output),
        ],
        check=True,
    )

    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        metadata = json.loads(archive.read("metadata.json"))

    assert {
        "best.pt",
        "last.pt",
        "run/weights/best.pt",
        "run/results.csv",
        "metadata.json",
    } <= names
    assert metadata["best_model"] == "best.pt"
    assert metadata["last_model"] == "last.pt"
