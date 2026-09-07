"""Package a completed Ultralytics training run for download or reuse."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

DEFAULT_RUNS_ROOT = Path("runs/detect")
DEFAULT_OUTPUT = Path("outputs/citd-yolo11s-training.zip")


def _latest_run(runs_root: Path) -> Path:
    candidates = sorted(
        (
            path
            for path in runs_root.glob("train*")
            if (path / "weights" / "best.pt").is_file()
        ),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"No completed training run found under {runs_root}")
    return candidates[-1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_run(run_dir: Path, output: Path) -> Path:
    run_dir = run_dir.expanduser().resolve()
    output = output.expanduser().resolve()
    best_model = run_dir / "weights" / "best.pt"
    if not best_model.is_file():
        raise FileNotFoundError(f"Training run has no best.pt: {best_model}")

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    metadata = {
        "run_dir": str(run_dir),
        "best_model": "best.pt",
        "last_model": "last.pt" if (run_dir / "weights" / "last.pt").is_file() else None,
        "best_model_sha256": _sha256(best_model),
        "files": [],
    }
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(run_dir.rglob("*")):
            if path.is_file():
                archive_name = Path("run") / path.relative_to(run_dir)
                archive.write(path, archive_name.as_posix())
                metadata["files"].append(archive_name.as_posix())
        archive.write(best_model, "best.pt")
        metadata["files"].append("best.pt")
        last_model = run_dir / "weights" / "last.pt"
        if last_model.is_file():
            archive.write(last_model, "last.pt")
            metadata["files"].append("last.pt")
        archive.writestr("metadata.json", json.dumps(metadata, indent=2))

    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Completed run directory; defaults to newest runs/detect/train*",
    )
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    run_dir = args.run_dir or _latest_run(args.runs_root)
    output = package_run(run_dir, args.output)
    print(
        json.dumps(
            {"run": str(run_dir), "archive": str(output), "bytes": output.stat().st_size},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
