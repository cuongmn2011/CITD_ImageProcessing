"""Fine-tune a PaddleOCR recognition model via the PaddleOCR training plugin.

PaddleOCR's own pip package only exposes inference (see src/lpr/ocr.py), and the
`paddlex` pip package's CLI/high-level API is inference/serving-only too --
training requires the `PaddleOCR` training plugin, installed on demand via
`python -m paddlex --install PaddleOCR`, which provides the real
`tools/train.py` entrypoint and its config YAML files.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _paddleocr_repo_dir() -> Path:
    import paddlex

    return Path(paddlex.__file__).parent / "repo_manager" / "repos" / "PaddleOCR"


def _ensure_paddleocr_plugin(repo_dir: Path) -> None:
    if (repo_dir / "tools" / "train.py").is_file():
        return
    # paddlex.repo_manager assumes its "repos" parent directory already exists when
    # it tries to clean up a previous (possibly absent) install before cloning.
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "-m", "paddlex", "--install", "PaddleOCR", "-y"], check=True
    )
    if not (repo_dir / "tools" / "train.py").is_file():
        raise RuntimeError(f"PaddleOCR plugin install did not produce {repo_dir}/tools/train.py")


def _find_config(repo_dir: Path, config: str) -> Path:
    config_path = Path(config)
    if config_path.is_file():
        return config_path
    matches = sorted(repo_dir.glob(f"**/*{config}*.yml")) + sorted(
        repo_dir.glob(f"**/*{config}*.yaml")
    )
    if not matches:
        raise RuntimeError(f"No config matching {config!r} found under {repo_dir}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a PaddleOCR recognition model")
    parser.add_argument("--dataset-dir", default="data/processed/ocr-rec-dataset")
    parser.add_argument(
        "--config",
        default="PP-OCRv5_mobile_rec",
        help="Config name to search for under the PaddleOCR plugin, or a path to a config yaml",
    )
    parser.add_argument("--pretrained-model", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the config's batch_size_per_card (128 by default; heavy on CPU)",
    )
    parser.add_argument("--output-dir", default="outputs/ocr-rec-training")
    parser.add_argument(
        "--use-gpu", action="store_true", help="Train on GPU instead of CPU"
    )
    args = parser.parse_args()

    try:
        import paddlex  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "Install OCR training support with: uv sync --extra ocr-train"
        ) from error

    repo_dir = _paddleocr_repo_dir()
    _ensure_paddleocr_plugin(repo_dir)
    config_path = _find_config(repo_dir, args.config)

    # tools/train.py runs with cwd=repo_dir (under site-packages), so relative paths
    # from the caller's working directory must be resolved to absolute paths first.
    dataset_dir = Path(args.dataset_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    overrides = [
        # tools/program.py reads config["Global"]["use_gpu"] (yaml-parsed, so lowercase
        # true/false is fine) to pick "gpu:{id}" vs "cpu"; PP-OCRv5 configs default to true.
        f"Global.use_gpu={'true' if args.use_gpu else 'false'}",
        f"Global.save_model_dir={output_dir}",
        f"Train.dataset.data_dir={dataset_dir}",
        f"Train.dataset.label_file_list=[{dataset_dir}/train.txt]",
        f"Eval.dataset.data_dir={dataset_dir}",
        f"Eval.dataset.label_file_list=[{dataset_dir}/val.txt]",
    ]
    if args.pretrained_model is not None:
        overrides.append(f"Global.pretrained_model={args.pretrained_model}")
    if args.epochs is not None:
        overrides.append(f"Global.epoch_num={args.epochs}")
    if args.batch_size is not None:
        overrides.append(f"Train.loader.batch_size_per_card={args.batch_size}")
        overrides.append(f"Eval.loader.batch_size_per_card={args.batch_size}")
        # MultiScaleSampler ignores loader.batch_size_per_card and uses first_bs instead.
        overrides.append(f"Train.sampler.first_bs={args.batch_size}")

    # PaddleOCR's -o takes nargs="+": a single "-o" followed by all key=value pairs.
    # Repeating "-o" for each override (the previous bug here) makes argparse keep
    # only the last one, silently dropping every earlier override.
    command = [sys.executable, "tools/train.py", "-c", str(config_path), "-o", *overrides]
    subprocess.run(command, check=True, cwd=repo_dir)


if __name__ == "__main__":
    main()
