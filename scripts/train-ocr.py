"""Fine-tune a PaddleOCR recognition model via the PaddleX CLI.

PaddleOCR's own pip package only exposes inference (see src/lpr/ocr.py); actual
recognition-model fine-tuning is done through PaddleX's ``-o Global.mode=train``
CLI against a dataset built by scripts/prepare-ocr-dataset.py.
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a PaddleOCR recognition model")
    parser.add_argument("--dataset-dir", default="data/processed/ocr-rec-dataset")
    parser.add_argument(
        "--config",
        default="PP-OCRv5_mobile_rec",
        help="PaddleX builtin recognition config name or path to a config yaml",
    )
    parser.add_argument("--pretrained-model", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output-dir", default="outputs/ocr-rec-training")
    parser.add_argument("--device", default=None, help="PaddleX device, for example gpu:0 or cpu")
    args = parser.parse_args()

    try:
        import paddlex  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "Install OCR training support with: uv sync --extra ocr-train"
        ) from error

    overrides = [
        "Global.mode=train",
        f"Global.dataset_dir={args.dataset_dir}",
        f"Global.output_dir={args.output_dir}",
    ]
    if args.pretrained_model is not None:
        overrides.append(f"Global.pretrained_model={args.pretrained_model}")
    if args.epochs is not None:
        overrides.append(f"Train.epochs_iters={args.epochs}")
    if args.device is not None:
        overrides.append(f"Global.device={args.device}")

    command = [sys.executable, "-m", "paddlex", "-c", args.config]
    for override in overrides:
        command.extend(["-o", override])
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
