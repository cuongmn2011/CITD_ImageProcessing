"""Export a trained YOLO plate detector to OpenVINO for faster CPU inference.

This converts the already-trained weights to a different runtime format; it does not
retrain anything. Requires the openvino extra: uv sync --extra vision --extra openvino

Run:
  uv run python scripts/export-yolo-openvino.py --model model/citd-yolo11s-training.zip
Then point YoloPlateDetector / --model at the printed <name>_openvino_model directory.
"""

from __future__ import annotations

import argparse

from lpr.server import resolve_model_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="model/citd-yolo11s-training.zip",
        help="YOLO weights (best.pt) or a training archive containing one",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640, help="Must match the imgsz used at inference time"
    )
    parser.add_argument(
        "--half", action="store_true", help="Export FP16 weights (smaller, GPU/NPU-oriented)"
    )
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit("Install YOLO support with: uv sync --extra vision") from error

    model_path = resolve_model_path(args.model)
    model = YOLO(str(model_path))
    try:
        exported = model.export(format="openvino", imgsz=args.imgsz, half=args.half)
    except Exception as error:
        raise SystemExit(
            "Export failed. Install the openvino extra with: uv sync --extra openvino"
        ) from error
    print(f"Exported to: {exported}")


if __name__ == "__main__":
    main()
