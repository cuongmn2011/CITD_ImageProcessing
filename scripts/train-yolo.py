"""Train a YOLO plate detector with a reproducible CLI."""

from __future__ import annotations

import argparse
import os

from lpr.dataset import (
    DEFAULT_DATASET_FORMAT,
    DEFAULT_DATASET_LOCATION,
    DEFAULT_DATASET_SPEC,
    DatasetPreparationError,
    ensure_roboflow_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="configs/plate-dataset.yaml")
    parser.add_argument(
        "--dataset", default=os.getenv("LPR_ROBOFLOW_DATASET", DEFAULT_DATASET_SPEC)
    )
    parser.add_argument("--dataset-location", default=DEFAULT_DATASET_LOCATION)
    parser.add_argument("--dataset-format", default=DEFAULT_DATASET_FORMAT)
    parser.add_argument(
        "--no-download-dataset",
        action="store_true",
        help="Use --data directly instead of ensuring the Roboflow dataset",
    )
    parser.add_argument(
        "--force-dataset",
        action="store_true",
        help="Replace an existing invalid or stale dataset cache",
    )
    parser.add_argument("--model", default="yolo11s.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", default="-1")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    data_config = args.data
    if not args.no_download_dataset:
        try:
            prepared = ensure_roboflow_dataset(
                args.dataset,
                args.dataset_location,
                model_format=args.dataset_format,
                force=args.force_dataset,
            )
        except (DatasetPreparationError, ValueError) as error:
            parser.error(str(error))
        data_config = str(prepared.data_yaml)

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("Install YOLO support with: uv sync --extra vision") from error

    model = YOLO(args.model)
    train_args = {
        "data": data_config,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": int(args.batch) if args.batch.lstrip("-").isdigit() else args.batch,
    }
    if args.device is not None:
        train_args["device"] = args.device
    model.train(**train_args)


if __name__ == "__main__":
    main()
