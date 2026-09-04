"""Train a YOLO plate detector with a reproducible CLI."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="configs/plate-dataset.yaml")
    parser.add_argument("--model", default="yolo11s.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", default="-1")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    train_args = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": int(args.batch) if args.batch.lstrip("-").isdigit() else args.batch,
    }
    if args.device is not None:
        train_args["device"] = args.device
    model.train(**train_args)


if __name__ == "__main__":
    main()
