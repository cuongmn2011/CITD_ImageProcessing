"""Command-line interface for image, video, and OCR evaluation runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

import cv2

from .detector import YoloPlateDetector
from .metrics import evaluate_ocr_pairs
from .ocr import EasyOCRBackend, PaddleOCRBackend, TesseractBackend
from .pipeline import LicensePlateRecognizer


def _backend(name: str, gpu: bool):
    if name == "tesseract":
        return TesseractBackend()
    if name == "easyocr":
        return EasyOCRBackend(gpu=gpu)
    if name == "paddleocr":
        return PaddleOCRBackend()
    raise ValueError(f"Unsupported OCR backend: {name}")


def _recognizer(args: argparse.Namespace) -> LicensePlateRecognizer:
    detector = YoloPlateDetector(args.model, confidence=args.confidence, device=args.device)
    backends = [_backend(name, args.gpu) for name in args.ocr]
    variants = tuple(args.variants.split(","))
    return LicensePlateRecognizer(detector, backends, variants=variants, crop_padding=args.padding)


def _add_pipeline_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="Path to trained YOLO weights")
    parser.add_argument("--ocr", nargs="+", default=["tesseract"], choices=["tesseract", "easyocr", "paddleocr"])
    parser.add_argument("--variants", default="otsu,clahe", help="Comma-separated preprocessing variants")
    parser.add_argument("--confidence", type=float, default=0.4)
    parser.add_argument("--padding", type=float, default=0.08)
    parser.add_argument("--device", default=None, help="YOLO device, for example 0 or cpu")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for EasyOCR")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lpr", description="Vietnamese license plate recognition")
    subparsers = parser.add_subparsers(dest="command", required=True)

    image = subparsers.add_parser("infer-image", help="Recognize plates in one image")
    image.add_argument("--image", required=True)
    _add_pipeline_arguments(image)

    video = subparsers.add_parser("infer-video", help="Recognize plates in a video")
    video.add_argument("--input", required=True)
    video.add_argument("--output", required=True)
    video.add_argument("--max-frames", type=int, default=None)
    _add_pipeline_arguments(video)

    evaluate = subparsers.add_parser("evaluate-ocr", help="Evaluate a CSV with ground_truth,prediction columns")
    evaluate.add_argument("--csv", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "evaluate-ocr":
        with args.csv.open(newline="", encoding="utf-8") as file:
            print(json.dumps(evaluate_ocr_pairs(csv.DictReader(file)), indent=2))
        return 0

    pipeline = _recognizer(args)
    if args.command == "infer-image":
        image = cv2.imread(args.image)
        if image is None:
            raise SystemExit(f"Could not read image: {args.image}")
        results = pipeline.recognize_image(image)
        print(json.dumps([
            {
                "bbox": result.detection.bbox,
                "detection_confidence": result.detection.confidence,
                "text": result.ocr.text if result.ocr else "",
                "ocr_confidence": result.ocr.confidence if result.ocr else 0.0,
                "ocr_backend": result.ocr.backend if result.ocr else None,
                "preprocessing": result.ocr.variant if result.ocr else None,
            }
            for result in results
        ], indent=2))
        return 0

    frames = pipeline.recognize_video(args.input, args.output, max_frames=args.max_frames)
    print(json.dumps({"frames_processed": frames, "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
