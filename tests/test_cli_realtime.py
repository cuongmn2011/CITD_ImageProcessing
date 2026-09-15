import pytest

from lpr.cli import build_parser


def test_serve_parser_exposes_runtime_options() -> None:
    args = build_parser().parse_args(
        [
            "serve",
            "--model",
            "best.pt",
            "--ocr-backend",
            "easyocr",
            "--imgsz",
            "480",
            "--port",
            "9000",
        ]
    )

    assert args.command == "serve"
    assert args.ocr_backend == "easyocr"
    assert args.imgsz == 480
    assert args.port == 9000


def test_pipeline_confidence_rejects_zero() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["infer-image", "--image", "input.jpg", "--model", "best.pt", "--confidence", "0"]
        )
