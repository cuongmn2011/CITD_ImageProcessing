from pathlib import Path

from lpr.cli import build_parser


def test_infer_image_accepts_annotated_output_path() -> None:
    args = build_parser().parse_args(
        [
            "infer-image",
            "--image",
            "input.jpg",
            "--model",
            "best.pt",
            "--output",
            "annotated.jpg",
        ]
    )

    assert args.output == Path("annotated.jpg")
