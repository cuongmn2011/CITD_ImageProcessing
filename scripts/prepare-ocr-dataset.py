"""Build a PaddleX OCR recognition dataset from the filename-labeled GitHub source.

Ground truth is read directly from each plate crop's filename (for example
``29A87180_1212_0.jpg``). The source repo has no LICENSE file: restrict use to
this academic project, see docs/ocr-dataset-selection.md for the full caveat.
"""

from __future__ import annotations

import argparse
import json

from lpr.dataset import (
    DEFAULT_OCR_DATASET_LOCATION,
    DEFAULT_OCR_DATASET_REF,
    DEFAULT_OCR_DATASET_REPO,
    DEFAULT_OCR_DATASET_SUBDIR,
    EXTRA_OCR_DATASET_LOCATION,
    EXTRA_OCR_DATASET_REF,
    EXTRA_OCR_DATASET_REPO,
    EXTRA_OCR_DATASET_SUBDIR,
    DatasetPreparationError,
    ensure_git_dataset,
)
from lpr.ocr_dataset import (
    DEFAULT_VAL_RATIO,
    OcrDatasetError,
    parse_filename_labeled_samples,
    write_paddlex_rec_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare a PaddleX OCR recognition dataset from filename-labeled plate crops"
    )
    parser.add_argument("--repo", default=DEFAULT_OCR_DATASET_REPO)
    parser.add_argument("--ref", default=DEFAULT_OCR_DATASET_REF)
    parser.add_argument("--subdir", default=DEFAULT_OCR_DATASET_SUBDIR)
    parser.add_argument("--location", default=DEFAULT_OCR_DATASET_LOCATION)
    parser.add_argument(
        "--extra-repo",
        default=EXTRA_OCR_DATASET_REPO,
        help="Second filename-labeled source to merge in (see docs/ocr-dataset-selection.md)",
    )
    parser.add_argument("--extra-ref", default=EXTRA_OCR_DATASET_REF)
    parser.add_argument("--extra-subdir", default=EXTRA_OCR_DATASET_SUBDIR)
    parser.add_argument("--extra-location", default=EXTRA_OCR_DATASET_LOCATION)
    parser.add_argument(
        "--no-extra-source",
        action="store_true",
        help="Skip the extra source and use only --repo/--subdir",
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing dataset cache")
    parser.add_argument("--out", default="data/processed/ocr-rec-dataset")
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    args = parser.parse_args()

    try:
        git_dataset = ensure_git_dataset(
            args.repo,
            args.location,
            ref=args.ref,
            subdir=args.subdir,
            force=args.force,
        )
        samples = parse_filename_labeled_samples(git_dataset.location)

        extra_dataset = None
        if not args.no_extra_source:
            extra_dataset = ensure_git_dataset(
                args.extra_repo,
                args.extra_location,
                ref=args.extra_ref,
                subdir=args.extra_subdir,
                force=args.force,
            )
            samples = samples + parse_filename_labeled_samples(extra_dataset.location)

        dataset_dir = write_paddlex_rec_dataset(samples, args.out, val_ratio=args.val_ratio)
    except (DatasetPreparationError, OcrDatasetError, ValueError) as error:
        parser.error(str(error))

    result = {
        "repo": args.repo,
        "ref": args.ref,
        "commit": git_dataset.commit,
        "location": str(git_dataset.location),
        "samples": len(samples),
        "output_dir": str(dataset_dir),
    }
    if extra_dataset is not None:
        result["extra_repo"] = args.extra_repo
        result["extra_ref"] = args.extra_ref
        result["extra_commit"] = extra_dataset.commit
        result["extra_location"] = str(extra_dataset.location)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
