"""Build a PaddleX OCR recognition dataset from two merged sources.

Primary source: a character-annotated Roboflow export (clean CC BY 4.0 license,
but only ~200 images). Supplementary source: the PBL4_Deep-Learning GitHub
repo, which has no LICENSE file (academic/non-commercial use only) but a much
larger, filename-labeled image set. See docs/ocr-dataset-selection.md.
"""

from __future__ import annotations

import argparse
import json
import os

from lpr.dataset import (
    DEFAULT_OCR_DATASET_LOCATION,
    DEFAULT_OCR_DATASET_SPEC,
    DEFAULT_OCR_SUPPLEMENT_LOCATION,
    DEFAULT_OCR_SUPPLEMENT_REF,
    DEFAULT_OCR_SUPPLEMENT_REPO,
    DEFAULT_OCR_SUPPLEMENT_SUBDIR,
    DatasetPreparationError,
    ensure_git_dataset,
    ensure_roboflow_dataset,
)
from lpr.ocr_dataset import (
    DEFAULT_VAL_RATIO,
    DEFAULT_Y_THRESHOLD,
    OcrDatasetError,
    convert_yolo_char_labels,
    load_class_names,
    parse_filename_labeled_samples,
    write_paddlex_rec_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare a PaddleX OCR recognition dataset from character-box labels"
        " plus an optional filename-labeled supplement"
    )
    parser.add_argument(
        "--dataset", default=os.getenv("LPR_OCR_ROBOFLOW_DATASET", DEFAULT_OCR_DATASET_SPEC)
    )
    parser.add_argument("--location", default=DEFAULT_OCR_DATASET_LOCATION)
    parser.add_argument(
        "--api-key", default=None, help="Roboflow API key; defaults to ROBOFLOW_API_KEY"
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing dataset cache")
    parser.add_argument(
        "--no-github-supplement",
        action="store_true",
        help="Skip the filename-labeled GitHub supplement and use Roboflow samples only",
    )
    parser.add_argument("--github-repo", default=DEFAULT_OCR_SUPPLEMENT_REPO)
    parser.add_argument("--github-ref", default=DEFAULT_OCR_SUPPLEMENT_REF)
    parser.add_argument("--github-subdir", default=DEFAULT_OCR_SUPPLEMENT_SUBDIR)
    parser.add_argument("--github-location", default=DEFAULT_OCR_SUPPLEMENT_LOCATION)
    parser.add_argument(
        "--force-github", action="store_true", help="Replace an existing GitHub dataset cache"
    )
    parser.add_argument("--out", default="data/processed/ocr-rec-dataset")
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--y-threshold", type=float, default=DEFAULT_Y_THRESHOLD)
    args = parser.parse_args()

    github_commit = None
    try:
        prepared = ensure_roboflow_dataset(
            args.dataset,
            args.location,
            api_key=args.api_key,
            force=args.force,
        )
        class_names = load_class_names(prepared.data_yaml)
        roboflow_samples = convert_yolo_char_labels(
            prepared.data_yaml.parent, class_names, y_threshold=args.y_threshold
        )

        github_samples = []
        if not args.no_github_supplement:
            git_dataset = ensure_git_dataset(
                args.github_repo,
                args.github_location,
                ref=args.github_ref,
                subdir=args.github_subdir,
                force=args.force_github,
            )
            github_commit = git_dataset.commit
            github_samples = parse_filename_labeled_samples(git_dataset.location)

        all_samples = roboflow_samples + github_samples
        dataset_dir = write_paddlex_rec_dataset(all_samples, args.out, val_ratio=args.val_ratio)
    except (DatasetPreparationError, OcrDatasetError, ValueError) as error:
        parser.error(str(error))

    print(
        json.dumps(
            {
                "dataset": str(prepared.spec),
                "location": str(prepared.location),
                "roboflow_samples": len(roboflow_samples),
                "github_samples": len(github_samples),
                "github_commit": github_commit,
                "total_samples": len(all_samples),
                "output_dir": str(dataset_dir),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
