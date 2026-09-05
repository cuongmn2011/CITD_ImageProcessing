"""Download and validate the configured Roboflow dataset on demand."""

from __future__ import annotations

import argparse
import json
import os

from lpr.dataset import (
    DEFAULT_DATASET_FORMAT,
    DEFAULT_DATASET_LOCATION,
    DEFAULT_DATASET_SPEC,
    DatasetPreparationError,
    ensure_roboflow_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare the Roboflow YOLO dataset on demand")
    parser.add_argument(
        "--dataset", default=os.getenv("LPR_ROBOFLOW_DATASET", DEFAULT_DATASET_SPEC)
    )
    parser.add_argument("--location", default=DEFAULT_DATASET_LOCATION)
    parser.add_argument("--format", dest="model_format", default=DEFAULT_DATASET_FORMAT)
    parser.add_argument(
        "--api-key", default=None, help="Roboflow API key; defaults to ROBOFLOW_API_KEY"
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing dataset cache")
    args = parser.parse_args()

    try:
        prepared = ensure_roboflow_dataset(
            args.dataset,
            args.location,
            api_key=args.api_key,
            model_format=args.model_format,
            force=args.force,
        )
    except (DatasetPreparationError, ValueError) as error:
        parser.error(str(error))

    print(
        json.dumps(
            {
                "dataset": str(prepared.spec),
                "location": str(prepared.location),
                "data_yaml": str(prepared.data_yaml),
                "manifest": str(prepared.manifest),
                "format": prepared.model_format,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
