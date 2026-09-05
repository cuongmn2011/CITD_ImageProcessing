# Dataset Runtime Integration

## Installation

```bash
uv sync --extra vision --extra dataset
```

The `dataset` extra provides the Roboflow SDK and YAML parser. The `vision` extra provides Ultralytics.

## Secret handling

Set the key in the process environment:

```bash
export ROBOFLOW_API_KEY="<your-roboflow-api-key>"
```

Do not place it in:

- Git files.
- `.dataset-manifest.json`.
- Shell scripts committed to the repository.
- Training output or screenshots.

## Explicit preparation

```bash
uv run --extra dataset python scripts/prepare-dataset.py
```

The command prints the local dataset location and generated `data.yaml`. The first successful run creates the ignored cache:

```text
data/processed/license-plates/
├── .dataset-manifest.json
├── data.yaml
├── train/
├── valid/ or val/
└── test/                  # when provided by the export
```

## Training preparation

The training script performs the same preparation automatically:

```bash
uv run --extra vision --extra dataset python scripts/train-yolo.py
```

Disable the Roboflow flow only when a manually prepared dataset configuration should be used:

```bash
uv run --extra vision python scripts/train-yolo.py \
  --no-download-dataset \
  --data configs/plate-dataset.yaml
```

## Cache behavior

| State | Behavior |
|---|---|
| Valid manifest and valid export | Reuse without API call. |
| No cache | Download, validate, move, and write manifest. |
| Existing non-cache directory | Refuse unless `--force` is supplied. |
| Invalid cache with `--force` | Download a fresh export and replace it. |
| Invalid cache without `--force` | Fail instead of silently deleting data. |
| Forced path outside `data/processed` | Refuse. |

## Operational troubleshooting

### Missing API key

Error indicates that `ROBOFLOW_API_KEY` is required. Export the key in the current shell and rerun.

### Missing SDK

Install the dataset extra:

```bash
uv sync --extra dataset
```

### Export validation failure

Check that the selected Roboflow version exports YOLOv8-compatible data with:

- `data.yaml`.
- `train/images` and `train/labels`.
- `valid/images` or `val/images` and corresponding labels.

Use `--force` only for the project-local dataset cache after confirming the source/version.

### Version override

```bash
export LPR_ROBOFLOW_DATASET="workspace/project/version"
```

or:

```bash
uv run --extra dataset python scripts/prepare-dataset.py \
  --dataset workspace/project/version
```

## Reproducibility checklist

Record the following in the final report after preparation:

- Roboflow workspace/project/version.
- Export format.
- Download date.
- Number of train/validation/test images.
- Number of labels and class distribution.
- Image dimensions and file formats.
- Duplicate or corrupt-file count.
- Exact training command.
- Model name, epochs, image size, batch, device, and seed if configured.
