# CITD Image Processing

Vietnamese license-plate detection and recognition pipeline.

## Scope

The project implements a two-stage ALPR flow:

```text
Input image/video
      ↓
YOLO license-plate detector
      ↓
Crop + perspective/preprocessing variants
      ↓
OCR backend selection
      ↓
Annotated output / OCR metrics
```

Supported OCR backends:

- Tesseract
- EasyOCR
- PaddleOCR

The selected training dataset is the [Vietnamese Car License Plate Dataset by Cuong Ta on Roboflow](https://universe.roboflow.com/cuong-ta-ulxex/vietnamese-car-license-plate). Dataset files are **not stored in this repository**. They are downloaded lazily into the ignored `data/processed/` directory when the training flow runs.

## Requirements

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/)
- A Roboflow API key for runtime dataset download
- A Tesseract binary when using the Tesseract backend

Install base and development dependencies:

```bash
uv sync --extra dev
```

Install training and runtime dataset support:

```bash
uv sync --extra vision --extra dataset
```

Install optional OCR packages:

```bash
uv sync --extra ocr
```

## Dataset setup

Export the Roboflow key only in the shell or secret manager; never commit it:

```bash
export ROBOFLOW_API_KEY="<your-key>"
```

Prepare the dataset on demand:

```bash
uv run --extra dataset python scripts/prepare-dataset.py
```

Default dataset spec:

```text
cuong-ta-ulxex/vietnamese-car-license-plate/1
```

Override the version or cache location when needed:

```bash
uv run --extra dataset python scripts/prepare-dataset.py \
  --dataset cuong-ta-ulxex/vietnamese-car-license-plate/1 \
  --location data/processed/license-plates
```

The preparation command validates the Roboflow YOLO export and writes a local manifest. A valid cache is reused. To replace a cache, use `--force`; forced replacement is restricted to child paths under `data/processed/`.

## Training

Training automatically ensures the dataset cache exists before loading Ultralytics:

```bash
uv run --extra vision --extra dataset python scripts/train-yolo.py \
  --model yolo11s.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch -1
```

Use an existing manual YOLO configuration without downloading Roboflow:

```bash
uv run --extra vision python scripts/train-yolo.py \
  --no-download-dataset \
  --data configs/plate-dataset.yaml
```

No dataset, model weights, video, or generated training run is committed to Git.

## Inference

Image inference:

```bash
uv run lpr infer-image \
  --image path/to/image.jpg \
  --model models/best.pt \
  --ocr tesseract \
  --variants otsu,clahe
```

Video inference:

```bash
uv run lpr infer-video \
  --input path/to/input.mp4 \
  --output outputs/annotated.mp4 \
  --model models/best.pt \
  --ocr tesseract
```

Inference does not download the training dataset. Dataset download belongs to the training/data-preparation flow only.

## OCR evaluation

The evaluator expects a CSV with `ground_truth` and `prediction` columns:

```bash
uv run lpr evaluate-ocr --csv path/to/ocr.csv
```

It reports exact accuracy, character accuracy, and character error rate (CER). Missing or empty ground-truth values are rejected instead of being silently scored.

## Verification

Run the complete local checks:

```bash
uv run pytest
uv run ruff check src tests scripts
uv run python -m compileall -q src scripts
uv lock --check
```

## Documentation

- [Documentation index](docs/README.md)
- [Research and dataset selection](docs/research-dataset-selection.md)
- [Architecture and implementation](docs/architecture-and-implementation.md)
- [Dataset runtime integration](docs/dataset-runtime-integration.md)
- [Project journal and final-report guide](docs/project-journal.md)
- [Colab/Kaggle training notebook](notebooks/train_pipeline_colab_kaggle.ipynb)
- [Delivery and merge order](docs/delivery-and-merge-order.md)

## Current limitations

- The Roboflow dataset has not been downloaded or trained in this repository yet.
- No model accuracy metrics are claimed until a real training/evaluation run is completed.
- The base OpenCV headless dependency and Ultralytics' transitive OpenCV dependency should be tested together before production packaging.
