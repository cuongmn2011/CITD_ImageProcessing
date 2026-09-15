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

## OCR training versus detector training

The artifact in `outputs/citd-yolo11s-training.zip` trains a one-class **plate detector** only.
It contains bounding boxes, not plate-text transcripts, so retraining YOLO from this artifact
cannot teach OCR to read characters. The current validation metrics (`mAP50=0.99495`,
`mAP50-95=0.72907`) measure detection, not OCR accuracy.

First diagnose OCR on real crops with a backend that is installed:

```bash
uv run lpr infer-image \
  --image path/to/car.jpg \
  --model outputs/.lpr-model/best.pt \
  --ocr easyocr \
  --variants raw,gray,otsu,adaptive,clahe
```

To fine-tune the detector for tighter plate crops, use the existing `best.pt` as initialization
after the dataset cache is available:

```bash
export ROBOFLOW_API_KEY="<your-key>"
uv run --extra vision --extra dataset python scripts/train-yolo.py \
  --model outputs/.lpr-model/best.pt \
  --epochs 100 \
  --imgsz 960 \
  --batch -1 \
  --workers 2
```

Actual OCR training requires a separate labeled set containing one plate crop and its exact
transcription per row, for example `crop_path,ground_truth`. Without those transcripts, improve
OCR by selecting PaddleOCR/EasyOCR, crop quality, preprocessing, and temporal voting instead of
claiming that detector retraining fixed recognition.

## OCR training

The OCR backends above are pretrained and inference-only. To fine-tune PaddleOCR's recognition
model on Vietnamese plate characters instead of relying on a generic pretrained model, install
the training extra:

```bash
uv sync --extra ocr-train
```

Build a PaddleX recognition dataset by merging two sources: a character-annotated Roboflow
export (per-character boxes converted into plate-text ground truth by line-clustering and
left-to-right sorting) and a filename-labeled GitHub supplement (ground truth read straight from
each crop's filename, requires `git` on `PATH`). Requires network access; `git clone` needs the
system `git` binary:

```bash
export ROBOFLOW_API_KEY="<your-key>"
uv run --extra dataset --extra ocr-train python scripts/prepare-ocr-dataset.py
```

Use `--no-github-supplement` to build from the Roboflow export alone. The GitHub supplement has
no LICENSE file — see [docs/ocr-dataset-selection.md](docs/ocr-dataset-selection.md) for the
academic/non-commercial-only caveat before using it beyond this project.

Fine-tune with PaddleX:

```bash
uv run --extra ocr-train python scripts/train-ocr.py \
  --dataset-dir data/processed/ocr-rec-dataset \
  --config PP-OCRv5_mobile_rec
```

Use the fine-tuned model at inference time:

```bash
uv run lpr infer-image \
  --image path/to/car.jpg \
  --model models/best.pt \
  --ocr paddleocr \
  --rec-model-dir outputs/ocr-rec-training/best_model
```

The dataset source, license, and the character-line reconstruction method are documented in
[docs/ocr-dataset-selection.md](docs/ocr-dataset-selection.md). No OCR accuracy claim is made
until a real fine-tuning run and held-out evaluation exist.

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

## Realtime React demo

The realtime demo keeps a local video in the browser and sends sampled JPEG frames to a GPU API. The API returns detection/OCR JSON over WebSocket; React draws the overlay locally. It does not wait for a complete annotated video.

Run the frontend locally:

```bash
cd web
npm install
npm run dev
```

Start the GPU API on Colab:

```bash
uv run --extra vision --extra paddle --extra web lpr serve \
  --model /content/drive/MyDrive/lpr/best.pt \
  --ocr-backend paddleocr \
  --device 0 \
  --imgsz 640 \
  --host 0.0.0.0 \
  --port 8000
```

Expose port 8000 with Cloudflare Quick Tunnel, then enter the generated HTTPS URL in the React UI. The WebSocket endpoint is `/ws/stream`; configure `LPR_ALLOWED_ORIGINS` for the Vercel origin and keep `LPR_DEMO_TOKEN` outside Git.

The complete protocol, runbook, supported claim, evaluation protocol, and feature branch order are in [docs/realtime-demo-spec.md](docs/realtime-demo-spec.md).

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
- [OCR character dataset selection](docs/ocr-dataset-selection.md)
- [Architecture and implementation](docs/architecture-and-implementation.md)
- [Dataset runtime integration](docs/dataset-runtime-integration.md)
- [Project journal and final-report guide](docs/project-journal.md)
- [Colab/Kaggle training notebook](notebooks/train_pipeline.ipynb)

## Current limitations

- The Roboflow dataset has not been downloaded or trained in this repository yet.
- No model accuracy metrics are claimed until a real training/evaluation run is completed.
- The base OpenCV headless dependency and Ultralytics' transitive OpenCV dependency should be tested together before production packaging.
