# CITD_ImageProcessing

Vietnamese license-plate detection and recognition (ALPR) pipeline.

## Pipeline

```
Input image/video → YOLO plate detector → crop + perspective/preprocessing variants
  → OCR backend (Tesseract / EasyOCR / PaddleOCR) → annotated output / OCR metrics
```

## Stack

- Python 3.12, dependency/build managed with `uv` + hatchling.
- CLI entry point: `lpr` (`src/lpr/cli.py`) — `lpr infer-image`, `lpr infer-video`, `lpr evaluate-ocr`, `lpr serve`.
- Frontend: `web/` — React + TypeScript + Vite, realtime WebSocket demo.
- Training: YOLO via Ultralytics, dataset from Roboflow.

## Repo layout

- `src/lpr/dataset.py` — lazy Roboflow dataset download/validation/cache manifest. Dataset files are never committed; cache lives in ignored `data/processed/`.
- `src/lpr/detector.py` — YOLO (Ultralytics) adapter, normalized `PlateDetection` records.
- `src/lpr/preprocessing.py` — crop padding, perspective rectification, grayscale/otsu/adaptive/clahe variants.
- `src/lpr/ocr.py` — Tesseract/EasyOCR/PaddleOCR adapters, candidate selection across backend + preprocessing variant.
- `src/lpr/pipeline.py` — end-to-end image/video inference orchestration.
- `src/lpr/metrics.py` — OCR exact accuracy, character accuracy, CER (edit-distance based).
- `src/lpr/server.py` / `src/lpr/stream.py` — FastAPI + WebSocket realtime streaming API (`/ws/stream`) for the browser demo; GPU inference typically hosted on Colab, exposed via Cloudflare Quick Tunnel.
- `web/src/App.tsx`, `web/src/hooks/useFrameSender.ts`, `web/src/hooks/usePlateSocket.ts`, `web/src/components/OverlayCanvas.tsx` — realtime demo: video stays local in the browser, sampled JPEG frames are sent to the GPU API over WebSocket, detection/OCR overlay is drawn locally client-side (does not wait for a fully annotated video).
- `scripts/prepare-dataset.py`, `scripts/train-yolo.py`, `scripts/bootstrap-kaggle.py`, `scripts/package-run.py` — training/dataset scripts.
- `notebooks/train_pipeline.ipynb`, `notebooks/inference_pipeline.ipynb` — used on Colab/Kaggle for GPU training/serving.
- `docs/` — architecture-and-implementation.md, dataset-runtime-integration.md, realtime-demo-spec.md (protocol/runbook for the realtime demo), research-dataset-selection.md, project-journal.md (engineering log + final-report outline), code-review-and-training-report.md, final-report-academic.md, delivery-and-merge-order.md.

## Dataset & training

Training dataset: Vietnamese Car License Plate Dataset by Cuong Ta on Roboflow (`cuong-ta-ulxex/vietnamese-car-license-plate/1`), one-class (`plate`) detector. Requires a Roboflow API key (kept out of Git). No model weights, dataset files, videos, or training-run artifacts are committed — they are gitignored and downloaded/generated on demand.

## Evidence discipline (important)

Detector validation metrics from a real training run are recorded (mAP50=0.99495, mAP50-95=0.72907 on artifact `model/citd-yolo11s-training.zip` / `model/.lpr-model/best.pt`), but that artifact contains bounding boxes only — no plate-text transcripts — so retraining it **cannot** improve OCR accuracy. No OCR exact accuracy / character accuracy / CER / end-to-end throughput / backend comparison has been measured yet on a held-out set.

Do not add invented or placeholder metrics to the README, journal, or final report — only record real command output and generated metric files after an experiment is actually run (see `docs/project-journal.md` section 6 for the protocol). To improve OCR accuracy, the correct levers are backend choice (PaddleOCR/EasyOCR), crop quality/preprocessing variant, and temporal voting — not detector retraining.

## Verification

```bash
uv run pytest
uv run ruff check src tests scripts
uv run python -m compileall -q src scripts
uv lock --check
```
