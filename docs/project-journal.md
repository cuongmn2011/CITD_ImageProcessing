# Project Journal and Final-Report Guide

This document records the engineering process and identifies evidence still required for the final semester report.

## 1. Problem definition

The project targets Vietnamese license-plate recognition from images and videos. The pipeline must detect plates, normalize the crop, run OCR, annotate outputs, and provide measurable OCR metrics.

Initial implementation scope:

- YOLO plate detection.
- Plate crop and perspective/preprocessing variants.
- Pluggable Tesseract, EasyOCR, and PaddleOCR backends.
- Image/video inference.
- OCR CSV evaluation.
- Reproducible dataset preparation.

## 2. Research process

The dataset research compared three candidates using Vietnamese-domain fit, annotation quality, size/diversity, export format, license/provenance, and reproducibility.

Selected source:

- Vietnamese Car License Plate Dataset — Cuong Ta / Roboflow.
- Reason: strongest immediate fit for the current YOLO detector, reported large image count, simple one-class detection task, and permissive public listing.

Rejected or deferred alternatives:

- Kaggle Vietnam License Plate Segment Dataset: useful polygon annotations, but license metadata is Unknown.
- Mì AI Vietnamese Plate Dataset: strong detector/character-recognition fit, but public provenance and license details are incomplete.

Full research evidence is in [research-dataset-selection.md](research-dataset-selection.md).

## 3. Implementation sequence

### Foundation

- Created the Python package and `src/lpr` module layout.
- Added project configuration, dependency locking, test configuration, and ignored data/model artifacts.
- Added the YOLO dataset configuration contract.

### Detection and preprocessing

- Added YOLO detector adapter and normalized `PlateDetection` records.
- Added outward bounding-box rounding to avoid losing border pixels.
- Added crop padding and bounds checks.
- Added perspective rectification and robust convex-quad ordering.
- Added shared resize for multiple OCR preprocessing variants.
- Added support for grayscale, BGR, and BGRA inputs.

### OCR and pipeline

- Added Tesseract, EasyOCR, and PaddleOCR adapters.
- Added candidate selection across backend and preprocessing variants.
- Added video resource guards, frame limits, FPS fallback, output-path safety, and annotation.
- Added OCR edit distance, exact accuracy, character accuracy, and CER.

### Dataset integration

- Added lazy Roboflow download through `src/lpr/dataset.py`.
- Added explicit preparation command: `scripts/prepare-dataset.py`.
- Added automatic preparation to `scripts/train-yolo.py`.
- Added export validation and cache manifest.
- Kept all dataset artifacts outside Git.

## 4. Verification evidence

Latest local verification is maintained in [code-review-and-training-report.md](code-review-and-training-report.md):

- 45 tests passed.
- `ruff check src tests scripts`: passed.
- `python -m compileall -q src scripts`: passed.
- `uv lock --check`: passed.
- Notebook code cells compile successfully without executing hosted-runtime cells.
- Training archive integrity check passed.
- `best.pt` loads as an Ultralytics detection model with one `plate` class.
- Detector validation metrics are recorded from the real `run/results.csv` artifact.
- No OCR/end-to-end metric is inferred from detector metrics.


## 5. What is not yet claimed

The training artifact now provides detector validation evidence. The following claims still require additional measured experiments:

- Number of usable images after validation.
- Train/validation/test counts and class distribution.
- Independent test-split detector metrics.
- OCR exact accuracy, character accuracy, and CER on a held-out set.
- End-to-end frame throughput.
- Comparison between OCR backends.
- Error analysis by one-line/two-line plate, blur, angle, weather, and lighting.

Do not add invented values to the final report. Record command output and generated metric files after the experiment is run. The current measured detector results and review findings are in [code-review-and-training-report.md](code-review-and-training-report.md).

## 6. Recommended final experiment protocol

1. Set `ROBOFLOW_API_KEY` outside Git.
2. Run `scripts/prepare-dataset.py` with the pinned dataset version.
3. Save the printed manifest and validated counts.
4. Train YOLO with a fixed seed and documented hyperparameters.
5. Evaluate detector metrics on the held-out split.
6. Run each available OCR backend on the same crop set.
7. Evaluate OCR from a CSV with `ground_truth,prediction`.
8. Compare preprocessing variants and backend latency.
9. Run end-to-end video inference on a fixed sample.
10. Add only observed results to the final report.

## 7. Final-report outline

1. Introduction and motivation.
2. Problem statement and requirements.
3. Dataset research and selection.
4. Dataset preparation and annotation format.
5. System architecture.
6. Detection model and training configuration.
7. Preprocessing and OCR design.
8. Evaluation methodology.
9. Quantitative results.
10. Qualitative error analysis.
11. Performance and limitations.
12. Reproducibility and ethical/license considerations.
13. Conclusion and future work.

## 8. Delivery evidence

The dataset feature is currently committed on the local branch `feature/dataset-roboflow`. GitHub push and PR creation must be recorded separately after write access to the target repository is available.
