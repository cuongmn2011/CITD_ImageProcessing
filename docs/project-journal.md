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

### End-to-end smoke test: preprocessing variant (2026-09-25)

Scope is deliberately small; this is a direction signal, **not** an accuracy claim.

- Input: a 90-frame slideshow video (10 fps, 15 frames per photo) built from 6 photos drawn at
  random from the detector dataset's `valid/` split: 1 one-line car plate, 4 two-line motorbike
  plates, 1 military plate. Parking/gate-camera images, not street footage.
- Ground truth: read manually from each photo.
- Models: YOLO `outputs/citd-yolo11s-training.zip` (`best.pt`), fine-tuned PaddleOCR
  `PP-OCRv5_mobile_rec` exported to `model/ocr-rec-training-merged/inference`.
- Run: `scripts/pipeline-demo.py` video mode (`RealtimePlateRecognizer`, ByteTrack tracking,
  3-vote stabilization), stride 1, imgsz 640, local CPU. Scored with
  `lpr.video_report.score_against_ground_truth`.

| Variant | Correct / 6 | Wrong | Missed | Exact accuracy | CER |
|---|---:|---:|---:|---:|---:|
| `otsu` (current default) | 1 | 3 | 2 | 0.167 | 0.429 |
| `raw` (colour crop) | 4 | 2 | 0 | 0.667 | 0.041 |

- With `otsu`, two-line motorbike plates had the letter read as a digit (`60F1` → `6051`) or
  were not read at all; `raw` read three of the four correctly. The remaining `raw` errors:
  `59P223103` → `599223103` (P read as 9) and military `KT6073` → `KT4073`.
- Throughput: 90 frames in about 55 s on local CPU (≈0.6 s/frame).
- Conclusion for now: Otsu binarization, not only the two-line layout, drove the motorbike
  errors. Project defaults stay unchanged until this is confirmed on a real street-camera
  video with a hand-labelled plate list.

### Street video: one reading per vehicle (2026-09-26)

Scope: the first 30 s (900 frames, 1920×1080, 30 fps) of one user-supplied street video
(`outputs/video-jobs/<job>/input.mp4`, not committed). No hand-labelled plate list exists yet,
so there is **no accuracy figure** here. Three plates were checked by eye on enlarged frames:
`24A07816`, `29K10425`, `24C09238`. Run: `lpr.video_report.process_video`, variant `raw`,
stride 1, imgsz 640, local CPU (i7-1185G7, no CUDA).

- **Fragmentation.** ByteTrack returned 52 tracks for about 6 vehicles; the report used to show
  one row per track. Small, fast plates move so far between frames that their boxes do not
  overlap, so ByteTrack never confirms a track and returns no box. About every 6th source
  frame repeats the previous one; only there does the plate stand still, and a track is
  confirmed once, under a new id. Fragments of those vehicles start exactly on repeated frames.
- **Skipping repeated frames was tried and reverted.** It cut tracks from 52 to 16 but lost the
  motorbike `24X1 124.42` entirely and left the `24C09238` pickup with 4 frames: the repeated
  frames were the only frames in which ByteTrack returned those plates.
- **Merge + vote.** `merge_fragments` joins fragments within 2 s whose readings differ by at most
  2 edits (fewer for short readings), and picks each vehicle's reading by a vote over every OCR
  read of all its fragments. Fixed along the way: once a track had locked in, a stray read that
  lost the vote was still shown (`24A07816` drawn as `22A07816`).

| Rule for the vehicle's reading | Rows | `24A07816` | `29K10425` | `24C09238` |
|---|---:|---|---|---|
| none (one row per track) | 52 | 10 rows | 2 rows | 7 rows |
| merge, locked-in fragment first | 18 | correct | correct | `24G09238` (wrong) |
| merge, vote over all reads | 20 | correct (15 of 16 reads) | correct (7 of 14 reads) | correct (6 of 8 reads) |

The last row also uses the mobile text detector (below), which changes individual reads. The
remaining rows are mostly unread fragments and 1–3 character noise.

**Where the time goes.** 60 frames with vehicles, measured while another inference process
shared the CPU, so absolute times are inflated; the ratios are comparable.

| Stage | Time |
|---|---:|
| YOLO + ByteTrack | 314 ms / frame |
| PaddleOCR, server text detector + fine-tuned rec | 675 ms / read |
| Writing the annotated copy, VP8 WebM 1920×1080 | 182 ms / frame |
| Drawing boxes, live-view JPEG | 9 ms / frame |

| PaddleOCR text detector (18 crops, 6 labelled slideshow plates × 3 frames) | Time / read | Exact | CER |
|---|---:|---:|---:|
| `PP-OCRv5_server_det` (PaddleOCR default) | 667 ms | 6/18 | 0.082 |
| `PP-OCRv5_mobile_det` | 339 ms | 7/18 | 0.075 |

- oneDNN (`enable_mkldnn=True`) fails with both detectors: `NotImplementedError ...
  ConvertPirAttribute2RuntimeAttribute`. Recognition alone on the whole crop, without the text
  detector, returned 1–5 character strings on street crops and is not usable.
- VP8 encoding per frame: 1920×1080 208 ms, 1280×720 107 ms, 960×540 61 ms.
- Adopted in `scripts/pipeline-demo.py` only: `--ocr-det-model PP-OCRv5_mobile_det` (default) and
  an annotated copy capped at 1280 px wide. `lpr` CLI/server defaults are unchanged; 18 crops is
  a direction signal, not an accuracy claim.
- Whole 30 s run: 391 s before these two changes, 343 s after; the demo server was also running
  during the start of the second run, so the gain is not measured cleanly.

### Street video: centroid tracker instead of ByteTrack (2026-09-26)

Same 30 s clip and settings as above (mobile text detector, output capped at 1280 px).

- **Why ByteTrack dropped plates.** Ultralytics' ByteTrack multiplies box overlap by detection
  confidence (`fuse_score: True`) and confirms a new track only when that reaches 0.3. Raw YOLO
  boxes for the `24X1 124.42` motorbike, frames 122-144: overlap with the previous frame
  0.35-0.54 at confidence 0.47-0.72, product 0.17-0.29 on every normal frame; the product
  passed only on repeated frames (overlap 1.0), which is exactly where its fragments started.
- **Plate motion in plate widths**, same frames: the same plate moved 0.1-0.8 widths per frame
  (the most right after a repeated frame); plates of different vehicles were 5+ widths apart.
  `lpr.tracking.CentroidTracker` matches a track's velocity-predicted centre to detections
  within 1.5 widths and at most 2x size change, and ids a new plate on its first frame.
- **Reads per vehicle.** The report now keeps reading every 3 frames after a track locks in, and
  a tied vote goes to the more confident reading instead of the one shown longer.

| Run (30 s, CPU) | Tracks | Rows | Time | `24A07816` | `29K10425` | `24C09238` | Motorbike `24X1 124.42` |
|---|---:|---:|---:|---|---|---|---|
| ByteTrack, stride 1 | 52 | 20 | 343 s | 15 of 16 reads | 7 of 14 | 6 of 8 | 3 frames, read `24X112452` |
| ByteTrack, stride 3 | 4 | 3 | 126 s | missed | 5 of 8 | missed | missed |
| Centroid, stride 1 | 19 | 15 | 305 s | 34 of 35 | 25 of 35 | 8 of 15 | 51 frames, read `24X112442` |
| Centroid, stride 3 | 20 | 13 | 136 s | 33 of 33 | 29 of 34 | **7 of 15, read `24G09238`** | 14 frames, read `24X112452` |

- Both ByteTrack rows still read only every 12 frames after lock-in. At centroid stride 1 the
  extra reads cost nothing measurable: 306 s with the old interval, 305 s with the new one.
- The `24C09238` pickup is a coin flip for this OCR model: its reads split about evenly between
  `24C09238` and `24G09238` in both centroid runs. Not a tracking problem.
- **One read is right about 43% of the time on street plates**: 39 of 91 crops of the three
  plates above (mean crop height 46 px), against 8 of 18 on the labelled slideshow (120 px).
  Correct plates come from voting over many reads. Crops are upscaled to 64 px high before OCR;
  reading them at detected size instead gave the same 39/91 (CER 0.168 vs 0.170) and 7/18 vs
  8/18, so preprocessing was left unchanged.
- Adopted in `scripts/pipeline-demo.py` only: `--tracker centroid` (default; `bytetrack` still
  available) and reads every 3 frames in video mode. `lpr serve` still uses ByteTrack.

### Street video: YOLO on OpenVINO instead of PyTorch (2026-09-26)

Same 30 s clip; centroid tracker, mobile OCR text detector, reads every 3 frames after
lock-in, output capped at 1280 px (all as above). CPU: i7-1185G7, no CUDA/GPU. Export:
`scripts/export-yolo-openvino.py` (`model.export(format="openvino")` on the existing
`best.pt` - converts the already-trained weights to a different runtime, no retraining).

- **Detection-only speed** (60 street frames, confidence 0.4 both): PyTorch 155-214 ms/frame
  (varied run to run on this shared machine) vs OpenVINO 49-77 ms/frame - consistently
  **3-4x faster** per frame.
- **OpenVINO's confidence scores run low.** Matched via IoU ≥ 0.5 against PyTorch at the same
  confidence=0.4: of 94 matched detections the mean confidence gap was 0.020 (median 0.016,
  max 0.072) - small. But 17 PyTorch detections (confidence 0.41-0.535, i.e. near the 0.4
  cutoff) had no OpenVINO match at all: at the threshold, that small a gap is enough to drop
  some. Lowering OpenVINO's confidence to 0.35 recovered most of them (17 → 6 missing) at the
  cost of 5 new low-confidence extra detections.
- **Full pipeline, same clip, PyTorch conf 0.4 vs OpenVINO conf 0.35:**

  | Run | Time | Vehicles | Motorbike `24X1 124.42` tracked | Other 4 plates |
  |---|---:|---:|---|---|
  | stride 3, PyTorch | 165 s | 13 | 14 frames, `24X112452` (1 wrong char) | all correct |
  | stride 3, OpenVINO | 138 s | 14 | 22 frames, `24X112442` (correct) | all correct |
  | stride 1, PyTorch | 305 s | 17 | 71 frames (2 fragments), correct | all correct |
  | stride 1, OpenVINO | **173 s** | 18 | 68 frames (2 fragments), correct | all correct |

  At stride 1 - the accuracy-first setting - OpenVINO is **43% faster end to end** (173 s vs
  305 s) with matching tracking/reading outcomes; at stride 3 the gain is smaller (17%)
  because PaddleOCR, not YOLO, dominates the per-frame cost once a plate is already being
  read. `24C09238` stayed a near-even split between `24C09238`/`24G09238` in every run here -
  confirmed again as an OCR-model limit, not a detector or tracker effect.
- **Not adopted as a default.** `resolve_model_path` (`src/lpr/server.py`) now also accepts an
  exported-model directory, and `scripts/pipeline-demo.py --model`'s help documents pairing it
  with a lower `--confidence` (e.g. 0.35). The export only exists after a user runs the export
  script locally (gitignored under `outputs/`), so the script's *default* model path could not
  point at it without breaking a fresh checkout; this is one clip's worth of evidence, not
  grounds to also change the CLI's or `lpr serve`'s defaults.

### Docker packaging attempt, reverted (2026-09-26)

Goal: run `scripts/pipeline-demo.py` identically from a container on Windows or macOS hosts.
Added, then removed, `Dockerfile`, `docker-compose.yml`, `.dockerignore`.

- `docker build` completed successfully: all dependencies (torch, paddlepaddle, ultralytics,
  fastapi, dev tools) installed and the image was tagged. Took about 27 minutes on this
  machine, dominated by downloading/installing torch and paddlepaddle.
- **Never verified further.** Mid-session the host's C: drive hit 8.9 GB free out of 399 GB;
  `AppData\Local\Docker` (Docker Desktop's own VM disk) alone was 178 GB, accumulated from
  unrelated past projects, not this one. Docker Desktop's backend then stopped responding and
  its named pipe disappeared - it had crashed. Starting the container and reaching the page
  over HTTP was never tried.
- Root cause is host disk space, not the Dockerfile; the image itself built cleanly. Given
  Docker Desktop was down and the fix (freeing host disk, possibly compacting its 178 GB VM
  disk) is outside this project, the Docker files were removed rather than left unverified in
  the repo. `uv run python scripts/pipeline-demo.py` (or `docker compose` again from this
  entry, if someone re-adds the same three files) remains the way to run it.


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
