# Research: OCR Character Dataset Selection

## Objective

Select a dataset that can fine-tune the OCR *recognition* stage on Vietnamese plate crops,
distinct from the detector dataset in [research-dataset-selection.md](research-dataset-selection.md).
No dataset with `crop_path,ground_truth` transcripts exists in this project; the OCR layer has
been inference-only against pretrained Tesseract/EasyOCR/PaddleOCR backends (see
`docs/project-journal.md`, "What is not yet claimed").

## Evaluation criteria

Same bar as the detector dataset: Vietnamese-plate relevance, annotation usefulness for the
target task (here: per-character labels convertible to a transcript), size, export format,
license/provenance, and reproducibility.

## Candidate — Vietnam-License-Plate-Recognition, Roboflow

Source: [Roboflow Universe](https://universe.roboflow.com/dataset-format-conversion-iidaz/vietnam-license-plate-recognition)
(workspace `dataset-format-conversion-iidaz`, project `vietnam-license-plate-recognition`).

Confirmed directly on the project page (2026-09-16): **200 images**, **1 dataset version**, and
**22 object-detection classes**: digits `0`-`9` plus the Latin letters
`A, B, C, D, E, F, G, H, M, T, V, X` used on Vietnamese plates — matching
`DEFAULT_OCR_DATASET_SPEC` (`.../1`) and `configs/ocr-character-dataset.yaml` exactly. Annotations
are per-character YOLO bounding boxes on plate-crop images, not transcript strings — there is no
ready-made `crop_path,ground_truth` pair.

### Strengths

- Vietnamese domain, license explicitly CC BY 4.0 (not "Unknown", the bar this project already
  enforces for the detector dataset).
- Character-level boxes cover the digit/letter set actually used on VN plates.
- YOLO export is directly reusable by `validate_yolo_export` (`src/lpr/dataset.py`) — that
  function validates train/val image+label directory shape and is not hardcoded to one class.

### Limitations

- Ground-truth plate text must be reconstructed, not read directly: character boxes are grouped
  into lines by vertical position and sorted left-to-right within each line
  (`src/lpr/ocr_dataset.py:group_lines`/`boxes_to_text`), which assumes characters on the same
  physical line share a similar center-y and lines don't visually overlap. This heuristic has not
  been validated against real exported images yet.
- Pinned version and class list were confirmed manually against the live Roboflow project page
  (200 images, 1 version, 22 classes matching exactly). License text itself still needs a final
  visual confirmation on the page before publication (see caveats below).
- Only 200 images total across the whole project (train+val+test); this is small for a
  recognition fine-tune and the resulting sample count after character-to-text conversion should
  be recorded and judged for adequacy once the real conversion runs.
- No character-recognition-specific accuracy claim exists yet; only the detector dataset has a
  measured training run.

## Candidate 2 — PBL4_Deep-Learning Vietnamese plates, GitHub

Source: [lephamcong/PBL4_Deep-Learning](https://github.com/lephamcong/PBL4_Deep-Learning),
path `Dataset/BiensoxeVietNam/train`.

The repo's README states the Vietnamese set was "compiled through real-world photography" and,
after quality selection and augmentation, totals **13,320 images** (12,320 train + 1,000 eval).
Ground truth is encoded directly in each filename, for example `29A87180_1212_0.jpg` → plate text
`29A87180` — no separate annotation file is needed, and no line-clustering heuristic is required
(unlike Candidate 1).

### Strengths

- ~66x more images than Candidate 1, which meaningfully improves recognition fine-tuning.
- Ground truth is trivial to extract (split filename on the first `_`), reusing
  `lpr.ocr.normalize_text` for cleanup — no per-character box reconstruction needed.
- Public and fetchable without an API key via a sparse `git clone` of just the `train` subdir.

### Limitations

- **No LICENSE file and no explicit usage-rights statement anywhere in the repo.** This is the
  same "Unknown license" condition that this project's own `docs/research-dataset-selection.md`
  already downgraded a detector-dataset candidate for.
- It is a student PBL (problem-based learning) project repo, not a dataset published with formal
  provenance or a citable release.
- The default fetch ref (`main`) is a moving branch, not an immutable version like a Roboflow
  export version; `ensure_git_dataset` (`src/lpr/dataset.py`) resolves and records the exact
  commit SHA actually cloned (via `git rev-parse HEAD`) into `.git-dataset-manifest.json` so the
  real snapshot used is always known, but the SHA is not pinned in code ahead of time.

## Decision

**Combine both sources.** Candidate 1 (Roboflow, CC BY 4.0, confirmed) remains the primary,
cleanly-licensed source. Candidate 2 (GitHub, unlicensed) is used as a volume-boosting
supplement — restricted to academic/non-commercial use in this project, not redistributed as a
derived dataset, and never presented as license-clear in the final report. `OcrSample.source`
(`src/lpr/ocr_dataset.py`) tags every sample as `"roboflow-chars"` or `"github-filename"` so the
final report can state exactly how many samples came from each source.

Defaults, both overridable:

```text
Roboflow: dataset-format-conversion-iidaz/vietnam-license-plate-recognition/1
GitHub:   lephamcong/PBL4_Deep-Learning @ main, subdir Dataset/BiensoxeVietNam/train
```

configurable through `--dataset`/`LPR_OCR_ROBOFLOW_DATASET` and `--github-repo`/`--github-ref`/
`--github-subdir` (or `--no-github-supplement` to use Roboflow alone), mirroring the existing
`LPR_ROBOFLOW_DATASET` convention for the detector dataset.

## Research caveats

- Re-verify Candidate 1's license against the live Roboflow project page before the first real
  download; update this document and `DEFAULT_OCR_DATASET_SPEC` if it has changed.
- Record the actual resolved commit SHA for Candidate 2 (from `scripts/prepare-ocr-dataset.py`'s
  JSON output / `.git-dataset-manifest.json`) after the first real fetch, and consider pinning
  `DEFAULT_OCR_SUPPLEMENT_REF` to that SHA instead of `main` for long-term reproducibility.
- Record actual per-source and total sample counts, train/val split sizes, and the line-clustering
  `y_threshold` used after the first real conversion run — do not add invented numbers to this
  document or the final report before that run happens (same discipline as
  `docs/project-journal.md`).
- If the line-clustering heuristic misgroups two-line plates on real data, treat that as a data
  quality issue to fix in `src/lpr/ocr_dataset.py`, not a reason to silently drop those samples.
- If a formal, license-clear alternative to Candidate 2 is found later, prefer it and drop the
  GitHub supplement rather than keeping an unlicensed source out of convenience.
