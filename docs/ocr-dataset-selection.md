# Research: OCR Character Dataset Selection

## Objective

Select a dataset that can fine-tune the OCR *recognition* stage on Vietnamese plate crops,
distinct from the detector dataset in [research-dataset-selection.md](research-dataset-selection.md).
No dataset with `crop_path,ground_truth` transcripts exists in this project; the OCR layer has
been inference-only against pretrained Tesseract/EasyOCR/PaddleOCR backends (see
`docs/project-journal.md`, "What is not yet claimed").

## Evaluation criteria

Same bar as the detector dataset: Vietnamese-plate relevance, annotation usefulness for the
target task (a `crop_path,ground_truth` transcript pair), size, export format,
license/provenance, and reproducibility.

## Candidate 1 — Vietnam-License-Plate-Recognition, Roboflow (rejected)

Source: [Roboflow Universe](https://universe.roboflow.com/dataset-format-conversion-iidaz/vietnam-license-plate-recognition)
(workspace `dataset-format-conversion-iidaz`, project `vietnam-license-plate-recognition`).

Confirmed directly on the project page (2026-09-16): **200 images**, **1 dataset version**, and
**22 object-detection classes**: digits `0`-`9` plus the Latin letters
`A, B, C, D, E, F, G, H, M, T, V, X` used on Vietnamese plates. Annotations are per-character
YOLO bounding boxes on plate-crop images, not transcript strings, so ground-truth text would need
reconstruction (line-clustering + left-to-right sort) before it could feed a recognition trainer.

### Strengths

- Vietnamese domain, license explicitly CC BY 4.0 (not "Unknown").
- Character-level boxes cover the digit/letter set actually used on VN plates.

### Rejected because

- **Only 200 images total** across the whole project (train+val+test). That is too small to be
  worth a recognition fine-tune, and disproportionate to the conversion complexity it would
  require (per-character box parsing + line-clustering heuristic for two-line plates) versus
  Candidate 2's much larger, already-transcript-labeled set. Not used in the current pipeline.

## Candidate 2 — PBL4_Deep-Learning Vietnamese plates, GitHub (selected)

Source: [lephamcong/PBL4_Deep-Learning](https://github.com/lephamcong/PBL4_Deep-Learning),
path `Dataset/BiensoxeVietNam/train`.

The repo's README states the Vietnamese set was "compiled through real-world photography" and,
after quality selection and augmentation, totals **13,320 images** (12,320 train + 1,000 eval).
Ground truth is encoded directly in each filename, for example `29A87180_1212_0.jpg` → plate text
`29A87180` — no separate annotation file or line-clustering heuristic is needed;
`parse_filename_labeled_samples` (`src/lpr/ocr_dataset.py`) reads the label straight from the
filename and normalizes it with `lpr.ocr.normalize_text`.

### Strengths

- ~66x more images than Candidate 1, which meaningfully improves recognition fine-tuning.
- Ground truth is trivial and reliable to extract (split filename on the first `_`).
- Public and fetchable without an API key via a sparse `git clone` of just the `train` subdir
  (`ensure_git_dataset` in `src/lpr/dataset.py`), so no Roboflow account/API key is needed at all
  for the OCR training flow.

### Limitations

- **No LICENSE file and no explicit usage-rights statement anywhere in the repo.** This is the
  same "Unknown license" condition that this project's own `docs/research-dataset-selection.md`
  already downgraded a detector-dataset candidate for. Use is restricted to this academic
  project — not redistributed as a derived dataset, and never presented as license-clear in the
  final report.
- It is a student PBL (problem-based learning) project repo, not a dataset published with formal
  provenance or a citable release.
- The default fetch ref (`main`) is a moving branch, not an immutable version like a Roboflow
  export version; `ensure_git_dataset` resolves and records the exact commit SHA actually cloned
  (via `git rev-parse HEAD`) into `.git-dataset-manifest.json`, so the real snapshot used is
  always known even though the SHA is not pinned in code ahead of time.

## Candidate 3 — LPRNet (NguyenHuuThDat), GitHub `valid/` subdir (selected, additive)

Source: [NguyenHuuThDat/LPRNet](https://github.com/NguyenHuuThDat/LPRNet), path
`Dataset/BiensoxeVietNam/valid`, commit `7e7c3982c6ce1e9dea51b3eea85f632e09d7b91c`.

This repo's `Dataset/BiensoxeVietNam/train/` (12,320 files) is **byte-for-byte identical** to
Candidate 2's `train/` -- every filename matches, confirmed by diffing the full listing of both.
Both are the same underlying PBL4 coursework dataset; fetching this repo's `train/` would add
nothing. Its `valid/` subdir, however, was never fetched by this project before (Candidate 2 only
pulls `train/`) and contains **1,000 images / 136 unique plate texts**, all 136 with **zero
overlap** against the 421 plate texts already in use from Candidate 2 (confirmed by diffing the
label sets extracted from filenames).

### Strengths

- Real, verified diversity gain: +136 plate identities (~32% over the existing 421), for free --
  no new photography, no API key, already the same filename-label convention as Candidate 2 so
  `parse_filename_labeled_samples` reads it unchanged.

### Limitations

- Same as Candidate 2: **no LICENSE file, no usage-rights statement** in the repo. Academic use
  only, same restriction already applied to Candidate 2.
- Small in absolute terms (1,000 images) and, since it comes from the same collection effort as
  Candidate 2, may share similar photography conditions/camera -- it is not an independent data
  source, just an unused slice of a related one.

## Decision

**Candidate 2 (GitHub, `train/`) + Candidate 3 (GitHub, `valid/`), merged.** Candidate 1
(Roboflow) is rejected for being too small to justify its conversion complexity. Also evaluated
and rejected as an OCR source: `cuong-ta-ulxex/vietnamese-car-license-plate` (the Roboflow
project already used for the plate *detector*, see `docs/research-dataset-selection.md`) --
confirmed via a manual YOLOv11 export (`nc: 1`, `names: ['plate']`) to be whole-plate bounding
boxes only, no character/text labels at all, so it cannot feed OCR training regardless of image
count (matches the evidence-discipline note in `CLAUDE.md`).

`scripts/prepare-ocr-dataset.py` fetches Candidate 2's `train/` and Candidate 3's `valid/` and
merges their samples before the train/val split (by default; `--no-extra-source` restricts it to
Candidate 2 only).

Defaults, overridable via `--repo`/`--ref`/`--subdir` and `--extra-repo`/`--extra-ref`/`--extra-subdir`:

```text
lephamcong/PBL4_Deep-Learning @ 054cc054cdc2b24305f3675a9c14832459c8f9db,
subdir Dataset/BiensoxeVietNam/train

NguyenHuuThDat/LPRNet @ 7e7c3982c6ce1e9dea51b3eea85f632e09d7b91c,
subdir Dataset/BiensoxeVietNam/valid
```

## Real fetch results (2026-09-16)

`scripts/prepare-ocr-dataset.py` was run once against the `main` branch, which resolved to commit
`054cc054cdc2b24305f3675a9c14832459c8f9db`. That commit is now pinned as
`DEFAULT_OCR_DATASET_REF` in `src/lpr/dataset.py` instead of the moving `main` branch, for
reproducibility.

- Filename-labeled samples extracted from `Dataset/BiensoxeVietNam/train`: **12,320** (matches
  the README's stated train-split count exactly).

### Merged run with Candidate 3 (2026-09-16)

`scripts/prepare-ocr-dataset.py --out data/processed/ocr-rec-dataset-merged` (both sources, code
defaults, `val_ratio=0.1`):

- Total samples merged: **13,320** (12,320 + 1,000).
- `train.txt`: **12,110** lines, **501** unique plate texts.
- `val.txt`: **1,210** lines, **56** unique plate texts.
- 0 plate texts shared between `train.txt` and `val.txt` (verified by diffing the label sets).
- Unique plate texts overall: **557**, up from 421 before Candidate 3 (+136, exactly the count
  predicted from Candidate 3's label diff above).

## Research caveats

- If GitHub's default branch moves forward, `DEFAULT_OCR_DATASET_REF` will still resolve to the
  exact pinned commit above (via `git fetch origin <sha>`) rather than silently drifting; update
  the pin deliberately (and this section) if a newer snapshot is intentionally adopted.
- Record actual train/val split sizes after the first real conversion run — do not add invented
  numbers to this document or the final report before that run happens (same discipline as
  `docs/project-journal.md`).
- If a formal, license-clear alternative to Candidate 2 is found later, prefer it and drop this
  GitHub source rather than keeping an unlicensed source out of convenience.
