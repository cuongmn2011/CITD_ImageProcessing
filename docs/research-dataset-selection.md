# Research: Dataset Selection

## Objective

Select a license-plate dataset that fits the current pipeline:

```text
YOLO plate detection → crop/preprocessing → OCR
```

The immediate training target is the detector. The OCR layer currently supports Tesseract, EasyOCR, and PaddleOCR as interchangeable backends rather than a project-owned OCR model.

## Evaluation criteria

| Criterion | Why it matters |
|---|---|
| Vietnamese plates | Reduces domain mismatch in plate layout, typography, and motorcycle/car proportions. |
| Detection annotations | Required by the current YOLO training script. |
| Dataset size and diversity | Improves robustness across angle, blur, lighting, and plate type. |
| Export format | Reduces conversion risk and preparation time. |
| License/provenance | Required for an academic report and any future redistribution. |
| Reproducibility | Versioned source and deterministic local preparation are preferred. |

## Candidate 1 — Vietnamese Car License Plate Dataset, Cuong Ta / Roboflow

Source: [Roboflow Universe](https://universe.roboflow.com/cuong-ta-ulxex/vietnamese-car-license-plate)

Search-indexed dataset metadata reports approximately **8,255 Vietnamese images**, one object-detection class (`plate`), and a Public Domain listing. The dataset is directly aligned with the current detector and can be exported in a YOLO-compatible format.

### Strengths

- Vietnamese domain.
- Largest of the researched Vietnam-specific detection candidates.
- One-class plate detection matches `PlateDetection` and `configs/plate-dataset.yaml`.
- Roboflow export makes YOLO preparation reproducible.
- Public Domain listing is more permissive than the alternatives reviewed.

### Limitations

- Public metadata does not establish a character-level OCR annotation set.
- The exact export version must remain pinned in the runtime dataset spec.
- Actual image counts and split counts must be recorded after download, not inferred only from the web listing.

## Candidate 2 — Vietnam License Plate Segment Dataset, Kaggle

Sources:

- [Kaggle dataset page](https://www.kaggle.com/datasets/duydieunguyen/licenseplates)
- [Kaggle metadata API](https://www.kaggle.com/api/v1/datasets/view/duydieunguyen/licenseplates)

The published metadata describes approximately **5,135 images**: 3,510 one-line plates and 1,625 two-line plates. It uses polygon/corner annotations and reports data collected from internet and real environments under different weather, time, and shooting angles. The metadata reports approximately 70% training and 30% evaluation data.

### Strengths

- Vietnamese domain.
- Explicit one-line/two-line distinction.
- Polygon labels are useful for perspective correction and plate geometry analysis.
- Large enough to support a detector experiment.

### Limitations

- Kaggle metadata currently reports the license as **Unknown**.
- Polygon labels require conversion to YOLO boxes or a deliberate segmentation workflow.
- The current project has not implemented a Kaggle-specific downloader.

## Candidate 3 — Vietnamese Plate Dataset, Mì AI / community distribution

Sources:

- [Dataset documentation](https://github.com/winter2897/Real-time-Auto-License-Plate-Recognition-with-Jetson-Nano/blob/main/doc/dataset.md)
- [Reference implementation using the dataset](https://github.com/trungdinh22/License-Plate-Recognition)

The documentation describes two separate datasets: a plate-detection set and a character-recognition set. Both VOC/PASCAL and YOLO downloads are documented. The reference implementation explicitly uses the two sets for the two stages of Vietnamese license-plate recognition.

### Strengths

- Best conceptual fit for training both detector and character detector.
- Supports one-line and two-line Vietnamese plates.
- YOLO and VOC formats are already documented.
- Useful for a future project-owned OCR model.

### Limitations

- Distributed through Google Drive/community repositories.
- Public documentation does not provide a clear formal license or complete provenance statement.
- Official image/label counts are not clearly stated in the documentation.
- Download integrity and duplicate checks would be required before use.

## Decision

Selected dataset: **Candidate 1 — Cuong Ta / Roboflow**.

Reason: the current project first needs a reliable YOLO plate detector. Candidate 1 has the closest task match, the largest reported Vietnamese detection collection among the shortlisted candidates, an easy YOLO export path, and the clearest permissive license listing.

The dataset is integrated lazily. No images are committed. The default runtime spec is:

```text
cuong-ta-ulxex/vietnamese-car-license-plate/1
```

The version is configurable through `--dataset` or `LPR_ROBOFLOW_DATASET` so the experiment can be reproduced if a later Roboflow version is selected.

## Research caveats

- Roboflow Universe pages can require browser/API access, so the repository records the dataset slug and runtime version rather than vendoring the export.
- License claims must be rechecked against the dataset page at the time of publication.
- Final dataset counts, split counts, image dimensions, label distribution, and duplicate rate belong in the final experiment report after the first real download.
