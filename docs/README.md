# Project Documentation

This directory contains the project evidence and operating documentation. Source code and tests remain the authority for executable behavior; these documents explain decisions, workflow, provenance, and current evidence.

## Documents

1. [Research and dataset selection](research-dataset-selection.md)
   - Problem scope and evaluation criteria.
   - Comparison of the three researched dataset candidates.
   - Selection rationale for the Cuong Ta Roboflow dataset.
   - License and provenance risks.

2. [Architecture and implementation](architecture-and-implementation.md)
   - End-to-end image/video recognition flow.
   - Detector, preprocessing, OCR, metrics, and CLI responsibilities.
   - Runtime dataset preparation boundary.

3. [Dataset runtime integration](dataset-runtime-integration.md)
   - Roboflow download contract.
   - Cache, manifest, validation, and training behavior.
   - Commands and failure handling.

4. [Project journal and final-report guide](project-journal.md)
   - Chronological implementation record.
   - Verification evidence.
   - Final report structure and remaining measurements.

5. [Delivery and merge order](delivery-and-merge-order.md)
   - `develop` target branch.
   - Stacked PR dependency order.
   - GitHub delivery prerequisites.

6. [Colab/Kaggle training notebook](../notebooks/train_pipeline.ipynb)
   - Thin orchestration layer for clone, install, dataset preparation, training, inference, and optional OCR evaluation.

7. [Code review and training report](code-review-and-training-report.md)
   - Full source scan, static/security checks, test and coverage evidence.
   - Findings ranked by severity with source evidence and recommendations.
   - Observed YOLO training configuration and validation metrics.

8. [Academic final report](final-report-academic.md)
   - Detailed academic structure from problem statement to conclusion.
   - Project architecture, training flow, inference flow, and sequence diagrams.
   - Methodology, equations, observed training results, limitations, and future work.

## Evidence policy

- Claims about behavior must point to source files, tests, or command output.
- Training metrics must be added only after an actual dataset/model run.
- Dataset files, API keys, model weights, videos, and generated runs stay outside Git.
