# Delivery and Merge Order

## Target branch

`develop` is the integration branch. It is not the production/default branch.

The current feature branch is:

```text
feature/dataset-roboflow
```

It contains the full local implementation history plus the lazy Roboflow dataset feature and documentation.

## Recommended stacked order

The cleanest history is to seed `develop` from the project setup commit, then merge from the oldest dependency to the newest feature:

```text
feature/project-setup              → develop
feature/plate-preprocessing        → develop
feature/yolo-detector              → feature/plate-preprocessing
feature/ocr-pipeline               → feature/yolo-detector
feature/recognition-pipeline       → feature/ocr-pipeline
feature/evaluation-cli             → feature/recognition-pipeline
fix/yolo-detector-bounds           → feature/evaluation-cli
fix/code-quality-review            → fix/yolo-detector-bounds
feature/dataset-roboflow           → fix/code-quality-review
```

Merge PRs in the same top-to-bottom order shown above. Each child PR should target its immediate parent branch until the parent is merged. After each parent merges into `develop`, update the next PR base if the GitHub workflow requires it.

## Why not target every PR directly at develop?

Every PR can technically target `develop`, but stacked branches then contain overlapping commits. GitHub diffs become noisy and the same changes appear repeatedly. Parent-targeted PRs keep each review focused and preserve the dependency order.

If the team requires every PR to have `develop` as its base, merge the branches bottom-up and expect duplicate diffs/conflict resolution. Do not merge a child before its parent.

## Current delivery blocker

The repository must have:

- A writable GitHub credential for the contributor.
- A remote `develop` branch or permission to create it.
- The SSH remote configured as `git@github.com:cuongmn2011/CITD_ImageProcessing.git`.

No claim of a pushed branch or PR URL should be added until GitHub confirms the remote ref and PR creation succeeds.
