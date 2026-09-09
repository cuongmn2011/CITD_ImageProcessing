# Báo cáo scan, check, review source và huấn luyện

## 1. Mục tiêu và phạm vi

Báo cáo này ghi lại kết quả kiểm tra snapshot hiện có của repository `CITD_ImageProcessing`, gồm:

- toàn bộ Python source trong `src/lpr/`;
- các CLI/script trong `scripts/`;
- test suite trong `tests/`;
- hai notebook train/inference trong `notebooks/`;
- cấu hình, dependency lock, tài liệu vận hành;
- training artifact `outputs/citd-yolo11s-training.zip`.

Source và test là nguồn sự thật cho hành vi thực thi. Báo cáo này là evidence snapshot: các số liệu train, coverage, test và finding có thể thay đổi khi dataset, dependency hoặc artifact thay đổi.

## 2. Tóm tắt điều hành

### Kết luận

- Pipeline hiện có cấu trúc rõ: YOLO detector → crop/preprocessing → OCR backend → annotation/metrics.
- Codebase vượt qua các kiểm tra tĩnh và test hiện có.
- Không phát hiện secret có định dạng độ tin cậy cao, pattern injection phổ biến, file `.env`/model/data bị Git track, hoặc vulnerability dependency đã biết trong lần quét này.
- Đã có training artifact thật. Detector đã train đủ 50 epoch và model `best.pt` load thành công bằng Ultralytics.
- Chất lượng detector trên **validation split** tốt ở IoU 0.50 nhưng giảm đáng kể ở mAP50-95. Đây chưa phải bằng chứng cho chất lượng OCR hoặc end-to-end ALPR.
- Các điểm cần xử lý trước khi dùng làm kết quả học thuật/prod: validator label còn lỏng, pipeline tính thừa preprocessing variants, metadata chưa đủ tái lập, packaging có edge case tự đưa archive vào chính nó, và hosted bootstrap phụ thuộc mạnh vào môi trường có sẵn.

### Phân loại tổng quan

| Nhóm | Kết quả |
|---|---:|
| Python files được Git track | 23 |
| Tổng dòng Python | 2.121 |
| Core executable statements được coverage | 720 |
| Coverage hiện tại | 74% |
| Tests | 45 passed |
| Ruff | Passed |
| Compileall | Passed |
| `uv lock --check` | Passed |
| Notebook code cells kiểm tra cú pháp | 19/19 passed |
| Training epochs trong artifact | 50 |
| Secret pattern độ tin cậy cao | Không phát hiện |
| Code vulnerability pattern phổ biến | Không phát hiện |
| Dependency known vulnerabilities | Không phát hiện trong `uvx pip-audit` |

## 3. Bằng chứng và cách kiểm tra

Các lệnh đã chạy từ repository root:

```bash
uv run pytest
uv run pytest --cov=src/lpr --cov-report=term-missing
uv run ruff check src tests scripts
uv run python -m compileall -q src scripts
uv lock --check
uv run lpr --help
uv run python scripts/train-yolo.py --help
uv run python scripts/prepare-dataset.py --help
uv run python scripts/package-run.py --help
uvx --from pip-audit pip-audit --format json
unzip -t outputs/citd-yolo11s-training.zip
```

Smoke verification bổ sung:

- Notebook được parse và compile từng code cell, không thực thi các cell cần secret/dataset/video.
- `best.pt` trong archive được load bằng Ultralytics với `task=detect`, class `{0: "plate"}`, kích thước 19.177.178 bytes.
- Archive có 29 members và `unzip -t` không báo lỗi.
- Validator hiện tại được kiểm tra với một label cố ý sai (`class_id=99`, tọa độ ngoài khoảng chuẩn); validator vẫn chấp nhận. Đây là bằng chứng cho finding F-01, không phải lỗi của test suite.
- `package_run()` được kiểm tra khi output nằm bên trong `run_dir`; archive tạo ra member `run/artifact.zip`, tức tự đưa file đang được ghi vào archive.

## 4. Kiến trúc và phạm vi source

| Khu vực | Vai trò | Evidence |
|---|---|---|
| `src/lpr/detector.py` | Chuẩn hóa kết quả Ultralytics, clamp bbox, crop | `YoloPlateDetector`, `detections_from_result` |
| `src/lpr/preprocessing.py` | Rectification, crop, resize, 5 variants | `generate_variants`, `preprocess_plate` |
| `src/lpr/ocr.py` | Tesseract/EasyOCR/PaddleOCR adapters, normalize và format check | `OCRResult`, các `*Backend` |
| `src/lpr/pipeline.py` | Ghép detector/OCR, image/video loop, annotation | `LicensePlateRecognizer` |
| `src/lpr/metrics.py` | Levenshtein, exact accuracy, character accuracy, CER | `evaluate_ocr_pairs` |
| `src/lpr/dataset.py` | Roboflow download, cache, YAML normalization, manifest | `ensure_roboflow_dataset` |
| `src/lpr/cli.py` | CLI inference và OCR evaluation | `build_parser`, `main` |
| `scripts/train-yolo.py` | Chuẩn bị dataset và gọi Ultralytics train | `main` |
| `scripts/bootstrap-kaggle.py` | Cài dependency cho hosted runtime | `main`, `_ensure_packages` |
| `scripts/package-run.py` | Đóng gói run và weights | `package_run` |

### Luồng runtime

1. `scripts/train-yolo.py` gọi `ensure_roboflow_dataset()` trừ khi có `--no-download-dataset`.
2. Dataset được download vào temporary directory, validate rồi move vào cache.
3. Ultralytics đọc `data.yaml` và train detector một class `plate`.
4. Inference nạp model, detect plate, tạo variants, gọi từng OCR backend, chọn candidate hợp regex và confidence cao nhất.
5. Video loop xử lý từng frame và ghi MP4 annotated.

## 5. Training artifact và số liệu train

### 5.1 Cấu hình quan sát được

Nguồn: `outputs/citd-yolo11s-training.zip`, members `metadata.json` và `run/args.yaml`.

| Tham số | Giá trị quan sát được |
|---|---|
| Dataset spec | `cuong-ta-ulxex/vietnamese-car-license-plate/1` |
| Model | `yolo11s.pt` |
| Task | Detection |
| Classes | 1 (`plate`) |
| Epochs | 50 |
| Image size | 640 |
| Batch | `-1` (Ultralytics auto batch) |
| Device | `auto` trong metadata; `device: ''` trong args YAML |
| Workers | 2 |
| Seed | 0 trong `run/args.yaml` |
| Deterministic | `true` trong `run/args.yaml` |
| Validation split | `val` |
| AMP | `true` |
| Optimizer | `auto` |
| Artifact model | `best.pt`, 19.177.178 bytes |
| Training time ghi trong results | 4.728,94 giây ≈ 78,82 phút ≈ 1,314 giờ |

Lưu ý: seed/deterministic xuất hiện trong Ultralytics run arguments, nhưng CLI `scripts/train-yolo.py` chưa expose rõ các tham số này. Muốn tái lập nghiêm ngặt nên truyền chúng từ command/config và ghi thêm commit/dependency fingerprint.

### 5.2 Kết quả theo epoch đầu/cuối

Nguồn: `outputs/citd-yolo11s-training.zip:run/results.csv`. Các metric detector là metric trên validation split của Ultralytics.

| Epoch | Precision | Recall | mAP50 | mAP50-95 | Train box | Train cls | Train dfl |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0,88006 | 0,95203 | 0,94891 | 0,62567 | 1,26153 | 1,35033 | 1,14126 |
| 50 | 0,99447 | 0,99542 | 0,99495 | 0,72907 | 0,97004 | 0,34340 | 1,00403 |

Epoch đạt mAP50-95 cao nhất trong file là epoch 50. Vì epoch 50 cũng là row cuối, không có bằng chứng về early stopping hoặc checkpoint tốt hơn ở epoch giữa.

### 5.3 Diễn giải

- Precision `0,99447`: trong các detection được chọn, tỷ lệ detection đúng rất cao theo ngưỡng đánh giá của Ultralytics.
- Recall `0,99542`: phần lớn plate ground-truth trong validation được phát hiện.
- mAP50 `0,99495`: detector tốt ở ngưỡng IoU 0,50.
- mAP50-95 `0,72907`: khi yêu cầu localization chặt hơn qua các IoU từ 0,50 đến 0,95, chất lượng giảm rõ. Khoảng cách `0,99495 - 0,72907 = 0,26588` là tín hiệu cần kiểm tra bounding-box precision, plate nhỏ, góc nghiêng và domain shift.
- Đây là **detector validation result**, không phải OCR exact accuracy, character accuracy, CER hay end-to-end frame accuracy.
- Artifact không chứa dataset export và không có bảng đếm train/val/test, duplicate count, class distribution hoặc image dimension distribution. Không được thay các số liệu thiếu này bằng con số approximate trên trang dataset.

### 5.4 Số liệu chưa có và không được suy diễn

Chưa có evidence runtime cho:

- số image thực tế sau khi download và validation;
- số label hợp lệ theo từng split;
- phân bố kích thước ảnh và class;
- duplicate/corrupt-file count;
- kết quả trên test split độc lập;
- OCR exact accuracy, character accuracy và CER;
- so sánh Tesseract/EasyOCR/PaddleOCR;
- FPS, latency/frame và tổng thời gian video;
- accuracy theo one-line/two-line, blur, lighting, angle, weather;
- end-to-end plate recognition rate.

## 6. Findings review code

### F-01 — High: Dataset validator chưa validate nội dung label

**Evidence:** `src/lpr/dataset.py:163-170` chỉ kiểm tra thư mục `labels` tồn tại và có ít nhất một `.txt`. Không kiểm tra:

- label có cùng stem với image;
- mỗi dòng có đúng 5 trường;
- `class_id` nằm trong `names`;
- tọa độ normalized nằm trong `[0, 1]`;
- width/height dương;
- image có decode được;
- label không rỗng hoặc không chứa NaN/Infinity.

**Đã tái hiện:** export có image bytes giả và label `99 2.0 -1.0 4.0 4.0` vẫn được `validate_yolo_export()` chấp nhận.

**Impact:** Một cache hỏng có thể đi vào training hoặc làm metric không đáng tin. Với báo cáo học thuật, đây là rủi ro trực tiếp đối với validity của dataset và kết quả.

**Khuyến nghị:** Thêm validator YOLO theo từng file, kiểm tra pairing, parse numeric, class range, bounds, duplicate stem và decode ảnh; manifest ghi số lượng theo split/class sau validation.

### F-02 — Medium: Pipeline tính tất cả preprocessing variants dù chỉ dùng một phần

**Evidence:** `src/lpr/pipeline.py:61-68` gọi `generate_variants(crop)`, còn `src/lpr/preprocessing.py:178-181` luôn tạo cả `raw`, `gray`, `otsu`, `adaptive`, `clahe`. Sau đó pipeline chỉ đọc các variant được cấu hình.

**Impact:** Khi CLI dùng mặc định `otsu,clahe`, ba variant không dùng vẫn tốn resize/memory; video dài sẽ chịu overhead cho mỗi detected plate/frame. Điều này khớp với risk latency đã ghi trong handoff nhưng chưa có benchmark FPS.

**Khuyến nghị:** Thêm API `generate_variants(image, variants=...)` hoặc tạo lazy theo variant; benchmark cùng input/video trước và sau.

### F-03 — Medium: Training metadata chưa đủ để tái lập nghiêm ngặt

**Evidence:**

- `scripts/train-yolo.py:17-77` không nhận `seed`, `deterministic`, `project`, `name` hoặc config augmentation làm tham số rõ ràng.
- `scripts/package-run.py:48-67` chỉ ghi run path, tên weights, SHA-256 của best model và danh sách file.
- Notebook ghi dataset/epoch/image-size/batch/device tại `notebooks/train_pipeline.ipynb` nhưng không ghi Git SHA, dataset manifest/counts/checksum, Python/Ultralytics/PyTorch versions hoặc GPU model.

**Impact:** Có thể biết model đã train thế nào ở mức cơ bản nhưng khó tái tạo đúng artifact hoặc audit data provenance sau khi môi trường thay đổi.

**Khuyến nghị:** Ghi `git rev-parse HEAD`, lockfile hash, package versions, GPU/driver, dataset manifest hash, split counts, label checksum, seed, deterministic, command line và UTC timestamp vào metadata của run.

### F-04 — Medium: `package_run` tự đưa output vào archive nếu output nằm trong run directory

**Evidence:** `scripts/package-run.py:44-60` tạo output trước khi duyệt `run_dir.rglob("*")`, nhưng không loại trừ output. Smoke edge case tạo `run/artifact.zip` và archive chứa member `run/artifact.zip`.

**Impact:** Archive có thể chứa chính file đang được ghi, gây self-inclusion, artifact phình to hoặc nội dung member không phản ánh archive hoàn chỉnh. Default output hiện nằm ngoài `runs/detect`, nên lỗi chỉ xuất hiện khi user chọn path nguy hiểm.

**Khuyến nghị:** Reject `output` nằm trong `run_dir`, hoặc filter bằng resolved path trước khi `archive.write()`; thêm regression test cho cùng thư mục và path trùng.

### F-05 — Medium: Hosted bootstrap giả định dependency transitive đã có sẵn

**Evidence:** `scripts/bootstrap-kaggle.py:27-45` cài package với `--no-deps`; `VISION_PACKAGES` chỉ bổ sung `ultralytics` và `thop`. `_ensure_roboflow()` cũng cài direct requirements với `--no-deps`.

**Impact:** Runtime phụ thuộc vào image Kaggle/Colab cụ thể. Nếu thiếu một dependency transitive của Ultralytics/OCR, import có thể fail; module verification không bảo đảm mọi code path train/inference chạy được.

**Khuyến nghị:** Tạo compatibility matrix cho hosted image, kiểm tra import/function smoke cho từng mode, và chỉ dùng `--no-deps` khi danh sách dependency đã được pin/kiểm chứng cho image đó.

### F-06 — Medium: Hai OpenCV distributions cùng tồn tại

**Evidence:** `pyproject.toml:6-10` khai báo `opencv-python-headless`; dependency tree của `ultralytics` kéo thêm `opencv-python`. Runtime import hiện trả về OpenCV `5.0.0` từ `.venv`.

**Impact:** Hai distribution cùng cung cấp module `cv2`, có thể gây khác biệt package resolution hoặc hành vi codec giữa local và hosted environment. README đã ghi đây là limitation nhưng chưa có compatibility test production.

**Khuyến nghị:** Chọn một OpenCV provider phù hợp deployment, kiểm tra lockfile và test `cv2.imread`, `VideoCapture`, `VideoWriter` trên target runtime.

### F-07 — Low: CLI chưa validate confidence trước khi khởi tạo detector

**Evidence:** `src/lpr/cli.py:79` nhận `--confidence` bằng `float`, trong khi `YoloPlateDetector.__init__` yêu cầu `(0, 1]` tại `src/lpr/detector.py:61-64`. Giá trị `0`, NaN hoặc Infinity có thể qua argparse rồi fail ở runtime bằng `ValueError`.

**Impact:** Error message kém thân thiện và input invalid không bị chặn ở boundary CLI.

**Khuyến nghị:** Dùng validator tương tự `_unit_interval`, nhưng loại `0` nếu giữ contract detector hiện tại; bổ sung test boundary.

### F-08 — Low: Tài liệu verification đã stale

**Evidence:** `docs/project-journal.md` từng ghi `36 tests passed`, trong khi lần chạy hiện tại là `45 passed`; tài liệu cũ cũng nói chưa có training run dù repository hiện có archive và `results.csv`.

**Impact:** Người đọc báo cáo có thể dùng sai trạng thái dự án hoặc bỏ qua training evidence đã tồn tại.

**Khuyến nghị:** Đã cập nhật journal để trỏ tới báo cáo này và phân biệt detector metrics với OCR/end-to-end metrics. Các số liệu thay đổi về sau phải cập nhật từ command output/artifact thật.

## 7. Security review

### Kết quả không phát hiện

- Không có match AWS/GitHub/Stripe/Slack/Google/Anthropic key, private key hoặc JWT trong source/config/script/notebook đã quét.
- Không có match SQL injection, XSS DOM sink, command injection pattern, `eval`, `new Function`, TLS verification disabled hoặc hardcoded credential trong phạm vi source.
- Không có `.env`, data, model, run hoặc output artifact được Git track theo `git ls-files`.
- `ROBOFLOW_API_KEY` được đọc từ environment/secret store; manifest không ghi secret.
- `uvx --from pip-audit pip-audit --format json` trả về `No known vulnerabilities found` cho environment được audit.

### Giới hạn của kết luận security

Đây là static scan và dependency advisory scan, không phải penetration test. Không kết luận được runtime isolation, supply-chain trust của dataset, quyền Kaggle/GitHub, hay safety của file media độc hại. Dataset và model từ bên ngoài vẫn cần được coi là untrusted input.

## 8. Test quality và coverage

45 test hiện tại bao phủ tốt các pure utility và guard chính:

- bbox clamp/rounding;
- preprocessing geometry và channel handling;
- OCR normalization/format filtering;
- Levenshtein và OCR CSV metrics;
- dataset cache/manifest/force path;
- CLI parsing;
- package archive cơ bản.

Coverage đo được là 74% trên 720 statements. Các vùng còn thiếu đáng chú ý:

- detector thật và Ultralytics inference;
- Tesseract/EasyOCR/PaddleOCR runtime thật;
- video `VideoCapture`/`VideoWriter` và lỗi codec;
- dataset download/error path với SDK thật;
- dataset label semantic validation vì validator chưa có;
- hosted bootstrap train/inference trên image thật;
- notebook execution với secret, model, dataset, video.

Test pass không đồng nghĩa hệ thống đã được đánh giá end-to-end. Cần thêm smoke/integration trên artifact và một fixture media nhỏ trước khi claim production readiness.

## 9. Kế hoạch ưu tiên

### P0 — trước khi chốt số liệu học thuật

1. Validate semantic toàn bộ YOLO labels và image-label pairing.
2. Ghi dataset counts, class distribution, image dimensions, corrupt/duplicate counts và dataset hash.
3. Tách rõ validation metrics và test metrics; không dùng mAP validation làm test result.
4. Tạo OCR ground-truth CSV và chạy cùng crop set cho từng backend/variant.

### P1 — trước khi chạy video dài hoặc đóng gói release

1. Chỉ tạo variants được yêu cầu; đo FPS và latency/frame.
2. Chặn output archive nằm trong run directory.
3. Ghi đầy đủ provenance: Git SHA, lock hash, versions, GPU, seed, command, dataset manifest.
4. Chốt một OpenCV distribution và kiểm thử codec ở target runtime.
5. Smoke test model + OCR + video trên hosted runtime trước khi chạy clip dài.

### P2 — cải thiện maintainability

1. Validate `--confidence` ngay tại argparse.
2. Bổ sung integration tests cho optional OCR backends khi dependency có mặt.
3. Cập nhật report/journal từ artifact generator thay vì sửa số thủ công.
4. Cân nhắc thêm temporal aggregation/tracking nếu mục tiêu là video ALPR ổn định, không chỉ frame-by-frame OCR.

## 10. Kết luận dùng cho báo cáo học thuật

Có thể kết luận có evidence rằng hệ thống đã triển khai được pipeline detection-to-OCR, test suite hiện tại pass, và một YOLO11s detector đã train 50 epoch trên dataset Roboflow version 1 với validation precision `0,99447`, recall `0,99542`, mAP50 `0,99495`, mAP50-95 `0,72907`.

Chưa thể kết luận hệ thống nhận dạng biển số end-to-end đạt accuracy tương ứng. Lý do: artifact hiện chỉ chứng minh detector validation metrics; chưa có OCR ground truth, CER, exact accuracy, test split độc lập, dataset counts sau validation, hay benchmark video. Các phần này phải được đo thực tế và thêm vào báo cáo sau khi chạy experiment tương ứng; không được suy diễn từ mAP detector.

## 11. Evidence index

- `README.md`: contract, commands, limitations.
- `docs/architecture-and-implementation.md`: module boundaries và runtime flow.
- `docs/dataset-runtime-integration.md`: dataset preparation và reproducibility checklist.
- `docs/project-journal.md`: lịch sử triển khai và trạng thái evidence.
- `src/lpr/dataset.py`: dataset validation/cache.
- `src/lpr/pipeline.py`, `src/lpr/preprocessing.py`: inference và preprocessing.
- `scripts/train-yolo.py`, `notebooks/train_pipeline.ipynb`: training orchestration.
- `outputs/citd-yolo11s-training.zip:metadata.json`.
- `outputs/citd-yolo11s-training.zip:run/args.yaml`.
- `outputs/citd-yolo11s-training.zip:run/results.csv`.
