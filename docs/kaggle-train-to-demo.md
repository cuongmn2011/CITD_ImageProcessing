# Kaggle: train mới đến demo

Runbook cho một vòng làm việc mới hoàn toàn:

```text
Roboflow dataset
    ↓
train_pipeline.ipynb
    ↓
best.pt + training archive
    ↓
Kaggle Dataset chứa model
    ↓
inference_pipeline.ipynb
    ↓
video/image annotated demo
```

Notebook training là [`notebooks/train_pipeline.ipynb`](../notebooks/train_pipeline.ipynb). Notebook demo là [`notebooks/inference_pipeline.ipynb`](../notebooks/inference_pipeline.ipynb). Code thực thi vẫn nằm trong [`scripts/train-yolo.py`](../scripts/train-yolo.py), [`scripts/package-run.py`](../scripts/package-run.py) và CLI `lpr`.

## 0. Quy ước cho lần train mới

- Dùng **Kaggle Notebook mới** hoặc xóa toàn bộ workspace cũ trước khi chạy.
- Không dùng `best.pt`, `last.pt` hoặc run cũ làm checkpoint đầu vào.
- Pipeline hiện tại khởi tạo từ `yolo11s.pt` pretrained và chạy một training run mới; đây là fresh run, không phải resume từ model cũ.
- Dataset mặc định là `cuong-ta-ulxex/vietnamese-car-license-plate/1`.
- Model, video, dataset cache và output sinh ra không commit vào Git.
- Sau mỗi lần train, tạo một Kaggle Dataset/version mới cho model để tránh dùng nhầm artifact cũ.

## 1. Chuẩn bị Roboflow Secret trên Kaggle

1. Mở Kaggle và tạo Notebook mới.
2. Bật **Internet** trong Notebook settings.
3. Chọn accelerator **GPU**, thông thường là T4.
4. Vào **Add-ons → Secrets → Add a new secret**.
5. Tạo secret:

```text
Name: ROBOFLOW_API_KEY
Value: <Roboflow private API key>
```

Notebook đọc secret qua `kaggle_secrets.UserSecretsClient`. Không viết key trực tiếp vào notebook và không in giá trị key ra output.

## 2. Lấy notebook mới nhất từ repository

Sau khi PR đã merge vào `develop`, tải file sau từ repository:

```text
notebooks/train_pipeline.ipynb
```

Trong Kaggle chọn **File → Import Notebook** rồi upload file này. Không sử dụng snapshot notebook cũ đã tải từ một Kaggle session trước đó.

Notebook mặc định clone:

```text
https://github.com/cuongmn2011/CITD_ImageProcessing.git
branch: develop
```

Nếu đang thử một branch chưa merge, đặt `LPR_REPO_REF` trong cell bootstrap trước khi chạy các cell còn lại.

## 3. Reset workspace cho fresh run

Nếu đây là Kaggle Notebook mới thì bỏ qua bước này. Nếu tái sử dụng một session, chạy cell riêng **trước khi chạy notebook training**:

```python
from pathlib import Path
import shutil

for target in (
    Path("/kaggle/working/CITD_ImageProcessing"),
    Path("/kaggle/working/runs"),
    Path("/kaggle/working/outputs"),
    Path("/kaggle/working/citd-yolo11s-training"),
    Path("/kaggle/working/citd-yolo11s-training.zip"),
):
    if target.is_dir():
        shutil.rmtree(target)
    elif target.exists():
        target.unlink()
```

Mục đích là loại bỏ code clone, dataset cache, run directory và artifact cũ. Cell này chỉ nên chạy trong workspace Kaggle riêng cho pipeline này.

## 4. Cấu hình training

Trong `train_pipeline.ipynb`, cell cấu hình nằm sau cell load secret. Giá trị bắt đầu khuyến nghị:

```text
EPOCHS = 50
IMAGE_SIZE = 640
BATCH_SIZE = -1
DEVICE = auto
```

Ý nghĩa:

- `EPOCHS`: số epoch.
- `IMAGE_SIZE`: kích thước ảnh YOLO.
- `BATCH_SIZE = -1`: để Ultralytics tự ước lượng batch theo GPU.
- `DEVICE = auto`: để Ultralytics tự chọn GPU. Có thể dùng `0` nếu muốn chỉ định GPU đầu tiên.

Nếu muốn một vòng train dài hơn, đổi `EPOCHS` trong cell cấu hình trước khi chạy cell train. Không đổi sang `best.pt` cũ làm tham số `--model`.

## 5. Chạy notebook training theo thứ tự

Chạy tuần tự các cell trong `train_pipeline.ipynb`.

### 5.1 Bootstrap repository và dependency

Cell đầu tiên sẽ:

- chọn `/kaggle/working` làm workspace;
- clone branch `develop`;
- cài `uv` và `ipywidgets`.

Cell dependency tiếp theo chạy tương đương:

```bash
uv sync --extra vision --extra dataset --extra ocr
```

### 5.2 Load secret và chuẩn bị dataset

Notebook gọi `scripts/prepare-dataset.py` với secret từ Kaggle. Dataset được tải vào cache runtime, không lưu vào Git.

Dataset spec mặc định:

```text
cuong-ta-ulxex/vietnamese-car-license-plate/1
```

Nếu dataset preparation lỗi:

1. kiểm tra Internet đã bật;
2. kiểm tra secret đúng tên `ROBOFLOW_API_KEY`;
3. kiểm tra key còn quyền truy cập dataset;
4. chạy lại cell preparation, không chạy train khi dataset chưa chuẩn bị thành công.

### 5.3 Train YOLO

Notebook gọi `scripts/train-yolo.py`. Lệnh tương đương là:

```bash
uv run --extra vision --extra dataset python scripts/train-yolo.py \
  --dataset cuong-ta-ulxex/vietnamese-car-license-plate/1 \
  --model yolo11s.pt \
  --epochs 50 \
  --imgsz 640 \
  --batch -1
```

Nếu chỉ định GPU:

```bash
  --device 0
```

Run hoàn tất khi output có `best.pt` dưới một thư mục dạng:

```text
/kaggle/working/CITD_ImageProcessing/runs/detect/train*/weights/best.pt
```

Notebook chọn run completed mới nhất có `weights/best.pt`, thay vì giả định tên cố định là `train` hoặc `train-2`.

### 5.4 Kiểm tra metrics

Cell metrics đọc `results.csv` của chính `RUN_DIR` vừa chọn và in:

- epoch;
- precision;
- recall;
- mAP50;
- mAP50-95;
- dòng tốt nhất theo mAP50-95;
- dòng cuối cùng.

Ghi lại các thông tin sau cho báo cáo/demo:

```text
repository revision
training run path
model path
dataset spec/version
epochs
image size
batch/device
best metrics
final metrics
```

Không lấy metrics từ notebook output cũ hoặc từ một thư mục run khác.

## 6. Đóng gói và tải artifact về máy local

Cell packaging của notebook tạo:

```text
/kaggle/working/citd-yolo11s-training/
/kaggle/working/citd-yolo11s-training.zip
```

Archive có cấu trúc tương đương:

```text
best.pt
last.pt
metadata.json
run/
  weights/
  results.csv
  args.yaml
  results.png
  confusion_matrix.png
  ...
```

Tải file sau từ Kaggle:

```text
citd-yolo11s-training.zip
```

Cell cũng tạo link trực tiếp cho `best.pt`; tải file này nếu muốn upload model nhỏ gọn lên Kaggle Dataset.

## 7. Cập nhật toàn bộ artifact local

Giả sử file tải về nằm ở `~/Downloads/citd-yolo11s-training.zip` và current directory là root repository:

```bash
cd /path/to/CITD_ImageProcessing

rm -rf /tmp/citd-yolo11s-training
mkdir -p /tmp/citd-yolo11s-training
unzip -q ~/Downloads/citd-yolo11s-training.zip \
  -d /tmp/citd-yolo11s-training

mkdir -p models outputs runs/detect
cp /tmp/citd-yolo11s-training/best.pt models/best.pt
cp /tmp/citd-yolo11s-training/last.pt models/last.pt
cp ~/Downloads/citd-yolo11s-training.zip \
  outputs/citd-yolo11s-training.zip

rm -rf runs/detect/train-fresh
mkdir -p runs/detect/train-fresh
cp -a /tmp/citd-yolo11s-training/run/. runs/detect/train-fresh/
```

`models/*.pt`, `outputs/` và `runs/` được giữ ngoài Git. Không copy thư mục `run/` vào root repository vì đây là generated artifact; nếu cần giữ metrics local, đặt nó dưới `runs/detect/` như lệnh trên.

Kiểm tra model local có thể load được:

```bash
uv sync --extra vision
uv run python -c 'from ultralytics import YOLO; YOLO("models/best.pt"); print("best.pt: load ok")'
uv run python -c 'from ultralytics import YOLO; YOLO("models/last.pt"); print("last.pt: load ok")'
```

Đóng gói lại bằng script repository nếu cần:

```bash
uv run python scripts/package-run.py \
  --run-dir runs/detect/train-fresh \
  --output outputs/citd-yolo11s-training.zip
```

## 8. Tạo Kaggle Dataset cho model mới

Tạo một Kaggle Dataset mới hoặc tạo version mới cho dataset model hiện tại.

Khuyến nghị upload tối thiểu:

```text
best.pt
```

Có thể upload thêm:

```text
citd-yolo11s-training.zip
```

Nếu upload archive, `inference_pipeline.ipynb` sẽ tự tìm và giải nén member có tên `best.pt`.

Đặt tên dataset có version rõ ràng, ví dụ:

```text
citd-yolo11s-model
```

Trong phần description ghi dataset version, epoch/config và SHA-256 từ `metadata.json`. Không dùng dataset model cũ nếu mục tiêu là demo model vừa train.

## 9. Chuẩn bị video/image demo

Có hai cách:

### Cách A — Kaggle Dataset

Tạo hoặc chọn một Kaggle Dataset chứa:

```text
*.mp4
*.jpg
*.jpeg
*.png
```

### Cách B — upload trực tiếp vào Notebook

Dùng nút upload của Kaggle hoặc đặt file trong `/kaggle/working`. Khi dùng trực tiếp, nhập path thực tế vào widget, ví dụ:

```text
/kaggle/input/citd-demo-video/demo.mp4
```

Không đặt video vào Git repository.

## 10. Chạy notebook inference/demo

Tải và mở:

```text
notebooks/inference_pipeline.ipynb
```

Trong Kaggle:

1. tạo Notebook mới;
2. bật GPU nếu muốn detector chạy trên GPU;
3. dùng **Add Input** để thêm model Dataset và video/image Dataset;
4. chạy các cell theo thứ tự.

Notebook inference chỉ cài:

```bash
uv sync --extra vision --extra ocr
```

Notebook này không cần `ROBOFLOW_API_KEY`, không download dataset và không chạy `train-yolo.py`.

### 10.1 Kiểm tra model được chọn

Cell model discovery sẽ in:

```text
Model: /kaggle/input/.../best.pt
Model size: ... MB
```

Nếu model Dataset chứa nhiều file, đặt biến môi trường trước cell discovery hoặc nhập path trong code:

```python
import os
os.environ["LPR_MODEL_PATH"] = "/kaggle/input/<dataset-name>/best.pt"
```

### 10.2 Chạy video demo

Trong widget:

| Trường | Giá trị khuyến nghị |
|---|---|
| Video | path tới `.mp4` |
| Output | `/kaggle/working/annotated.mp4` |
| OCR | `tesseract` |
| Device | `0` trên Kaggle GPU, hoặc `auto` |
| Max frames | `0` để chạy toàn bộ video; số dương để smoke test |

Bấm **Run video inference**. Kết quả gồm:

```text
/kaggle/working/annotated.mp4
```

Notebook hiển thị video inline và tạo link download.

Nếu muốn test nhanh trước khi chạy toàn bộ video, đặt `Max frames = 300`. Khi kết quả hợp lệ, đổi về `0` để chạy toàn bộ video.

Lệnh CLI tương đương:

```bash
uv run lpr infer-video \
  --input /kaggle/input/<video-dataset>/demo.mp4 \
  --output /kaggle/working/annotated.mp4 \
  --model /kaggle/input/<model-dataset>/best.pt \
  --ocr tesseract \
  --device 0
```

### 10.3 Chạy image demo

Nhập path image vào widget và bấm **Run image inference**.

Lệnh CLI tương đương:

```bash
uv run lpr infer-image \
  --image /kaggle/input/<image-dataset>/car.jpg \
  --model /kaggle/input/<model-dataset>/best.pt \
  --ocr tesseract \
  --variants otsu,clahe \
  --device 0
```

Output JSON chứa:

```text
bbox
detection_confidence
text
ocr_confidence
ocr_backend
preprocessing
```

## 11. Checklist demo cuối cùng

Trước khi kết luận demo thành công, kiểm tra:

- [ ] Notebook training dùng code từ `develop` mới nhất.
- [ ] Kaggle Secret có tên chính xác `ROBOFLOW_API_KEY`.
- [ ] Dataset preparation hoàn tất.
- [ ] Training chạy từ `yolo11s.pt`, không resume model cũ.
- [ ] `RUN_DIR` là run mới nhất và có `weights/best.pt`.
- [ ] Metrics được đọc từ `RUN_DIR/results.csv` tương ứng.
- [ ] `best.pt` đã được tải xuống và load thành công local.
- [ ] Kaggle model Dataset trỏ tới artifact mới.
- [ ] Inference notebook dùng đúng model path.
- [ ] Smoke test vài frame chạy thành công.
- [ ] Full video tạo được `annotated.mp4`.
- [ ] Có ít nhất một image prediction để kiểm tra bbox/OCR.
- [ ] Video annotated đã được tải về nếu cần trình diễn offline.

## 12. Xử lý lỗi thường gặp

### `ROBOFLOW_API_KEY` không tìm thấy

Kiểm tra secret name, bật Internet, rồi restart session và chạy lại cell load secret. Không dán key vào notebook output.

### Không tìm thấy `best.pt`

Kiểm tra model Dataset đã được add vào Notebook bằng **Add Input**. Nếu Dataset chỉ có archive, bảo đảm archive có member tên `best.pt`, hoặc đặt `LPR_MODEL_PATH` tới file model.

### Không tìm thấy video

Kiểm tra path thực tế dưới `/kaggle/input`. Widget tự điền video `.mp4` đầu tiên nhưng khi có nhiều video cần sửa path thủ công.

### GPU device `0` lỗi

Đổi trường `Device` thành `auto` hoặc `cpu`. Kiểm tra Notebook settings đã bật GPU.

### Tesseract không tìm thấy

Chạy lại cell cài system package. Tesseract là binary hệ điều hành, không chỉ là Python package.

### Chọn nhầm run cũ

Dùng Kaggle Notebook/session mới hoặc chạy reset ở bước 3. Training notebook chỉ chọn run có `weights/best.pt`, vì vậy không nên để các run cũ trong workspace khi muốn audit một vòng train mới.

### Output metrics thấp hoặc OCR rỗng

Tách riêng hai vấn đề:

1. kiểm tra detector có bbox và confidence hợp lý hay không;
2. sau đó so sánh OCR backend/preprocessing trên cùng crop.

Không kết luận model detector kém chỉ từ một OCR prediction rỗng. Để tính OCR metric, tạo CSV có đúng hai cột `ground_truth,prediction` rồi chạy:

```bash
uv run lpr evaluate-ocr --csv /path/to/ocr.csv
```
