"""Local upload page to try the full pipeline: YOLO plate detection + fine-tuned OCR.

Run:
  uv run python scripts/pipeline-demo.py \\
    --model outputs/citd-yolo11s-training.zip \\
    --rec-model-dir model/ocr-rec-training-merged/inference
Then open http://127.0.0.1:8081 and upload a full photo (not a crop) with a visible plate.
"""

from __future__ import annotations

import argparse
import base64
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse

MAX_UPLOAD_BYTES = 15 * 1024 * 1024

PAGE = """<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Thử pipeline detect + OCR</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 720px; margin: 2rem auto; padding: 0 1rem; }
  #drop { border: 2px dashed #888; border-radius: 8px; padding: 2rem; text-align: center;
          cursor: pointer; }
  #drop.over { background: #eef; }
  img { max-width: 100%; margin-top: 1rem; border: 1px solid #ccc; }
  #plates { margin-top: 1rem; }
  .plate { font-size: 1.5rem; font-weight: 700; letter-spacing: .08em; padding: .5rem 0;
           border-bottom: 1px solid #ddd; }
  .plate .conf { font-size: 1rem; font-weight: 400; color: #555; margin-left: .5rem; }
  .error { color: #b00; }
</style>
</head>
<body>
<h1>Thử pipeline detect + OCR</h1>
<p>Chọn <b>ảnh nguyên cảnh</b> (không cần crop sẵn) — model YOLO sẽ tự tìm biển số.</p>
<div id="drop">Kéo thả ảnh vào đây hoặc bấm để chọn
  <input id="file" type="file" accept="image/*" hidden>
</div>
<img id="preview" hidden alt="kết quả">
<div id="plates"></div>
<script>
const drop = document.getElementById("drop");
const input = document.getElementById("file");
const preview = document.getElementById("preview");
const plates = document.getElementById("plates");

async function send(file) {
  plates.innerHTML = "Đang xử lý...";
  try {
    const response = await fetch("/api/detect", { method: "POST", body: file });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || response.statusText);
    preview.src = "data:image/jpeg;base64," + payload.annotated_image;
    preview.hidden = false;
    if (!payload.plates.length) {
      plates.innerHTML = "Không phát hiện biển số nào.";
      return;
    }
    plates.innerHTML = payload.plates.map((plate) =>
      `<div class="plate">${plate.text || "(không đọc được)"}` +
      `<span class="conf">detect ${plate.detection_confidence.toFixed(2)}` +
      ` · ocr ${plate.ocr_confidence.toFixed(2)}</span></div>`
    ).join("");
  } catch (error) {
    plates.innerHTML = "";
    const message = document.createElement("div");
    message.className = "error";
    message.textContent = "Lỗi: " + error.message;
    plates.appendChild(message);
  }
}

drop.addEventListener("click", () => input.click());
input.addEventListener("change", () => input.files[0] && send(input.files[0]));
drop.addEventListener("dragover", (event) => {
  event.preventDefault();
  drop.classList.add("over");
});
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (event) => {
  event.preventDefault();
  drop.classList.remove("over");
  if (event.dataTransfer.files[0]) send(event.dataTransfer.files[0]);
});
</script>
</body>
</html>
"""


def create_app(recognizer: Any) -> FastAPI:
    from lpr.pipeline import annotate_image

    app = FastAPI(title="Pipeline demo")
    lock = threading.Lock()

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return PAGE

    @app.post("/api/detect")
    async def detect(request: Request) -> dict[str, Any]:
        body = await request.body()
        if not body:
            raise HTTPException(status_code=400, detail="Không có dữ liệu ảnh")
        if len(body) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Ảnh quá lớn (tối đa 15MB)")
        image = cv2.imdecode(np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Không đọc được file ảnh")
        with lock:
            recognitions = recognizer.recognize_image(image)
        annotated = annotate_image(image, recognitions)
        ok, encoded = cv2.imencode(".jpg", annotated)
        if not ok:
            raise HTTPException(status_code=500, detail="Không mã hoá được ảnh kết quả")
        plates = [
            {
                "bbox": list(recognition.detection.bbox),
                "detection_confidence": recognition.detection.confidence,
                "text": recognition.ocr.text if recognition.ocr else "",
                "ocr_confidence": recognition.ocr.confidence if recognition.ocr else 0.0,
            }
            for recognition in recognitions
        ]
        return {
            "plates": plates,
            "annotated_image": base64.b64encode(encoded.tobytes()).decode("ascii"),
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Local upload page for the full LPR pipeline")
    parser.add_argument(
        "--model",
        default="outputs/citd-yolo11s-training.zip",
        help="YOLO weights (best.pt) or a training archive containing one",
    )
    parser.add_argument(
        "--rec-model-dir",
        default="model/ocr-rec-training-merged/inference",
        help="Exported PaddleOCR inference directory (inference.json/.pdiparams/.yml)",
    )
    parser.add_argument("--confidence", type=float, default=0.4)
    parser.add_argument("--device", default=None, help="YOLO device, for example 0 or cpu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()

    if not Path(args.rec_model_dir, "inference.yml").is_file():
        parser.error(f"{args.rec_model_dir} is not an exported inference model directory")

    from lpr.detector import YoloPlateDetector
    from lpr.ocr import PaddleOCRBackend
    from lpr.pipeline import LicensePlateRecognizer
    from lpr.server import resolve_model_path

    model_path = resolve_model_path(args.model)
    detector = YoloPlateDetector(model_path, confidence=args.confidence, device=args.device)
    backend = PaddleOCRBackend(rec_model_dir=args.rec_model_dir)
    recognizer = LicensePlateRecognizer(detector, [backend])

    uvicorn.run(create_app(recognizer), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
