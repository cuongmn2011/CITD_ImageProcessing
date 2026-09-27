"""Local upload page to try the fine-tuned PaddleOCR recognition model on plate crops.

Run: uv run python scripts/ocr-demo.py --model-dir model/ocr-rec-training-merged/inference
Then open http://127.0.0.1:8080 and upload a tightly cropped license-plate image.
"""

from __future__ import annotations

import argparse
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse

MAX_UPLOAD_BYTES = 10 * 1024 * 1024

PAGE = """<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Thử OCR biển số</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 640px; margin: 2rem auto; padding: 0 1rem; }
  #drop { border: 2px dashed #888; border-radius: 8px; padding: 2rem; text-align: center;
          cursor: pointer; }
  #drop.over { background: #eef; }
  img { max-width: 100%; margin-top: 1rem; border: 1px solid #ccc; }
  #result { margin-top: 1rem; font-size: 2rem; font-weight: 700; letter-spacing: .1em; }
  #meta { color: #555; }
  .error { color: #b00; font-size: 1rem; font-weight: 400; letter-spacing: 0; }
</style>
</head>
<body>
<h1>Thử OCR biển số</h1>
<p>Chọn ảnh <b>đã crop sát biển số</b> (chưa có bước YOLO ở trang này).</p>
<div id="drop">Kéo thả ảnh vào đây hoặc bấm để chọn
  <input id="file" type="file" accept="image/*" hidden>
</div>
<img id="preview" hidden alt="ảnh đã chọn">
<div id="result"></div>
<div id="meta"></div>
<script>
const drop = document.getElementById("drop");
const input = document.getElementById("file");
const preview = document.getElementById("preview");
const result = document.getElementById("result");
const meta = document.getElementById("meta");

async function send(file) {
  preview.src = URL.createObjectURL(file);
  preview.hidden = false;
  result.className = "";
  result.textContent = "Đang đọc...";
  meta.textContent = "";
  try {
    const response = await fetch("/api/ocr", { method: "POST", body: file });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || response.statusText);
    result.textContent = payload.text || "(không đọc được)";
    meta.textContent = "Độ tin cậy: " + payload.confidence.toFixed(3)
      + (payload.raw_text && payload.raw_text !== payload.text
         ? " · thô: " + payload.raw_text : "");
  } catch (error) {
    result.className = "error";
    result.textContent = "Lỗi: " + error.message;
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


def create_app(backend: Any) -> FastAPI:
    app = FastAPI(title="OCR demo")
    lock = threading.Lock()

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return PAGE

    @app.post("/api/ocr")
    async def ocr(request: Request) -> dict[str, Any]:
        body = await request.body()
        if not body:
            raise HTTPException(status_code=400, detail="Không có dữ liệu ảnh")
        if len(body) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Ảnh quá lớn (tối đa 10MB)")
        image = cv2.imdecode(np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Không đọc được file ảnh")
        with lock:
            result = backend.recognize(image)
        return {
            "text": result.text,
            "confidence": result.confidence,
            "raw_text": result.raw_text,
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Local upload page for the fine-tuned OCR model")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("model/ocr-rec-training-merged/inference"),
        help="Exported PaddleOCR inference directory (inference.json/.pdiparams/.yml)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    if not (args.model_dir / "inference.yml").is_file():
        parser.error(f"{args.model_dir} is not an exported inference model directory")

    from lpr.ocr import PaddleOCRBackend

    backend = PaddleOCRBackend(rec_model_dir=str(args.model_dir))
    uvicorn.run(create_app(backend), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
