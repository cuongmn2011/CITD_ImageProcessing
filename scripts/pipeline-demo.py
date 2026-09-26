"""Local test page for the full pipeline: YOLO plate detection + fine-tuned OCR.

Image mode reads one photo; video mode tracks vehicles across a video file, votes on
each vehicle's plate, and scores the result against a list of the true plates.

Run:
  uv run python scripts/pipeline-demo.py \\
    --model outputs/citd-yolo11s-training.zip \\
    --rec-model-dir model/ocr-rec-training-merged/inference
Then open http://127.0.0.1:8081.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, Response

MAX_UPLOAD_BYTES = 15 * 1024 * 1024
LIVE_FRAME_WIDTH = 960
MAX_VIDEO_BYTES = 500 * 1024 * 1024
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

PAGE = """<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Thử pipeline detect + OCR</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 860px; margin: 2rem auto; padding: 0 1rem; }
  nav button { font-size: 1rem; padding: .4rem 1rem; margin-right: .3rem; cursor: pointer; }
  nav button.active { font-weight: 700; }
  .drop { border: 2px dashed #888; border-radius: 8px; padding: 2rem; text-align: center;
          cursor: pointer; }
  .drop.over { background: #eef; }
  img.result, video { max-width: 100%; margin-top: 1rem; border: 1px solid #ccc; }
  .video-live { display: flex; gap: 1rem; align-items: flex-start; flex-wrap: wrap; }
  .video-live .video-frame { flex: 2; min-width: 280px; }
  .video-live .video-frame img.result, .video-live .video-frame video { margin-top: 0; }
  #track-table { flex: 1; width: auto; min-width: 220px; margin-top: 0; }
  .plate { font-size: 1.5rem; font-weight: 700; letter-spacing: .08em; padding: .5rem 0;
           border-bottom: 1px solid #ddd; }
  .muted { color: #555; font-size: .9rem; font-weight: 400; letter-spacing: 0; }
  .error { color: #b00; }
  label { margin-right: 1rem; }
  progress { width: 100%; margin-top: .5rem; }
  table { border-collapse: collapse; width: 100%; margin-top: 1rem; }
  th, td { border-bottom: 1px solid #ddd; padding: .35rem; text-align: left;
           vertical-align: middle; }
  td.text { font-weight: 700; letter-spacing: .06em; }
  td img { max-height: 48px; }
  textarea { width: 100%; min-height: 6rem; font-family: monospace; }
  .correct { color: #070; } .wrong { color: #b60; } .missed { color: #b00; }
</style>
</head>
<body>
<h1>Thử pipeline detect + OCR</h1>
<nav><button id="tab-image" class="active">Ảnh</button><button id="tab-video">Video</button></nav>

<section id="image-section">
  <p>Chọn <b>ảnh nguyên cảnh</b> (không cần crop sẵn), model YOLO sẽ tự tìm biển số.</p>
  <div id="image-drop" class="drop">Kéo thả ảnh vào đây hoặc bấm để chọn
    <input id="image-file" type="file" accept="image/*" hidden>
  </div>
  <img id="image-preview" class="result" hidden alt="kết quả">
  <div id="image-plates"></div>
</section>

<section id="video-section" hidden>
  <p>Chọn file video: video phát ngay bình thường, việc nhận diện và đọc biển số chạy
    nền và biển số hiện dần bên phải khi đọc xong.</p>
  <p>
    <label>Bỏ bớt frame:
      <select id="stride">
        <option value="1">1 (xử lý mọi frame)</option>
        <option value="2">2</option>
        <option value="3">3</option>
      </select>
    </label>
    <label>Tiền xử lý:
      <select id="variants">
        <option value="raw" selected>raw (ảnh màu gốc, nên dùng)</option>
        <option value="otsu">otsu</option>
        <option value="raw,otsu">raw + otsu</option>
      </select>
    </label>
  </p>
  <p>
    <label>Bắt đầu từ giây: <input id="start" type="number" min="0" step="1" value="0"
      style="width:5rem"></label>
    <label>Chỉ xử lý (giây, 0 = đến hết video): <input id="duration" type="number" min="0"
      step="1" value="60" style="width:5rem"></label>
  </p>
  <p class="muted">Tốc độ xử lý nền (không ảnh hưởng lúc phát video): đo trên CPU với video
    đường phố 30fps, xử lý mọi frame mất khoảng 10 giây nền cho mỗi giây video; bỏ bớt 3 frame
    chỉ mất khoảng 4,5 giây nền và vẫn thấy đủ xe, nhưng biển khó dễ đọc sai hơn.
    otsu đọc sai nhiều biển xe máy 2 dòng.</p>
  <div id="video-drop" class="drop">Kéo thả video vào đây hoặc bấm để chọn
    <input id="video-file" type="file" accept="video/*" hidden>
  </div>
  <div id="video-status"></div>
  <progress id="video-progress" max="1" value="0" hidden></progress>
  <button id="stop-button" hidden>Dừng (giữ kết quả đã xử lý)</button>
  <div class="video-live">
    <div class="video-frame">
      <video id="video-result" controls hidden></video>
    </div>
    <table id="track-table" hidden>
      <thead><tr><th>Ảnh biển</th><th>Biển số đã chốt</th><th>Thời điểm</th></tr></thead>
      <tbody></tbody>
    </table>
  </div>
  <p id="video-download" hidden></p>
  <p class="muted">Video ở trên là file gốc bạn chọn, phát bình thường ngay. Video có khung
    nhận diện chỉ có sau khi xử lý nền xong, tải qua nút bên dưới.</p>
  <div id="score-box" hidden>
    <h3>Chấm điểm</h3>
    <p class="muted">Dán danh sách biển số thật xuất hiện trong video, mỗi dòng một biển.</p>
    <textarea id="truths" placeholder="51G48154&#10;60F140494"></textarea>
    <button id="score-button">Chấm điểm</button>
    <div id="score-result"></div>
  </div>
</section>

<script>
const $ = (id) => document.getElementById(id);

function showTab(name) {
  $("image-section").hidden = name !== "image";
  $("video-section").hidden = name !== "video";
  $("tab-image").classList.toggle("active", name === "image");
  $("tab-video").classList.toggle("active", name === "video");
}
$("tab-image").addEventListener("click", () => showTab("image"));
$("tab-video").addEventListener("click", () => showTab("video"));

function setError(element, message) {
  element.textContent = "";
  const div = document.createElement("div");
  div.className = "error";
  div.textContent = "Lỗi: " + message;
  element.appendChild(div);
}

function wireDrop(dropId, inputId, handler) {
  const drop = $(dropId);
  const input = $(inputId);
  drop.addEventListener("click", () => input.click());
  input.addEventListener("change", () => input.files[0] && handler(input.files[0]));
  drop.addEventListener("dragover", (event) => {
    event.preventDefault();
    drop.classList.add("over");
  });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (event) => {
    event.preventDefault();
    drop.classList.remove("over");
    if (event.dataTransfer.files[0]) handler(event.dataTransfer.files[0]);
  });
}

async function readJson(response) {
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail || response.statusText);
  return payload;
}

async function sendImage(file) {
  const plates = $("image-plates");
  plates.textContent = "Đang xử lý...";
  try {
    const payload = await readJson(await fetch("/api/detect", { method: "POST", body: file }));
    $("image-preview").src = "data:image/jpeg;base64," + payload.annotated_image;
    $("image-preview").hidden = false;
    plates.textContent = payload.plates.length ? "" : "Không phát hiện biển số nào.";
    for (const plate of payload.plates) {
      const row = document.createElement("div");
      row.className = "plate";
      row.textContent = plate.text || "(không đọc được)";
      const conf = document.createElement("span");
      conf.className = "muted";
      conf.textContent = ` detect ${plate.detection_confidence.toFixed(2)}` +
        ` · ocr ${plate.ocr_confidence.toFixed(2)}`;
      row.appendChild(conf);
      plates.appendChild(row);
    }
  } catch (error) {
    setError(plates, error.message);
  }
}

function formatTime(ms) {
  const seconds = ms / 1000;
  return `${Math.floor(seconds / 60)}:${(seconds % 60).toFixed(1).padStart(4, "0")}`;
}

function renderTracks(jobId, tracks) {
  const table = $("track-table");
  const body = table.querySelector("tbody");
  body.textContent = "";
  // Only vehicles with a locked-in reading; unstable ones are still being voted on.
  const stable = tracks.filter((track) => track.stable);
  table.hidden = stable.length === 0;
  // Newest locked-in vehicle first, so what just appeared is at the top.
  for (const track of [...stable].reverse()) {
    const row = document.createElement("tr");
    const imageCell = document.createElement("td");
    const image = document.createElement("img");
    // The crop improves while the vehicle is tracked; the version tag refreshes it.
    image.src = `/api/video/${jobId}/crop/${track.track_id}.jpg?v=${track.detection_confidence}`;
    image.alt = "biển số";
    image.onerror = () => { image.hidden = true; };
    imageCell.appendChild(image);
    const textCell = document.createElement("td");
    textCell.className = "text";
    textCell.textContent = track.text || "(chưa đọc được)";
    const timeCell = document.createElement("td");
    timeCell.textContent = `${formatTime(track.first_time_ms)} – ${formatTime(track.last_time_ms)}`;
    row.append(imageCell, textCell, timeCell);
    body.appendChild(row);
  }
}

let currentJob = null;

async function pollJob(jobId) {
  const status = $("video-status");
  const progress = $("video-progress");
  try {
    const job = await readJson(await fetch(`/api/video/${jobId}`));
    const total = job.frames_total || 0;
    progress.hidden = false;
    progress.max = total || 1;
    progress.value = Math.min(job.frames_done, total || 1);
    const stableCount = job.tracks.filter((track) => track.stable).length;
    const count = `${stableCount}/${job.tracks.length} xe đã chốt`;
    renderTracks(jobId, job.tracks);
    if (job.state === "running") {
      status.textContent = `Đang đọc biển số nền: frame ${job.frames_done}/${total || "?"}` +
        ` · ${count}`;
      $("stop-button").hidden = false;
      setTimeout(() => pollJob(jobId), 500);
      return;
    }
    $("stop-button").hidden = true;
    if (job.state === "error") {
      setError(status, job.error || "không rõ");
      return;
    }
    status.textContent = (job.stopped ? "Đã dừng" : "Xong") +
      `: ${job.frames_processed} frame đã xử lý · ${count}`;
    const videoUrl = `/api/video/${jobId}/video`;
    const link = document.createElement("a");
    link.href = videoUrl;
    link.download = job.output_name;
    link.textContent = "Tải video có khung nhận diện";
    $("video-download").textContent = "";
    $("video-download").appendChild(link);
    $("video-download").hidden = false;
    $("score-box").hidden = false;
  } catch (error) {
    setError(status, error.message);
  }
}

let previewUrl = null;

function playLocally(file, startSeconds) {
  // The file is already on this machine, so it plays instantly at normal speed, with no
  // wait for upload or background processing.
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  const preview = $("video-result");
  preview.src = previewUrl;
  preview.hidden = false;
  preview.muted = true; // autoplay is usually blocked with sound; controls stay enabled
  if (startSeconds > 0) {
    preview.addEventListener("loadedmetadata", () => { preview.currentTime = startSeconds; },
      { once: true });
  }
  preview.play().catch(() => {}); // user can press play if the browser still blocks it
}

async function sendVideo(file) {
  const status = $("video-status");
  $("video-download").hidden = true;
  $("score-box").hidden = true;
  $("score-result").textContent = "";
  $("track-table").hidden = true;
  const startSeconds = Number($("start").value) || 0;
  playLocally(file, startSeconds);
  status.textContent = "Video đang phát bình thường; đang tải lên để đọc biển số nền...";
  const params = new URLSearchParams({
    stride: $("stride").value,
    variants: $("variants").value,
    filename: file.name,
    start: String(startSeconds),
    duration: $("duration").value || "0",
  });
  try {
    const job = await readJson(await fetch(`/api/video?${params}`, { method: "POST", body: file }));
    currentJob = job.job_id;
    pollJob(currentJob);
  } catch (error) {
    setError(status, error.message);
  }
}

$("stop-button").addEventListener("click", async () => {
  if (!currentJob) return;
  $("stop-button").hidden = true;
  await fetch(`/api/video/${currentJob}/stop`, { method: "POST" });
});

$("score-button").addEventListener("click", async () => {
  const output = $("score-result");
  output.textContent = "Đang chấm...";
  try {
    const score = await readJson(await fetch(`/api/video/${currentJob}/score`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ truths: $("truths").value }),
    }));
    output.textContent = "";
    const summary = document.createElement("p");
    summary.textContent =
      `Đúng ${score.correct}/${score.samples} · sai ${score.wrong} · sót ${score.missed}` +
      ` · exact accuracy ${(score.exact_accuracy * 100).toFixed(1)}%` +
      ` · CER ${score.cer.toFixed(3)}`;
    output.appendChild(summary);
    const table = document.createElement("table");
    const head = document.createElement("tr");
    for (const title of ["Biển thật", "Đọc được", "Kết quả"]) {
      const cell = document.createElement("th");
      cell.textContent = title;
      head.appendChild(cell);
    }
    table.appendChild(head);
    const labels = { correct: "đúng", wrong: "sai", missed: "bỏ sót" };
    for (const detail of score.details) {
      const row = document.createElement("tr");
      for (const value of [detail.ground_truth, detail.prediction || "—", labels[detail.status]]) {
        const cell = document.createElement("td");
        cell.textContent = value;
        row.appendChild(cell);
      }
      row.lastChild.className = detail.status;
      table.appendChild(row);
    }
    output.appendChild(table);
    if (score.extra.length) {
      const extra = document.createElement("p");
      extra.className = "muted";
      extra.textContent = "Đọc ra nhưng không khớp biển thật nào: " + score.extra.join(", ");
      output.appendChild(extra);
    }
  } catch (error) {
    setError(output, error.message);
  }
});

wireDrop("image-drop", "image-file", sendImage);
wireDrop("video-drop", "video-file", sendVideo);
</script>
</body>
</html>
"""


@dataclass(slots=True)
class _VideoJob:
    job_id: str
    directory: Path
    stride: int
    variants: tuple[str, ...]
    start_seconds: float = 0.0
    duration_seconds: float | None = None
    state: str = "running"
    frames_done: int = 0
    frames_total: int = 0
    frames_processed: int = 0
    output_name: str = ""
    error: str = ""
    stopped: bool = False
    stop_requested: bool = False
    tracks: list[dict[str, Any]] = field(default_factory=list)
    # Live view while running: last annotated frame and each track's best crop so far.
    latest_frame: bytes | None = None
    frame_seq: int = 0
    live_crops: dict[int, np.ndarray] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "state": self.state,
            "stride": self.stride,
            "variants": list(self.variants),
            "start_seconds": self.start_seconds,
            "duration_seconds": self.duration_seconds,
            "frames_done": self.frames_done,
            "frames_total": self.frames_total,
            "frames_processed": self.frames_processed,
            "frame_seq": self.frame_seq,
            "output_name": self.output_name,
            "error": self.error,
            "stopped": self.stopped,
            "tracks": self.tracks,
        }


def _parse_variants(raw: str) -> tuple[str, ...]:
    from lpr.preprocessing import PREPROCESS_VARIANTS

    variants = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not variants or set(variants) - set(PREPROCESS_VARIANTS):
        raise HTTPException(status_code=400, detail=f"variants phải thuộc {PREPROCESS_VARIANTS}")
    return variants


def _load_finished_jobs(jobs_dir: Path) -> dict[str, _VideoJob]:
    """Re-register jobs from earlier server runs so their results stay viewable."""
    jobs: dict[str, _VideoJob] = {}
    for report_path in sorted(jobs_dir.glob("*/report.json")):
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            job = _VideoJob(
                job_id=report_path.parent.name,
                directory=report_path.parent,
                stride=int(report["stride"]),
                variants=tuple(report["variants"]),
                state="done",
                frames_done=int(report["frames_total"]),
                frames_total=int(report["frames_total"]),
                frames_processed=int(report["frames_processed"]),
                output_name=str(report["output_name"]),
                stopped=bool(report.get("stopped", False)),
                tracks=list(report["tracks"]),
            )
        except (OSError, ValueError, KeyError, TypeError):
            continue
        jobs[job.job_id] = job
    return jobs


def create_app(
    recognizer: Any,
    make_video_recognizer: Callable[[tuple[str, ...]], Any] | None = None,
    jobs_dir: Path = Path("outputs/video-jobs"),
) -> FastAPI:
    from lpr.pipeline import annotate_image
    from lpr.video_report import process_video, score_against_ground_truth

    app = FastAPI(title="Pipeline demo")
    image_lock = threading.Lock()
    jobs_lock = threading.Lock()
    jobs = _load_finished_jobs(jobs_dir) if jobs_dir.is_dir() else {}

    def running_job() -> _VideoJob | None:
        return next((job for job in jobs.values() if job.state == "running"), None)

    def get_job(job_id: str) -> _VideoJob:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Không tìm thấy job")
        return job

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return PAGE

    @app.post("/api/detect")
    async def detect(request: Request) -> dict[str, Any]:
        if running_job() is not None:
            raise HTTPException(status_code=409, detail="Đang xử lý video, thử lại sau")
        body = await request.body()
        if not body:
            raise HTTPException(status_code=400, detail="Không có dữ liệu ảnh")
        if len(body) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Ảnh quá lớn (tối đa 15MB)")
        image = cv2.imdecode(np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Không đọc được file ảnh")
        with image_lock:
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

    def run_job(job: _VideoJob, input_path: Path) -> None:
        def on_progress(done: int, total: int, tracks: list[Any]) -> None:
            job.frames_done = done
            job.frames_total = total
            job.tracks = [track.to_dict() for track in tracks]
            job.live_crops = {
                track.track_id: track.crop for track in tracks if track.crop is not None
            }

        def on_frame(annotated: np.ndarray, frame_index: int) -> None:
            height, width = annotated.shape[:2]
            if width > LIVE_FRAME_WIDTH:
                scale = LIVE_FRAME_WIDTH / width
                annotated = cv2.resize(
                    annotated, (LIVE_FRAME_WIDTH, int(height * scale)), interpolation=cv2.INTER_AREA
                )
            ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                job.latest_frame = encoded.tobytes()
                job.frame_seq += 1

        try:
            with image_lock:
                video_recognizer = make_video_recognizer(job.variants)
                report = process_video(
                    video_recognizer,
                    input_path,
                    job.directory / "annotated.webm",
                    stride=job.stride,
                    start_seconds=job.start_seconds,
                    duration_seconds=job.duration_seconds,
                    on_progress=on_progress,
                    on_frame=on_frame,
                    should_stop=lambda: job.stop_requested,
                )
            crops_dir = job.directory / "crops"
            crops_dir.mkdir(exist_ok=True)
            for track in report.tracks:
                if track.crop is not None and track.crop.size:
                    cv2.imwrite(str(crops_dir / f"{track.track_id}.jpg"), track.crop)
            job.tracks = [track.to_dict() for track in report.tracks]
            job.frames_total = report.frames_total
            job.frames_done = report.frames_total
            job.frames_processed = report.frames_processed
            job.output_name = report.output_path.name
            job.stopped = report.stopped
            job.live_crops = {}
            (job.directory / "report.json").write_text(
                json.dumps(
                    {
                        "stride": job.stride,
                        "variants": list(job.variants),
                        "start_seconds": job.start_seconds,
                        "duration_seconds": job.duration_seconds,
                        "frames_total": job.frames_total,
                        "frames_processed": job.frames_processed,
                        "stopped": job.stopped,
                        "output_name": job.output_name,
                        "tracks": job.tracks,
                        "fragments": [track.to_dict() for track in report.fragments],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            job.state = "done"
        except Exception as error:  # surfaced to the page instead of dying in the thread
            job.error = str(error)
            job.state = "error"

    @app.post("/api/video")
    async def upload_video(
        request: Request,
        stride: int = 1,
        variants: str = "raw",
        filename: str = "video.mp4",
        start: float = 0.0,
        duration: float = 0.0,
    ) -> dict[str, str]:
        if make_video_recognizer is None:
            raise HTTPException(status_code=503, detail="Chế độ video chưa được bật")
        if stride < 1 or stride > 10:
            raise HTTPException(status_code=400, detail="stride phải từ 1 đến 10")
        if start < 0 or duration < 0:
            raise HTTPException(status_code=400, detail="start và duration không được âm")
        parsed_variants = _parse_variants(variants)
        suffix = Path(filename).suffix.lower()
        if suffix not in VIDEO_SUFFIXES:
            suffix = ".mp4"
        with jobs_lock:
            if running_job() is not None:
                raise HTTPException(status_code=409, detail="Đang xử lý một video khác")
            job_id = uuid.uuid4().hex[:12]
            directory = jobs_dir / job_id
            directory.mkdir(parents=True)
            job = _VideoJob(
                job_id,
                directory,
                stride,
                parsed_variants,
                start_seconds=start,
                duration_seconds=duration or None,
            )
            jobs[job_id] = job

        input_path = directory / f"input{suffix}"
        size = 0
        try:
            with input_path.open("wb") as file:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > MAX_VIDEO_BYTES:
                        raise HTTPException(status_code=413, detail="Video quá lớn (tối đa 500MB)")
                    file.write(chunk)
            if size == 0:
                raise HTTPException(status_code=400, detail="Không có dữ liệu video")
        except HTTPException as error:
            job.state = "error"
            job.error = str(error.detail)
            input_path.unlink(missing_ok=True)
            raise

        threading.Thread(target=run_job, args=(job, input_path), daemon=True).start()
        return {"job_id": job_id}

    @app.get("/api/video/{job_id}")
    def video_status(job_id: str) -> dict[str, Any]:
        return get_job(job_id).to_dict()

    @app.get("/api/video/{job_id}/video")
    def video_output(job_id: str) -> FileResponse:
        job = get_job(job_id)
        path = job.directory / job.output_name if job.output_name else None
        if job.state != "done" or path is None or not path.is_file():
            raise HTTPException(status_code=404, detail="Chưa có video kết quả")
        media_type = "video/webm" if path.suffix == ".webm" else "video/mp4"
        return FileResponse(path, media_type=media_type, filename=path.name)

    @app.get("/api/video/{job_id}/frame.jpg")
    def video_frame(job_id: str) -> Response:
        frame = get_job(job_id).latest_frame
        if frame is None:
            raise HTTPException(status_code=404, detail="Chưa có frame nào")
        return Response(frame, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.post("/api/video/{job_id}/stop")
    def video_stop(job_id: str) -> dict[str, str]:
        job = get_job(job_id)
        if job.state == "running":
            job.stop_requested = True
        return {"state": job.state}

    @app.get("/api/video/{job_id}/crop/{track_id}.jpg")
    def video_crop(job_id: str, track_id: int) -> Response:
        job = get_job(job_id)
        path = job.directory / "crops" / f"{track_id}.jpg"
        if path.is_file():
            return FileResponse(path, media_type="image/jpeg")
        crop = job.live_crops.get(track_id)
        if crop is not None and crop.size:
            ok, encoded = cv2.imencode(".jpg", crop)
            if ok:
                return Response(encoded.tobytes(), media_type="image/jpeg")
        raise HTTPException(status_code=404, detail="Không có ảnh biển số")

    @app.post("/api/video/{job_id}/score")
    async def video_score(job_id: str, request: Request) -> dict[str, Any]:
        job = get_job(job_id)
        if job.state != "done":
            raise HTTPException(status_code=409, detail="Video chưa xử lý xong")
        try:
            payload = await request.json()
            raw_truths = payload["truths"]
        except (ValueError, KeyError, TypeError) as error:
            raise HTTPException(status_code=400, detail="Cần JSON {\"truths\": \"...\"}") from error
        truths = [line for line in re.split(r"[\r\n,;]+", str(raw_truths)) if line.strip()]
        try:
            score = score_against_ground_truth([track["text"] for track in job.tracks], truths)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        (job.directory / "score.json").write_text(
            json.dumps({"truths": truths, **score}, indent=2), encoding="utf-8"
        )
        return score

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Local test page for the full LPR pipeline")
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
    parser.add_argument(
        "--ocr-det-model",
        default="PP-OCRv5_mobile_det",
        help="PaddleOCR text detector; PP-OCRv5_server_det (PaddleOCR's default) was about "
        "twice as slow on CPU without reading better on the labelled test plates",
    )
    parser.add_argument(
        "--tracker",
        choices=("centroid", "bytetrack"),
        default="centroid",
        help="Video mode tracker; ByteTrack dropped small, fast plates (see lpr.tracking)",
    )
    parser.add_argument("--confidence", type=float, default=0.4)
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="YOLO input size; try 960 or 1280 for small, distant street-camera plates",
    )
    parser.add_argument("--device", default=None, help="YOLO device, for example 0 or cpu")
    parser.add_argument("--jobs-dir", type=Path, default=Path("outputs/video-jobs"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()

    if not Path(args.rec_model_dir, "inference.yml").is_file():
        parser.error(f"{args.rec_model_dir} is not an exported inference model directory")

    from lpr.detector import YoloPlateDetector
    from lpr.ocr import PaddleOCRBackend
    from lpr.pipeline import LicensePlateRecognizer
    from lpr.server import resolve_model_path
    from lpr.stream import RealtimePlateRecognizer
    from lpr.tracking import CentroidTrackingDetector

    model_path = resolve_model_path(args.model)

    def make_detector() -> YoloPlateDetector:
        return YoloPlateDetector(
            model_path, confidence=args.confidence, device=args.device, imgsz=args.imgsz
        )

    backend = PaddleOCRBackend(rec_model_dir=args.rec_model_dir, det_model_name=args.ocr_det_model)
    recognizer = LicensePlateRecognizer(make_detector(), [backend])

    def make_video_recognizer(variants: tuple[str, ...]) -> RealtimePlateRecognizer:
        # A fresh detector per video: Ultralytics' track(persist=True) keeps tracker state.
        detector = make_detector()
        if args.tracker == "centroid":
            detector = CentroidTrackingDetector(detector)
        # The report votes over every read, so keep reading after a track locks in: one
        # street-video read was right about 43% of the time, and on a 30 s clip this added
        # about 5% run time.
        return RealtimePlateRecognizer(
            detector, [backend], variants=variants, ocr_refresh_frames=3
        )

    app = create_app(recognizer, make_video_recognizer, args.jobs_dir)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
