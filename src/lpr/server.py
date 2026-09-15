"""FastAPI HTTP/WebSocket adapter for realtime frame inference."""

from __future__ import annotations

import asyncio
import json
import os
import zipfile
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .detector import YoloPlateDetector
from .ocr import EasyOCRBackend, PaddleOCRBackend, TesseractBackend
from .stream import FrameResult, RealtimePlateRecognizer, StreamPlate


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Configuration exposed by the API health endpoint."""

    model_path: Path
    ocr_backend: str
    device: str | int | None
    imgsz: int
    confidence: float
    variants: tuple[str, ...]


def _env_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def load_runtime_config() -> RuntimeConfig:
    """Read runtime settings without exposing secrets in source control."""
    raw_path = os.getenv("LPR_MODEL_PATH", "models/best.pt")
    backend = os.getenv("LPR_OCR_BACKEND", "paddleocr").strip().lower()
    if backend not in {"tesseract", "easyocr", "paddleocr"}:
        raise ValueError("LPR_OCR_BACKEND must be tesseract, easyocr, or paddleocr")
    device = os.getenv("LPR_DEVICE") or None
    raw_variants = tuple(
        part.strip() for part in os.getenv("LPR_VARIANTS", "otsu").split(",") if part.strip()
    )
    if not raw_variants:
        raise ValueError("LPR_VARIANTS must contain at least one variant")
    return RuntimeConfig(
        model_path=Path(raw_path).expanduser(),
        ocr_backend=backend,
        device=device,
        imgsz=_env_int("LPR_IMGSZ", 640),
        confidence=float(os.getenv("LPR_CONFIDENCE", "0.4")),
        variants=raw_variants,
    )


def resolve_model_path(path: str | Path) -> Path:
    """Resolve a best.pt path or extract it from a training archive."""
    candidate = Path(path).expanduser()
    if candidate.is_file() and candidate.suffix.lower() != ".zip":
        return candidate.resolve()
    if candidate.is_file() and candidate.suffix.lower() == ".zip":
        with zipfile.ZipFile(candidate) as archive:
            members = [name for name in archive.namelist() if Path(name).name == "best.pt"]
            if not members:
                raise FileNotFoundError(f"Training archive has no best.pt: {candidate}")
            destination = candidate.parent / ".lpr-model" / "best.pt"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(members[0]))
            return destination.resolve()
    raise FileNotFoundError(f"Model file does not exist: {candidate}")


def _create_backend(name: str) -> Any:
    if name == "tesseract":
        return TesseractBackend()
    if name == "easyocr":
        return EasyOCRBackend(gpu=True)
    if name == "paddleocr":
        return PaddleOCRBackend()
    raise ValueError(f"Unsupported OCR backend: {name}")


def _build_recognizer(config: RuntimeConfig) -> RealtimePlateRecognizer:
    model_path = resolve_model_path(config.model_path)
    detector = YoloPlateDetector(
        model_path,
        confidence=config.confidence,
        device=config.device,
        imgsz=config.imgsz,
    )
    detector.warmup()
    return RealtimePlateRecognizer(
        detector,
        [_create_backend(config.ocr_backend)],
        variants=config.variants,
    )


def _plate_payload(plate: StreamPlate, width: int, height: int) -> dict[str, Any]:
    x1, y1, x2, y2 = plate.recognition.detection.bbox
    ocr = plate.recognition.ocr
    return {
        "bbox": [x1, y1, x2, y2],
        "bbox_norm": [x1 / width, y1 / height, x2 / width, y2 / height],
        "track_id": plate.track_id,
        "detection_confidence": plate.recognition.detection.confidence,
        "text": ocr.text if ocr else "",
        "raw_text": ocr.raw_text if ocr else "",
        "ocr_confidence": ocr.confidence if ocr else 0.0,
        "ocr_backend": ocr.backend if ocr else None,
        "preprocessing": ocr.variant if ocr else None,
        "stable": plate.stable,
        "status": plate.status,
    }


def frame_result_payload(result: FrameResult, width: int, height: int) -> dict[str, Any]:
    """Convert the internal result into the stable WebSocket response contract."""
    if width <= 0 or height <= 0:
        raise ValueError("Frame dimensions must be positive")
    return {
        "type": "result",
        "frame_id": result.frame_id,
        "source_time_ms": result.source_time_ms,
        "latency_ms": result.latency_ms,
        "dropped_frames": 0,
        "plates": [_plate_payload(plate, width, height) for plate in result.plates],
    }


def _decode_jpeg(payload: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError("Frame payload is not a valid JPEG image")
    return image


def create_app(recognizer: RealtimePlateRecognizer | None = None) -> FastAPI:
    """Create the FastAPI app and its model lifecycle."""

    config = load_runtime_config()

    @asynccontextmanager
    async def lifespan(app: Any):
        app.state.runtime_config = config
        app.state.recognizer = recognizer or await asyncio.to_thread(_build_recognizer, config)
        app.state.model_loaded = True
        yield
        app.state.model_loaded = False
        app.state.recognizer = None

    app = FastAPI(title="LPR realtime inference API", version="0.1.0", lifespan=lifespan)
    allowed_origins = [
        origin.strip()
        for origin in os.getenv("LPR_ALLOWED_ORIGINS", "http://localhost:5173").split(",")
        if origin.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["content-type", "x-demo-token"],
    )

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        runtime = app.state.runtime_config
        return {
            "status": "ok" if app.state.model_loaded else "starting",
            "model_loaded": app.state.model_loaded,
            "model_path": str(runtime.model_path),
            "device": runtime.device or "auto",
            "imgsz": runtime.imgsz,
            "ocr_backends": [runtime.ocr_backend],
        }

    @app.websocket("/ws/stream")
    async def stream(websocket: WebSocket) -> None:
        expected_token = os.getenv("LPR_DEMO_TOKEN")
        if expected_token and websocket.query_params.get("token") != expected_token:
            await websocket.close(code=1008, reason="Invalid demo token")
            return
        await websocket.accept()
        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                text = message.get("text")
                if text is not None:
                    payload = json.loads(text)
                    message_type = payload.get("type")
                    if message_type == "config":
                        config_width = int(payload.get("source_width", 0))
                        config_height = int(payload.get("source_height", 0))
                        if config_width <= 0 or config_height <= 0:
                            raise ValueError("config requires positive source dimensions")
                        await websocket.send_json({"type": "ready"})
                    elif message_type == "frame_meta":
                        frame_width = int(payload.get("width", 0))
                        frame_height = int(payload.get("height", 0))
                        if frame_width <= 0 or frame_height <= 0:
                            raise ValueError("frame_meta requires positive dimensions")
                        frame = await websocket.receive_bytes()
                        image = _decode_jpeg(frame)
                        result = await asyncio.to_thread(
                            app.state.recognizer.process_frame,
                            image,
                            int(payload.get("frame_id", 0)),
                            float(payload["source_time_ms"])
                            if payload.get("source_time_ms") is not None
                            else None,
                        )
                        await websocket.send_json(
                            frame_result_payload(result, frame_width, frame_height)
                        )
                    else:
                        await websocket.send_json(
                            {"type": "error", "message": "Unknown message type"}
                        )
                else:
                    await websocket.send_json(
                        {"type": "error", "message": "Expected JSON metadata"}
                    )
        except (WebSocketDisconnect, RuntimeError):
            return
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            await websocket.send_json({"type": "error", "message": str(error)})
            await websocket.close(code=1003)

    return app


app = create_app()
