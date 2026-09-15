# Realtime React Demo Specification

## Scope

The demo runs a local video in a React browser UI. The browser samples frames and sends only JPEG frames to a GPU inference API. The API returns structured detections and OCR results over WebSocket; React draws the overlay locally.

This is a one-session academic demo, not a production public service.

## Supported claim

The first release claims Vietnamese civilian plates represented in the benchmark dataset and evaluation videos:

- white and yellow car plates;
- one-line car plates;
- one-line and two-line motorcycle plates when present in the validated dataset.

Military, diplomatic, official, special-format, or non-ASCII plates are reported as `unsupported_format` until each group has independent data and metrics. A single detection-class dataset does not prove coverage of every Vietnamese plate type.

## Runtime architecture

```text
React on local/Vercel
  -> hidden canvas samples local video
  -> HTTPS/WSS tunnel
FastAPI on Colab GPU
  -> bounded single-frame queue
  -> YOLO detection + ByteTrack
  -> OCR gate/cache + temporal voting
  -> JSON FrameResult
React
  -> canvas bbox overlay
  -> current plate table
  -> stable event history and CSV export
```

The browser connects directly to the GPU API. Vercel serverless functions are not used as a WebSocket proxy.

## WebSocket contract

1. Client sends a text configuration message:

```json
{"type":"config","source_width":1280,"source_height":720}
```

2. For each frame, client sends metadata followed immediately by binary JPEG bytes:

```json
{"type":"frame_meta","frame_id":142,"source_time_ms":4733,"width":960,"height":540}
```

3. Server returns:

```json
{
  "type": "result",
  "frame_id": 142,
  "source_time_ms": 4733,
  "latency_ms": 118.4,
  "dropped_frames": 0,
  "plates": [
    {
      "bbox_norm": [0.31,0.42,0.48,0.53],
      "track_id": 7,
      "detection_confidence": 0.94,
      "text": "59H12345",
      "raw_text": "59H1 234.5",
      "ocr_confidence": 0.88,
      "ocr_backend": "paddleocr",
      "preprocessing": "otsu",
      "stable": true,
      "status": "stable"
    }
  ]
}
```

The server uses the transmitted frame dimensions for `bbox_norm`; the source video dimensions are metadata only.

## Realtime behavior

- React sends at most one frame in flight.
- The API keeps at most one queued frame and drops stale work when necessary.
- YOLO uses the configured `imgsz`; Colab GPU should be benchmarked at 640 first.
- OCR uses one selected GPU backend and one preprocessing variant on the realtime path.
- OCR runs for new or unstable tracks and refreshes after a configurable interval.
- ByteTrack provides identity across frames.
- Stable text requires repeated valid observations; a candidate remains visible while it is not stable.
- Closing the socket releases session OCR state.

## API configuration

```bash
export LPR_MODEL_PATH=/content/drive/MyDrive/lpr/best.pt
export LPR_OCR_BACKEND=paddleocr
export LPR_DEVICE=0
export LPR_IMGSZ=640
export LPR_VARIANTS=otsu
export LPR_ALLOWED_ORIGINS=https://your-project.vercel.app,http://localhost:5173
export LPR_DEMO_TOKEN='change-this-for-the-defense'

uv run --extra vision --extra paddle --extra web lpr serve \
  --model "$LPR_MODEL_PATH" \
  --ocr-backend "$LPR_OCR_BACKEND" \
  --device "$LPR_DEVICE" \
  --imgsz "$LPR_IMGSZ" \
  --host 0.0.0.0 \
  --port 8000
```

The model can also be a training zip; the API extracts `best.pt` into a sibling `.lpr-model/` directory. Keep model artifacts outside Git and verify the SHA-256 before the demo.

## Cloudflare Quick Tunnel

In the Colab runtime, expose the API only for the defense session:

```bash
cloudflared tunnel --url http://127.0.0.1:8000
```

Use the generated `https://...trycloudflare.com` URL in the React `GPU API URL` field. The WebSocket URL is derived as `wss://...trycloudflare.com/ws/stream`.

Before opening the UI, verify:

```bash
curl https://<tunnel-host>/api/health
```

Quick Tunnel URLs are ephemeral. The React UI deliberately keeps the API URL as runtime configuration instead of baking it into a Vercel build.

## React deployment

Local development:

```bash
cd web
npm install
npm run dev
```

Vercel settings:

- Root directory: `web`.
- Build command: `npm run build`.
- Output directory: `dist`.
- No server-side inference function.
- Enter the current tunnel URL in the UI at runtime.

## UI acceptance criteria

- User selects a local video and playback remains continuous.
- Bboxes update while playback is running; no full-video upload is required.
- Plate table has one row per current track, not one row per frame.
- Stable event history and CSV export are available in the browser.
- Overlay, table, round-trip latency, inference FPS, target FPS, and dropped frames are visible.
- API disconnect, invalid model, invalid JPEG, and timeout states are visible.
- A benchmark run with one session reaches p95 glass-to-overlay latency below 500 ms on the selected videos, or the measured limitation is reported instead of hidden.

## Evaluation protocol

Do not split adjacent frames from one video across train and test. Split by video or vehicle instance.

Ground truth should include:

```text
video_id, frame_id, gt_plate_id, bbox, vehicle_type, line_count,
plate_background, ground_truth, condition
```

Report separately:

- detector precision/recall/mAP on an independent test split;
- OCR exact accuracy, character accuracy, and CER;
- end-to-end plate recognition rate;
- track-level stable-text rate;
- p50/p95 detector, OCR, and glass-to-overlay latency;
- dropped frames and inference FPS;
- results by vehicle type, line count, plate background, day/night/rain, and angle.

Do not infer OCR or end-to-end accuracy from the existing validation mAP.

## Feature branch and merge order

Use one branch per feature, stacked in dependency order:

```text
feature/realtime-streaming
  -> feature/realtime-api
  -> feature/react-realtime-dashboard
  -> feature/colab-realtime-deployment
```

Each branch has its own focused commit and can be reviewed independently. The existing untracked `.coverage` file is local workspace state and must not be committed.
