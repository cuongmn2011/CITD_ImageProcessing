export type ConnectionState = "disconnected" | "connecting" | "connected" | "error";

export type PlateStatus = "stable" | "candidate" | "detected";

export interface PlateResult {
  bbox: [number, number, number, number];
  bbox_norm: [number, number, number, number];
  track_id: number | null;
  detection_confidence: number;
  text: string;
  raw_text: string;
  ocr_confidence: number;
  ocr_backend: string | null;
  preprocessing: string | null;
  stable: boolean;
  status: PlateStatus;
}

export interface FrameResult {
  type: "result";
  frame_id: number;
  source_time_ms: number | null;
  latency_ms: number;
  dropped_frames: number;
  plates: PlateResult[];
}

export interface RuntimeHealth {
  status: string;
  model_loaded: boolean;
  model_path: string;
  device: string;
  imgsz: number;
  ocr_backends: string[];
}

export interface PlateEvent {
  id: string;
  trackId: number | null;
  text: string;
  firstSeen: number;
  lastSeen: number;
  confidence: number;
}
