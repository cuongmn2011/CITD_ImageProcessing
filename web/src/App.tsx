import { useCallback, useEffect, useMemo, useRef, useState, type ChangeEvent } from "react";
import { OverlayCanvas } from "./components/OverlayCanvas";
import { PlateTable } from "./components/PlateTable";
import { RuntimeMetrics } from "./components/RuntimeMetrics";
import { useFrameSender } from "./hooks/useFrameSender";
import { usePlateSocket } from "./hooks/usePlateSocket";
import type { PlateEvent, PlateResult } from "./types";
import "./styles.css";

const DEFAULT_API_URL = import.meta.env.VITE_LPR_API_URL || "http://localhost:8000";
const API_URL_KEY = "lpr-api-url";

function formatTime(milliseconds: number): string {
  const seconds = Math.max(0, Math.round(milliseconds / 1000));
  return `${Math.floor(seconds / 60).toString().padStart(2, "0")}:${(seconds % 60).toString().padStart(2, "0")}`;
}

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const captureCanvasRef = useRef<HTMLCanvasElement>(null);
  const [apiUrl, setApiUrl] = useState(() => localStorage.getItem(API_URL_KEY) || DEFAULT_API_URL);
  const [token, setToken] = useState("");
  const [videoUrl, setVideoUrl] = useState("");
  const [plates, setPlates] = useState<PlateResult[]>([]);
  const [history, setHistory] = useState<PlateEvent[]>([]);
  const [latency, setLatency] = useState(0);
  const [dropped, setDropped] = useState(0);
  const [inferenceSamples, setInferenceSamples] = useState<number[]>([]);
  const [sampleFps, setSampleFps] = useState(10);
  const { state, lastResult, health, error, connect, disconnect, sendConfig, sendFrame } = usePlateSocket();

  const connected = state === "connected";
  useFrameSender({
    videoRef,
    canvasRef: captureCanvasRef,
    enabled: connected && Boolean(videoUrl),
    sampleFps,
    sendConfig,
    sendFrame,
  });

  useEffect(() => {
    if (!lastResult) return;
    setPlates(lastResult.plates);
    setLatency(lastResult.latency_ms);
    setDropped((value) => value + lastResult.dropped_frames);
    setInferenceSamples((values) => [...values.slice(-19), performance.now()]);
    const video = videoRef.current;
    const now = lastResult.source_time_ms ?? Date.now();
    setHistory((events) => {
      const next = [...events];
      for (const plate of lastResult.plates.filter((item) => item.stable && item.text)) {
        const id = `${plate.track_id ?? "text"}:${plate.text}`;
        const existing = next.find((event) => event.id === id);
        if (existing) {
          existing.lastSeen = now;
          existing.confidence = plate.ocr_confidence;
        } else {
          next.unshift({
            id,
            trackId: plate.track_id,
            text: plate.text,
            firstSeen: now,
            lastSeen: now,
            confidence: plate.ocr_confidence,
          });
        }
      }
      return next.slice(0, 100);
    });
  }, [lastResult]);

  const inferenceFps = useMemo(() => {
    if (inferenceSamples.length < 2) return 0;
    const elapsed = inferenceSamples[inferenceSamples.length - 1] - inferenceSamples[0];
    return elapsed > 0 ? ((inferenceSamples.length - 1) * 1000) / elapsed : 0;
  }, [inferenceSamples]);

  const chooseVideo = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    if (videoUrl) URL.revokeObjectURL(videoUrl);
    setVideoUrl(URL.createObjectURL(file));
    setPlates([]);
    setHistory([]);
    setDropped(0);
  };

  const connectApi = () => {
    const normalized = apiUrl.trim().replace(/\/$/, "");
    localStorage.setItem(API_URL_KEY, normalized);
    connect(normalized, token);
  };

  const exportCsv = useCallback(() => {
    const rows = [
      ["track_id", "plate", "first_seen_ms", "last_seen_ms", "ocr_confidence"],
      ...history.map((event) => [event.trackId ?? "", event.text, event.firstSeen, event.lastSeen, event.confidence.toFixed(3)]),
    ];
    const blob = new Blob([rows.map((row) => row.join(",")).join("\n")], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "lpr-events.csv";
    anchor.click();
    URL.revokeObjectURL(url);
  }, [history]);

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">CITD / COMPUTER VISION</p>
          <h1>Realtime License Plate Monitor</h1>
          <p className="subtitle">Video local · inference GPU · overlay tức thì</p>
        </div>
        <div className={`connection-badge ${state}`}><span />{state}</div>
      </header>

      <section className="control-panel panel">
        <div className="field api-field">
          <label htmlFor="api-url">GPU API URL</label>
          <input id="api-url" value={apiUrl} onChange={(event) => setApiUrl(event.target.value)} placeholder="https://...trycloudflare.com" />
        </div>
        <div className="field token-field">
          <label htmlFor="token">Demo token</label>
          <input id="token" type="password" value={token} onChange={(event) => setToken(event.target.value)} placeholder="optional" />
        </div>
        <div className="button-row">
          <button className="primary-button" onClick={connected ? disconnect : connectApi}>{connected ? "Ngắt API" : "Kết nối GPU"}</button>
          <label className="file-button">Chọn video<input type="file" accept="video/*" onChange={chooseVideo} /></label>
        </div>
        <div className="field fps-field">
          <label htmlFor="sample-fps">Sample FPS</label>
          <input id="sample-fps" type="number" min="1" max="20" value={sampleFps} onChange={(event) => setSampleFps(Number(event.target.value))} />
        </div>
      </section>

      {error && <div className="error-banner">{error}</div>}
      <section className="runtime-strip">
        <span>{health?.model_loaded ? `Model: ${health.ocr_backends.join(", ")} / ${health.device}` : "Model chưa sẵn sàng"}</span>
        <span>{videoUrl ? "Video đã chọn" : "Chưa chọn video"}</span>
      </section>

      <section className="main-grid">
        <div className="video-column">
          <div className="video-frame">
            {videoUrl ? (
              <>
                <video ref={videoRef} src={videoUrl} controls muted playsInline />
                <OverlayCanvas plates={plates} videoRef={videoRef} />
              </>
            ) : (
              <div className="video-placeholder"><div className="play-glyph">▶</div><p>Chọn một video để bắt đầu</p><span>Video được phát local; chỉ frame mẫu gửi tới GPU API.</span></div>
            )}
          </div>
          <canvas ref={captureCanvasRef} className="capture-canvas" aria-hidden="true" />
          <RuntimeMetrics latency={latency} dropped={dropped} inferenceFps={inferenceFps} sourceFps={sampleFps} />
        </div>
        <aside className="side-column">
          <PlateTable plates={plates} />
          <section className="panel history-panel">
            <div className="panel-heading"><div><p className="eyebrow">EVENT LOG</p><h2>Lịch sử stable plate</h2></div><button className="ghost-button" onClick={exportCsv} disabled={!history.length}>Export CSV</button></div>
            {history.length === 0 ? <p className="empty-state">Stable plate sẽ xuất hiện ở đây.</p> : <div className="history-list">{history.map((event) => <div className="history-item" key={event.id}><strong>{event.text}</strong><span>Track #{event.trackId ?? "—"} · {formatTime(event.lastSeen)} · {(event.confidence * 100).toFixed(0)}%</span></div>)}</div>}
          </section>
        </aside>
      </section>
    </main>
  );
}
