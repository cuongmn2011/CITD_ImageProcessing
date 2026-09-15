interface RuntimeMetricsProps {
  latency: number;
  dropped: number;
  inferenceFps: number;
  sourceFps: number;
}

export function RuntimeMetrics({ latency, dropped, inferenceFps, sourceFps }: RuntimeMetricsProps) {
  return (
    <section className="metrics-grid" aria-label="Runtime metrics">
      <div className="metric"><span>Round-trip</span><strong>{latency ? `${latency.toFixed(0)} ms` : "—"}</strong></div>
      <div className="metric"><span>Inference FPS</span><strong>{inferenceFps ? inferenceFps.toFixed(1) : "—"}</strong></div>
      <div className="metric"><span>Target FPS</span><strong>{sourceFps ? sourceFps.toFixed(1) : "—"}</strong></div>
      <div className="metric"><span>Dropped</span><strong>{dropped}</strong></div>
    </section>
  );
}
