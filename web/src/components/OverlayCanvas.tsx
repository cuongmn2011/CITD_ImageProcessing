import { useEffect, useRef, type RefObject } from "react";
import type { PlateResult } from "../types";

interface OverlayCanvasProps {
  plates: PlateResult[];
  videoRef: RefObject<HTMLVideoElement | null>;
}

export function OverlayCanvas({ plates, videoRef }: OverlayCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    const container = canvas?.parentElement;
    if (!canvas || !video || !container) return;

    const draw = () => {
      const containerRect = container.getBoundingClientRect();
      const videoRect = video.getBoundingClientRect();
      const width = Math.max(1, videoRect.width);
      const height = Math.max(1, videoRect.height);
      const scale = window.devicePixelRatio || 1;
      canvas.style.left = `${videoRect.left - containerRect.left}px`;
      canvas.style.top = `${videoRect.top - containerRect.top}px`;
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      canvas.width = Math.round(width * scale);
      canvas.height = Math.round(height * scale);
      const context = canvas.getContext("2d");
      if (!context) return;
      context.setTransform(scale, 0, 0, scale, 0, 0);
      context.clearRect(0, 0, width, height);
      context.font = "600 13px ui-monospace, monospace";
      plates.forEach((plate) => {
        const [x1, y1, x2, y2] = plate.bbox_norm;
        const left = x1 * width;
        const top = y1 * height;
        const boxWidth = (x2 - x1) * width;
        const boxHeight = (y2 - y1) * height;
        const color = plate.stable
          ? "#39e58c"
          : plate.status === "candidate"
            ? "#ffd166"
            : "#ff6b6b";
        context.strokeStyle = color;
        context.lineWidth = 2;
        context.strokeRect(left, top, boxWidth, boxHeight);
        const label = plate.text || "detecting...";
        const labelWidth = context.measureText(label).width + 12;
        const labelTop = Math.max(0, top - 24);
        context.fillStyle = color;
        context.fillRect(left, labelTop, labelWidth, 22);
        context.fillStyle = "#08111f";
        context.fillText(label, left + 6, Math.max(15, labelTop + 16));
      });
    };

    draw();
    const observer = new ResizeObserver(draw);
    observer.observe(container);
    observer.observe(video);
    window.addEventListener("resize", draw);
    return () => {
      observer.disconnect();
      window.removeEventListener("resize", draw);
    };
  }, [plates, videoRef]);

  return <canvas ref={canvasRef} className="overlay-canvas" aria-label="Detection overlay" />;
}
