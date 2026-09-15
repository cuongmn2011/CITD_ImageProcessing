import { useEffect, type RefObject } from "react";

interface FrameSenderOptions {
  videoRef: RefObject<HTMLVideoElement | null>;
  canvasRef: RefObject<HTMLCanvasElement | null>;
  enabled: boolean;
  sampleFps: number;
  sendConfig: (width: number, height: number) => void;
  sendFrame: (frame: Blob, frameId: number, sourceTimeMs: number, width: number, height: number) => boolean;
}

export function useFrameSender({
  videoRef,
  canvasRef,
  enabled,
  sampleFps,
  sendConfig,
  sendFrame,
}: FrameSenderOptions): void {
  useEffect(() => {
    if (!enabled) return;
    let animationFrame = 0;
    let frameId = 0;
    let lastSentAt = 0;
    let configured = false;
    const interval = 1000 / Math.max(1, sampleFps);

    const capture = (now: number) => {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (video && canvas && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
        if (!configured && video.videoWidth > 0 && video.videoHeight > 0) {
          sendConfig(video.videoWidth, video.videoHeight);
          configured = true;
        }
        if (!video.paused && !video.ended && now - lastSentAt >= interval) {
          const width = Math.min(video.videoWidth, 960);
          const height = Math.max(1, Math.round((video.videoHeight / video.videoWidth) * width));
          canvas.width = width;
          canvas.height = height;
          const context = canvas.getContext("2d");
          if (context) {
            context.drawImage(video, 0, 0, width, height);
            canvas.toBlob((blob) => {
              if (blob && sendFrame(blob, frameId++, video.currentTime * 1000, width, height)) {
                lastSentAt = now;
              }
            }, "image/jpeg", 0.75);
          }
        }
      }
      animationFrame = requestAnimationFrame(capture);
    };

    animationFrame = requestAnimationFrame(capture);
    return () => cancelAnimationFrame(animationFrame);
  }, [canvasRef, enabled, sampleFps, sendConfig, sendFrame, videoRef]);
}
