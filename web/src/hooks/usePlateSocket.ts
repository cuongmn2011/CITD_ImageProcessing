import { useCallback, useEffect, useRef, useState } from "react";
import type { ConnectionState, FrameResult, RuntimeHealth } from "../types";

interface SocketHook {
  state: ConnectionState;
  lastResult: FrameResult | null;
  health: RuntimeHealth | null;
  error: string;
  connect: (baseUrl: string, token?: string) => void;
  disconnect: () => void;
  sendConfig: (width: number, height: number) => void;
  sendFrame: (frame: Blob, frameId: number, sourceTimeMs: number, width: number, height: number) => boolean;
  canSendFrame: boolean;
}

function websocketUrl(baseUrl: string, token: string): string {
  const url = new URL(baseUrl.replace(/^http/, "ws"));
  url.pathname = `${url.pathname.replace(/\/$/, "")}/ws/stream`;
  if (token.trim()) url.searchParams.set("token", token.trim());
  return url.toString();
}

export function usePlateSocket(): SocketHook {
  const socketRef = useRef<WebSocket | null>(null);
  const inFlightRef = useRef(false);
  const [state, setState] = useState<ConnectionState>("disconnected");
  const [lastResult, setLastResult] = useState<FrameResult | null>(null);
  const [health, setHealth] = useState<RuntimeHealth | null>(null);
  const [error, setError] = useState("");

  const disconnect = useCallback(() => {
    socketRef.current?.close();
    socketRef.current = null;
    inFlightRef.current = false;
    setState("disconnected");
  }, []);

  const connect = useCallback((baseUrl: string, token = "") => {
    disconnect();
    let healthUrl: string;
    try {
      const parsed = new URL(baseUrl);
      parsed.pathname = `${parsed.pathname.replace(/\/$/, "")}/api/health`;
      parsed.protocol = parsed.protocol === "https:" ? "https:" : "http:";
      healthUrl = parsed.toString();
      setState("connecting");
      setError("");
      fetch(healthUrl)
        .then(async (response) => {
          if (!response.ok) throw new Error(`Health check failed (${response.status})`);
          setHealth((await response.json()) as RuntimeHealth);
        })
        .catch((healthError: unknown) => {
          setError(healthError instanceof Error ? healthError.message : "Health check failed");
        });
      const socket = new WebSocket(websocketUrl(baseUrl, token));
      socket.binaryType = "arraybuffer";
      socket.onopen = () => {
        socketRef.current = socket;
        setState("connected");
      };
      socket.onmessage = (event) => {
        try {
          const payload = JSON.parse(String(event.data)) as { type: string; message?: string };
          if (payload.type === "result") {
            inFlightRef.current = false;
            setLastResult(payload as FrameResult);
          } else if (payload.type === "error") {
            inFlightRef.current = false;
            setError(payload.message ?? "Inference error");
          }
        } catch {
          inFlightRef.current = false;
          setError("Invalid response from inference API");
        }
      };
      socket.onerror = () => {
        inFlightRef.current = false;
        setState("error");
        setError("WebSocket connection failed");
      };
      socket.onclose = () => {
        inFlightRef.current = false;
        socketRef.current = null;
        setState((current) => (current === "error" ? current : "disconnected"));
      };
    } catch (connectionError: unknown) {
      setState("error");
      setError(connectionError instanceof Error ? connectionError.message : "Invalid API URL");
    }
  }, [disconnect]);

  const sendConfig = useCallback((width: number, height: number) => {
    if (socketRef.current?.readyState !== WebSocket.OPEN) return;
    socketRef.current.send(JSON.stringify({ type: "config", source_width: width, source_height: height }));
  }, []);

  const sendFrame = useCallback(
    (frame: Blob, frameId: number, sourceTimeMs: number, width: number, height: number) => {
      const socket = socketRef.current;
      if (!socket || socket.readyState !== WebSocket.OPEN || inFlightRef.current) return false;
      inFlightRef.current = true;
      socket.send(JSON.stringify({ type: "frame_meta", frame_id: frameId, source_time_ms: sourceTimeMs, width, height }));
      socket.send(frame);
      return true;
    },
    [],
  );

  useEffect(() => disconnect, [disconnect]);

  return {
    state,
    lastResult,
    health,
    error,
    connect,
    disconnect,
    sendConfig,
    sendFrame,
    canSendFrame: state === "connected" && !inFlightRef.current,
  };
}
