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
}

function apiOrigin(baseUrl: string): URL {
  const url = new URL(baseUrl);
  url.pathname = "/";
  url.search = "";
  url.hash = "";
  return url;
}

function websocketUrl(baseUrl: string, token: string): string {
  const url = apiOrigin(baseUrl);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  url.pathname = "/ws/stream";
  if (token.trim()) url.searchParams.set("token", token.trim());
  return url.toString();
}

export function usePlateSocket(): SocketHook {
  const socketRef = useRef<WebSocket | null>(null);
  const healthAbortRef = useRef<AbortController | null>(null);
  const generationRef = useRef(0);
  const inFlightRef = useRef(false);
  const [state, setState] = useState<ConnectionState>("disconnected");
  const [lastResult, setLastResult] = useState<FrameResult | null>(null);
  const [health, setHealth] = useState<RuntimeHealth | null>(null);
  const [error, setError] = useState("");

  const disconnect = useCallback(() => {
    generationRef.current += 1;
    healthAbortRef.current?.abort();
    healthAbortRef.current = null;
    socketRef.current?.close();
    socketRef.current = null;
    inFlightRef.current = false;
    setState("disconnected");
  }, []);

  const connect = useCallback((baseUrl: string, token = "") => {
    disconnect();
    const generation = generationRef.current;
    try {
      const origin = apiOrigin(baseUrl);
      origin.protocol = origin.protocol === "https:" ? "https:" : "http:";
      const healthUrl = new URL("/api/health", origin).toString();
      const controller = new AbortController();
      healthAbortRef.current = controller;
      setState("connecting");
      setError("");
      fetch(healthUrl, { signal: controller.signal })
        .then(async (response) => {
          if (!response.ok) throw new Error(`Health check failed (${response.status})`);
          if (generationRef.current !== generation) return;
          setHealth((await response.json()) as RuntimeHealth);
        })
        .catch((healthError: unknown) => {
          if (healthError instanceof DOMException && healthError.name === "AbortError") return;
          if (generationRef.current !== generation) return;
          setError(healthError instanceof Error ? healthError.message : "Health check failed");
        });

      const socket = new WebSocket(websocketUrl(baseUrl, token));
      socket.binaryType = "arraybuffer";
      socket.onopen = () => {
        if (generationRef.current !== generation) {
          socket.close();
          return;
        }
        socketRef.current = socket;
        setState("connected");
      };
      socket.onmessage = (event) => {
        if (generationRef.current !== generation) return;
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
        if (generationRef.current !== generation) return;
        inFlightRef.current = false;
        setState("error");
        setError("WebSocket connection failed");
      };
      socket.onclose = () => {
        if (generationRef.current !== generation) return;
        inFlightRef.current = false;
        if (socketRef.current === socket) socketRef.current = null;
        setState((current) => (current === "error" ? current : "disconnected"));
      };
    } catch (connectionError: unknown) {
      if (generationRef.current !== generation) return;
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

  return { state, lastResult, health, error, connect, disconnect, sendConfig, sendFrame };
}
