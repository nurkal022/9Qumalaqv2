import { useEffect, useRef, useState, useCallback } from "react";
import type { ServerMsg, ClientMsg } from "../api/ws";

export type Status = "connecting" | "connected" | "reconnecting" | "down";

export function useGameSocket(
  gameId: number,
  onMsg: (m: ServerMsg) => void,
) {
  const [status, setStatus] = useState<Status>("connecting");
  const wsRef = useRef<WebSocket | null>(null);
  const lastSeqRef = useRef<number | undefined>(undefined);
  const retryRef = useRef(0);
  const onMsgRef = useRef(onMsg);

  // Keep latest onMsg callback without re-subscribing
  useEffect(() => {
    onMsgRef.current = onMsg;
  }, [onMsg]);

  useEffect(() => {
    let cancelled = false;
    let heartbeatInterval: ReturnType<typeof setInterval> | undefined;
    let timeoutHandle: ReturnType<typeof setTimeout> | undefined;

    function connect() {
      if (cancelled) return;
      const proto = location.protocol === "https:" ? "wss" : "ws";
      const ws = new WebSocket(`${proto}://${location.host}/ws/games/${gameId}`);
      wsRef.current = ws;

      ws.onopen = () => {
        retryRef.current = 0;
        setStatus("connected");
        const hello: ClientMsg = { type: "hello", lastSeenSeq: lastSeqRef.current };
        ws.send(JSON.stringify(hello));
      };

      ws.onmessage = (ev) => {
        try {
          const m: ServerMsg = JSON.parse(ev.data);
          if (typeof m.seq === "number") lastSeqRef.current = m.seq;
          onMsgRef.current(m);
        } catch {
          // ignore non-JSON frames
        }
      };

      ws.onclose = () => {
        if (cancelled) return;
        if (retryRef.current >= 3) {
          setStatus("down");
          return;
        }
        setStatus("reconnecting");
        const delay = 250 * 2 ** retryRef.current;
        retryRef.current += 1;
        timeoutHandle = setTimeout(connect, delay);
      };

      ws.onerror = () => {
        ws.close();
      };
    }

    connect();

    heartbeatInterval = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        const ping: ClientMsg = { type: "ping" };
        wsRef.current.send(JSON.stringify(ping));
      }
    }, 30_000);

    return () => {
      cancelled = true;
      if (heartbeatInterval) clearInterval(heartbeatInterval);
      if (timeoutHandle) clearTimeout(timeoutHandle);
      wsRef.current?.close();
    };
  }, [gameId]);

  const send = useCallback((msg: ClientMsg) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(msg));
    }
  }, []);

  return { status, send };
}
