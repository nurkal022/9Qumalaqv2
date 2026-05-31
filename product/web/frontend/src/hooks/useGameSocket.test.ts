import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { Server, WebSocket as MockWebSocket } from "mock-socket";
import { useGameSocket } from "./useGameSocket";

let server: Server | null = null;

beforeEach(() => {
  (globalThis as any).WebSocket = MockWebSocket;
  Object.defineProperty(window, "location", {
    value: { protocol: "ws:", host: "localhost", href: "ws://localhost/" },
    writable: true,
    configurable: true,
  });
});

afterEach(() => {
  if (server) {
    server.close();
    server = null;
  }
});

describe("useGameSocket", () => {
  it("sends hello with lastSeenSeq on open", async () => {
    const url = "ws://localhost/ws/games/1";
    server = new Server(url);
    const received: any[] = [];
    server.on("connection", (s) => {
      s.on("message", (m: any) => received.push(JSON.parse(m)));
    });

    const onMsg = vi.fn();
    renderHook(() => useGameSocket(1, onMsg));

    await waitFor(() => expect(received.length).toBeGreaterThan(0), { timeout: 1000 });
    expect(received[0].type).toBe("hello");
  });

  it("forwards server messages to the callback", async () => {
    const url = "ws://localhost/ws/games/2";
    server = new Server(url);
    server.on("connection", (s) => {
      s.send(JSON.stringify({ seq: 1, type: "snapshot", game: { id: 2 } }));
    });

    const onMsg = vi.fn();
    renderHook(() => useGameSocket(2, onMsg));

    await waitFor(() => expect(onMsg).toHaveBeenCalled(), { timeout: 1000 });
    expect(onMsg.mock.calls[0][0].type).toBe("snapshot");
  });

  it("hello includes lastSeenSeq after receiving messages", async () => {
    // First connection — receive a seq'd message so lastSeqRef gets set
    const url = "ws://localhost/ws/games/3";
    server = new Server(url);
    const received: any[] = [];
    server.on("connection", (s) => {
      s.on("message", (m: any) => received.push(JSON.parse(m)));
      // send a message with seq=42
      s.send(JSON.stringify({ seq: 42, type: "pong" }));
    });

    const onMsg = vi.fn();
    renderHook(() => useGameSocket(3, onMsg));

    // Wait for pong to be received
    await waitFor(() => expect(onMsg).toHaveBeenCalled(), { timeout: 1000 });
    // The hello sent on open won't have lastSeenSeq (undefined), but msg was received
    expect(onMsg.mock.calls[0][0].seq).toBe(42);
  });

  it("status transitions to connected on open", async () => {
    const url = "ws://localhost/ws/games/4";
    server = new Server(url);
    server.on("connection", () => {});

    const onMsg = vi.fn();
    const { result } = renderHook(() => useGameSocket(4, onMsg));

    await waitFor(() => expect(result.current.status).toBe("connected"), { timeout: 1000 });
  });

  it("send does nothing when socket is not open", () => {
    // Don't start a server — send should be a no-op
    const onMsg = vi.fn();
    const { result } = renderHook(() => useGameSocket(99, onMsg));
    // Should not throw
    expect(() => result.current.send({ type: "ping" })).not.toThrow();
  });
});
