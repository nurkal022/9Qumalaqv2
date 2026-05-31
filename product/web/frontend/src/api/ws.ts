import type { GameState } from "./play";

export type ServerMsg = { seq: number } & (
  | { type: "snapshot"; game: GameState }
  | { type: "move_applied"; ply: number; side: 0 | 1; moveUci: string; fenAfter: string; clock: any }
  | { type: "engine_thinking"; started: boolean; sinceMs?: number }
  | { type: "engine_move"; ply: number; side: 0 | 1; moveUci: string; fenAfter: string; evalCp: number | null; evalDepth: number | null; thinkMs: number; clock: any }
  | { type: "event"; event: { type: string; actor: string; payload?: unknown } }
  | { type: "game_finished"; result: string; resultReason: string; finalScore: string }
  | { type: "pong" }
  | { type: "error"; code: string; messageKk: string; messageRu: string; fatal: boolean }
);

export type ClientMsg =
  | { type: "hello"; lastSeenSeq?: number }
  | { type: "move"; moveUci: string }
  | { type: "request_snapshot" }
  | { type: "cancel_thinking" }
  | { type: "ping" };
