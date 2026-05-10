import { api } from "./client";

export type Clock = {
  initialMs: number;
  incrementMs: number;
  whiteMs: number;
  blackMs: number;
  runningSide: 0 | 1 | null;
};

export type Move = {
  ply: number;
  side: 0 | 1;
  actor: "human" | "engine" | "book";
  moveUci: string;
  fenAfter: string;
  evalCp?: number | null;
  evalDepth?: number | null;
  thinkTimeMs?: number | null;
  clockAfterMs?: number | null;
};

export type GameEvent = {
  plyAt: number;
  actor: string;
  type: string;
  payload?: unknown;
};

export type GameState = {
  id: number;
  mode: string;
  side: 0 | 1;
  status: "active" | "finished" | "aborted";
  result?: "win_white" | "win_black" | "draw" | null;
  resultReason?: string | null;
  finalScore?: string | null;
  startFen: string;
  currentFen: string;
  currentPly: number;
  sideToMove: 0 | 1;
  clock: Clock;
  engineThinking: boolean;
  hintsUsed: number;
  hintsLimit: number;
  moves: Move[];
  events: GameEvent[];
  startedAt: string;
  finishedAt?: string | null;
};

export type NewGameReq = {
  side: 0 | 1;
  engineLevel: "easy" | "normal" | "hard";
  clock: { initialMs: number; incrementMs: number } | null;
  useBook: boolean;
};

export const playApi = {
  new: (body: NewGameReq) =>
    api<{ game: GameState }>("/api/play/new", { method: "POST", body: JSON.stringify(body) }),
  get: (id: number) => api<{ game: GameState }>(`/api/play/${id}`),
  move: (id: number, moveUci: string) =>
    api<{ game: GameState }>(`/api/play/${id}/move`, {
      method: "POST",
      body: JSON.stringify({ moveUci }),
    }),
  undo: (id: number) =>
    api<{ game: GameState }>(`/api/play/${id}/undo`, { method: "POST" }),
  takeback: (id: number, toPly: number) =>
    api<{ game: GameState }>(`/api/play/${id}/takeback`, {
      method: "POST",
      body: JSON.stringify({ toPly }),
    }),
  resign: (id: number) =>
    api<{ game: GameState }>(`/api/play/${id}/resign`, { method: "POST" }),
  drawOffer: (id: number) =>
    api<{ game: GameState; accepted: boolean }>(`/api/play/${id}/draw_offer`, {
      method: "POST",
    }),
  hint: (id: number) =>
    api<{ move: string; evalCp: number | null; depth: number | null; pv: string[] }>(
      `/api/play/${id}/hint`,
      { method: "POST" },
    ),
};
