import { api } from "./client";
import type { GameState } from "./play";

export type GameSummary = {
  id: number;
  mode: string;
  opponentLabel: string;
  result: string | null;
  finalScore: string | null;
  side: number;
  moveCount: number;
  startedAt: string;
  finishedAt: string | null;
  durationMs: number | null;
};

export const gamesApi = {
  list: (params: { page?: number; pageSize?: number; status?: string } = {}) => {
    const qs = new URLSearchParams(
      Object.entries(params).filter(([, v]) => v !== undefined) as [string, string][],
    ).toString();
    return api<{ items: GameSummary[]; total: number; page: number; pageSize: number }>(
      `/api/games${qs ? `?${qs}` : ""}`,
    );
  },
  get: (id: number) => api<{ game: GameState }>(`/api/games/${id}`),
  delete: (id: number) => api<void>(`/api/games/${id}`, { method: "DELETE" }),
};
