import { api } from "./client";

export type UserOut = {
  id: number;
  username: string;
  displayName?: string | null;
  locale: string;
};
export type Me =
  | { kind: "user"; user: UserOut }
  | { kind: "anon"; anonId: string };

export const authApi = {
  me: () => api<Me>("/api/auth/me"),
  register: (body: { username: string; password: string; locale?: string }) =>
    api<{ user: UserOut; migratedGamesCount: number }>("/api/auth/register", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  login: (body: { username: string; password: string }) =>
    api<{ user: UserOut; migratedGamesCount: number }>("/api/auth/login", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  logout: () => api<void>("/api/auth/logout", { method: "POST" }),
};
