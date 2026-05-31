import { toast } from "sonner";

export class ApiError extends Error {
  constructor(
    public code: string,
    public status: number,
    public messageKk: string,
    public messageRu: string,
    public details?: unknown,
  ) {
    super(code);
  }
}

export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const res = await fetch(path, {
    ...init,
    credentials: "include",
    headers: { "Content-Type": "application/json", ...(init.headers || {}) },
  });
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    const e = body?.error;
    const apiError = new ApiError(
      e?.code ?? "unknown",
      res.status,
      e?.messageKk ?? "Қате",
      e?.messageRu ?? "Ошибка",
      e?.details,
    );
    const silent = ["auth_required", "validation_failed", "username_taken", "invalid_credentials"];
    if (!silent.includes(apiError.code)) {
      const locale = (typeof localStorage !== "undefined" ? localStorage.getItem("locale") : null) ?? "kk";
      toast.error(locale === "ru" ? apiError.messageRu : apiError.messageKk);
    }
    throw apiError;
  }
  if (res.status === 204) return undefined as T;
  return res.json();
}
