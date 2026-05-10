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
    throw new ApiError(
      e?.code ?? "unknown",
      res.status,
      e?.messageKk ?? "Қате",
      e?.messageRu ?? "Ошибка",
      e?.details,
    );
  }
  if (res.status === 204) return undefined as T;
  return res.json();
}
