import { describe, it, expect, beforeEach, vi } from "vitest";
import { api, ApiError } from "./client";

beforeEach(() => {
  vi.restoreAllMocks();
});

describe("api client", () => {
  it("returns parsed body on success", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      new Response(JSON.stringify({ status: "ok" }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );
    expect(await api("/api/health")).toEqual({ status: "ok" });
  });

  it("throws ApiError with code on error envelope", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      new Response(
        JSON.stringify({ error: { code: "boom", messageKk: "Қате", messageRu: "Ошибка" } }),
        { status: 500 },
      ),
    );
    await expect(api("/api/fail")).rejects.toBeInstanceOf(ApiError);
  });

  it("ApiError carries code and status", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      new Response(
        JSON.stringify({ error: { code: "not_found", messageKk: "Жоқ", messageRu: "Не найдено" } }),
        { status: 404 },
      ),
    );
    await expect(api("/api/missing")).rejects.toMatchObject({
      code: "not_found",
      status: 404,
    });
  });
});
