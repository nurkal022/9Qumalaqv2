import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router";
import { Inbox, Play } from "lucide-react";
import { gamesApi } from "../api/games";
import GameRow from "../components/history/GameRow";
import { useTranslation } from "react-i18next";

export default function History() {
  const { t } = useTranslation();
  const { data, isLoading } = useQuery({
    queryKey: ["games", { page: 1 }],
    queryFn: () => gamesApi.list({ page: 1, pageSize: 30 }),
  });

  return (
    <div className="p-4 max-w-screen-md mx-auto space-y-3">
      <header className="flex items-end justify-between mb-1">
        <h2 className="text-2xl font-semibold">{t("history.title")}</h2>
        {data && data.total > 0 && (
          <span className="text-sm text-fg-muted font-mono">{data.total}</span>
        )}
      </header>

      {isLoading && <HistorySkeleton />}

      {!isLoading && data && data.items.length === 0 && <HistoryEmpty />}

      {!isLoading && data && data.items.length > 0 && (
        <ul className="space-y-2">
          {data.items.map((g) => (
            <li key={g.id}>
              <GameRow g={g} mySide={g.side} />
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function HistorySkeleton() {
  return (
    <ul className="space-y-2" aria-hidden="true">
      {Array.from({ length: 4 }).map((_, i) => (
        <li
          key={i}
          className="h-14 bg-bg-raised rounded-lg animate-pulse"
        />
      ))}
    </ul>
  );
}

function HistoryEmpty() {
  const { t } = useTranslation();
  return (
    <div className="bg-bg-raised rounded-xl py-12 px-6 flex flex-col items-center text-center gap-3 mt-4">
      <Inbox size={42} strokeWidth={1.4} className="text-fg-muted" />
      <p className="text-fg-secondary">{t("history.empty")}</p>
      <Link
        to="/lobby"
        className="mt-2 inline-flex items-center gap-2 bg-accent-amber text-bg-base px-4 py-2 rounded-lg font-medium"
      >
        <Play size={16} fill="currentColor" />
        {t("lobby.newGame")}
      </Link>
    </div>
  );
}
