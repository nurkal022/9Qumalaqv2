import { useQuery } from "@tanstack/react-query";
import { gamesApi } from "../api/games";
import GameRow from "../components/history/GameRow";
import { useTranslation } from "react-i18next";

export default function History() {
  const { t } = useTranslation();
  const { data, isLoading } = useQuery({
    queryKey: ["games", { page: 1 }],
    queryFn: () => gamesApi.list({ page: 1, pageSize: 20 }),
  });
  if (isLoading || !data) return <p className="p-4">{t("common.loading")}</p>;
  return (
    <div className="p-4 max-w-screen-md mx-auto space-y-2">
      <h2 className="text-2xl mb-3">{t("history.title")}</h2>
      {data.items.map((g) => <GameRow key={g.id} g={g} mySide={g.side} />)}
      {data.items.length === 0 && <p className="text-fg-secondary">{t("history.empty")}</p>}
    </div>
  );
}
