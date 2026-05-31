import { useParams } from "react-router";
import { useQuery } from "@tanstack/react-query";
import { gamesApi } from "../api/games";
import { useState } from "react";
import Board from "../components/board/Board";
import { parsePos } from "../domain/board";
import { useTranslation } from "react-i18next";

export default function Replay() {
  const { t } = useTranslation();
  const { id } = useParams<{ id: string }>();
  const gid = Number(id);
  const { data, isLoading } = useQuery({
    queryKey: ["replay", gid],
    queryFn: () => gamesApi.get(gid).then((r) => r.game),
  });
  const [ply, setPly] = useState(0);

  if (isLoading || !data) return <p className="p-4">{t("common.loading")}</p>;

  const fen = ply === 0 ? data.startFen : data.moves[ply - 1].fenAfter;
  const layout = parsePos(fen);
  const evalCp = ply > 0 ? data.moves[ply - 1].evalCp : null;

  return (
    <div className="p-3 max-w-screen-md mx-auto space-y-3">
      <Board layout={layout} side={data.side} sideToMove={(ply % 2) as 0 | 1} onMove={() => {}} disabled />
      <div className="flex justify-between items-center">
        <button onClick={() => setPly(0)} className="bg-bg-raised px-3 py-2 rounded">{"|<"}</button>
        <button onClick={() => setPly((p) => Math.max(0, p - 1))} className="bg-bg-raised px-3 py-2 rounded">{"<"}</button>
        <span className="font-mono">{ply}/{data.moves.length}</span>
        <button onClick={() => setPly((p) => Math.min(data.moves.length, p + 1))} className="bg-bg-raised px-3 py-2 rounded">{">"}</button>
        <button onClick={() => setPly(data.moves.length)} className="bg-bg-raised px-3 py-2 rounded">{">|"}</button>
      </div>
      <p className="text-sm text-fg-secondary">eval: {evalCp != null ? `${evalCp > 0 ? "+" : ""}${(evalCp / 100).toFixed(2)}` : "—"}</p>
      {data.status === "finished" && data.finalScore && (
        <p className="text-fg-secondary">{t(`game.result.${data.result}`)} ({data.finalScore})</p>
      )}
    </div>
  );
}
