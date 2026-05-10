import { useParams } from "react-router";
import { useCallback } from "react";
import { useGameQuery } from "../hooks/useGameQuery";
import { useGameSocket } from "../hooks/useGameSocket";
import { playApi } from "../api/play";
import type { ServerMsg } from "../api/ws";
import Board from "../components/board/Board";
import EvalBar from "../components/play/EvalBar";
import Clock from "../components/play/Clock";
import EngineStatus from "../components/play/EngineStatus";
import MoveList from "../components/play/MoveList";
import GameControls from "../components/play/GameControls";
import { parsePos } from "../domain/board";
import { useTranslation } from "react-i18next";

export default function Game() {
  const { t } = useTranslation();
  const { id } = useParams<{ id: string }>();
  const gid = Number(id);
  const { data: g, isLoading, applyServerSnapshot, refetch } = useGameQuery(gid);

  const onWsMsg = useCallback((m: ServerMsg) => {
    if (m.type === "snapshot") {
      applyServerSnapshot(m.game);
    } else if (m.type === "engine_move" || m.type === "move_applied" || m.type === "game_finished" || m.type === "engine_thinking") {
      refetch();
    }
  }, [applyServerSnapshot, refetch]);

  const { status } = useGameSocket(gid, onWsMsg);

  if (isLoading || !g) return <p className="p-4">{t("common.loading")}</p>;

  const layout = parsePos(g.currentFen);
  const myClockMs = g.side === 0 ? g.clock.whiteMs : g.clock.blackMs;
  const oppClockMs = g.side === 0 ? g.clock.blackMs : g.clock.whiteMs;
  const myRunning = g.clock.runningSide === g.side && g.status === "active";
  const oppRunning = g.clock.runningSide !== g.side && g.clock.runningSide !== null && g.status === "active";
  const lastEval = g.moves.length > 0 ? g.moves[g.moves.length - 1].evalCp : null;

  async function onMove(pitIndex: number) {
    try {
      const r = await playApi.move(gid, String(pitIndex));
      applyServerSnapshot(r.game);
    } catch (e: any) {
      // Toast handled by global error wiring (Task 25); for now log
      console.error("move failed", e);
    }
  }

  async function doResign() { const r = await playApi.resign(gid); applyServerSnapshot(r.game); }
  async function doDraw() { const r = await playApi.drawOffer(gid); applyServerSnapshot(r.game); }
  async function doUndo() { const r = await playApi.undo(gid); applyServerSnapshot(r.game); }
  async function doHint() { await playApi.hint(gid); refetch(); }

  return (
    <div className="p-3 max-w-screen-md mx-auto space-y-3">
      <div className="flex justify-between items-center">
        <EngineStatus thinking={g.engineThinking} />
        <Clock ms={oppClockMs} running={oppRunning} />
      </div>

      <EvalBar cp={lastEval} perspectiveSide={g.side} />

      <Board
        layout={layout}
        side={g.side}
        sideToMove={g.sideToMove}
        onMove={onMove}
        disabled={g.status !== "active" || g.engineThinking}
      />

      <div className="flex justify-between items-center">
        <Clock ms={myClockMs} running={myRunning} />
        <GameControls
          hintsUsed={g.hintsUsed}
          hintsLimit={g.hintsLimit}
          disabled={g.status !== "active"}
          onResign={doResign}
          onDraw={doDraw}
          onUndo={doUndo}
          onHint={doHint}
        />
      </div>

      <details>
        <summary className="text-fg-secondary cursor-pointer">{t("game.moveList")}</summary>
        <MoveList moves={g.moves} />
      </details>

      {status !== "connected" && (
        <p className="text-sm text-state-warn">{t("game.wsStatus", { status })}</p>
      )}

      {g.status === "finished" && (
        <div className="bg-bg-raised rounded p-3 text-center">
          <p className="text-lg">{t(`game.result.${g.result}`)}</p>
          {g.finalScore && <p className="text-fg-secondary">{g.finalScore}</p>}
        </div>
      )}
    </div>
  );
}
