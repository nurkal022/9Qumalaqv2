import { useParams } from "react-router";
import { useTranslation } from "react-i18next";
import { Settings } from "lucide-react";
import { useGameSession } from "../hooks/useGameSession";
import { useUI } from "../stores/ui";
import Board from "../components/board/Board";
import EvalBar from "../components/play/EvalBar";
import Clock from "../components/play/Clock";
import EngineStatus from "../components/play/EngineStatus";
import MoveList from "../components/play/MoveList";
import GameControls from "../components/play/GameControls";

/**
 * Desktop / wide-screen layout: nameplate-flanked board with eval bar above,
 * clocks on each side, move list collapsed below, settings + controls on a row.
 */
export default function GameDesktop() {
  const { t } = useTranslation();
  const { id } = useParams<{ id: string }>();
  const gid = Number(id);
  const openSettings = useUI((s) => s.openSettings);

  const session = useGameSession(gid);

  if (session.isLoading || !session.g || !session.layout) {
    return <p className="p-4">{t("common.loading")}</p>;
  }
  const g = session.g;
  const myClockMs = g.side === 0 ? g.clock.whiteMs : g.clock.blackMs;
  const oppClockMs = g.side === 0 ? g.clock.blackMs : g.clock.whiteMs;
  const myRunning = g.clock.runningSide === g.side && g.status === "active";
  const oppRunning =
    g.clock.runningSide !== g.side && g.clock.runningSide !== null && g.status === "active";
  const lastEval = g.moves.length > 0 ? g.moves[g.moves.length - 1].evalCp : null;

  return (
    <div className="p-3 max-w-screen-md mx-auto space-y-3">
      <div className="flex justify-between items-center">
        <EngineStatus thinking={g.engineThinking} />
        <Clock ms={oppClockMs} running={oppRunning} />
      </div>

      <EvalBar cp={lastEval} perspectiveSide={g.side} />

      <Board
        layout={session.layout}
        side={g.side}
        sideToMove={g.sideToMove}
        onMove={session.onMove}
        disabled={session.boardDisabled}
        activePit={session.activePit}
      />

      <div className="flex justify-between items-center gap-2">
        <Clock ms={myClockMs} running={myRunning} />
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={openSettings}
            aria-label={t("settings.title")}
            className="bg-bg-raised hover:bg-bg-inset transition rounded p-2 text-fg-secondary"
          >
            <Settings size={18} />
          </button>
          <GameControls
            hintsUsed={g.hintsUsed}
            hintsLimit={g.hintsLimit}
            disabled={session.actionsDisabled}
            onResign={session.doResign}
            onDraw={session.doDraw}
            onUndo={session.doUndo}
            onHint={session.doHint}
          />
        </div>
      </div>

      <details>
        <summary className="text-fg-secondary cursor-pointer">{t("game.moveList")}</summary>
        <MoveList moves={g.moves} />
      </details>

      {session.wsStatus !== "connected" && (
        <p className="text-sm text-state-warn">
          {t("game.wsStatus", { status: session.wsStatus })}
        </p>
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
