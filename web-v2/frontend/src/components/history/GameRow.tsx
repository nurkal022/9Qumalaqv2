import { Link } from "react-router";
import type { GameSummary } from "../../api/games";

const TONE: Record<string, string> = {
  win_white: "bg-state-win/10",
  win_black: "bg-state-win/10",
  draw: "bg-state-draw/10",
};

export default function GameRow({ g, mySide }: { g: GameSummary; mySide: number }) {
  // Determine if this user won, lost, or drew
  let label = "";
  if (g.result) {
    if (g.result === "draw") label = "draw";
    else if (g.result === (mySide === 0 ? "win_white" : "win_black")) label = "win";
    else label = "loss";
  }
  return (
    <Link to={`/replay/${g.id}`} className={`flex items-center justify-between p-3 rounded ${TONE[g.result ?? ""] ?? "bg-bg-raised"}`}>
      <span className="text-fg-primary">{g.opponentLabel}</span>
      <span className="text-sm text-fg-secondary">{g.finalScore ?? "—"}</span>
      <span className="text-sm text-fg-muted">{new Date(g.startedAt).toLocaleString()}</span>
    </Link>
  );
}
