import { Link } from "react-router";
import { Trophy, Frown, Minus, Clock as ClockIcon } from "lucide-react";
import type { GameSummary } from "../../api/games";

type Outcome = "win" | "loss" | "draw" | "active";

function classify(g: GameSummary, mySide: number): Outcome {
  if (!g.result) return "active";
  if (g.result === "draw") return "draw";
  return g.result === (mySide === 0 ? "win_white" : "win_black") ? "win" : "loss";
}

const OUTCOME_STYLE: Record<Outcome, { bg: string; ring: string; icon: React.ReactNode; label: string }> = {
  win: {
    bg: "bg-state-win/10",
    ring: "ring-state-win/40",
    icon: <Trophy size={16} className="text-state-win" />,
    label: "W",
  },
  loss: {
    bg: "bg-state-loss/10",
    ring: "ring-state-loss/30",
    icon: <Frown size={16} className="text-state-loss" />,
    label: "L",
  },
  draw: {
    bg: "bg-state-draw/10",
    ring: "ring-state-draw/30",
    icon: <Minus size={16} className="text-state-draw" />,
    label: "½",
  },
  active: {
    bg: "bg-bg-raised",
    ring: "ring-accent-gold/30",
    icon: <ClockIcon size={16} className="text-accent-gold animate-pulse" />,
    label: "…",
  },
};

export default function GameRow({ g, mySide }: { g: GameSummary; mySide: number }) {
  const outcome = classify(g, mySide);
  const style = OUTCOME_STYLE[outcome];
  const date = new Date(g.startedAt);
  const dateStr = date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
  const timeStr = date.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });

  return (
    <Link
      to={`/replay/${g.id}`}
      className={`group flex items-center gap-3 p-3 rounded-lg ${style.bg} hover:ring-1 ${style.ring} transition`}
    >
      <span
        className={`shrink-0 w-9 h-9 rounded-full flex items-center justify-center font-mono font-bold text-sm ${style.bg} ring-1 ${style.ring}`}
      >
        {style.label}
      </span>

      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 text-fg-primary truncate">
          {style.icon}
          <span className="truncate">{g.opponentLabel}</span>
        </div>
        <div className="text-xs text-fg-muted mt-0.5 flex items-center gap-2">
          <span>{dateStr}, {timeStr}</span>
          {g.moveCount > 0 && <span className="text-fg-muted">· {g.moveCount} ply</span>}
        </div>
      </div>

      <span className="font-mono text-fg-secondary text-sm tabular-nums">
        {g.finalScore ?? "—"}
      </span>
    </Link>
  );
}
