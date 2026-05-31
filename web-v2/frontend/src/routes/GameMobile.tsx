import { Link, useParams } from "react-router";
import { useTranslation } from "react-i18next";
import { motion, AnimatePresence } from "framer-motion";
import { ArrowLeft, Settings, Flag, HelpCircle, Undo2, Handshake } from "lucide-react";
import { useGameSession } from "../hooks/useGameSession";
import { useUI } from "../stores/ui";
import BoardMobile from "../components/board/BoardMobile";

/**
 * Mobile / portrait layout. Strip everything except the board.
 *
 *   [← back]   78 — 78   [⚙ gear]
 *   ───────────────────────────────
 *
 *   <board fills the whole vertical space>
 *
 *   ───────────────────────────────
 *   [hint]  [undo]  [draw]  [resign]
 */
export default function GameMobile() {
  const { t } = useTranslation();
  const { id } = useParams<{ id: string }>();
  const gid = Number(id);
  const openSettings = useUI((s) => s.openSettings);
  const session = useGameSession(gid);

  if (session.isLoading || !session.g || !session.layout) {
    return (
      <div className="fixed inset-0 z-50 bg-bg-base flex items-center justify-center">
        <BoardSkeleton />
      </div>
    );
  }
  const g = session.g;
  // Score from the viewer's perspective: their kazan first
  const myKazan = g.side === 0 ? session.layout.whiteKazan : session.layout.blackKazan;
  const oppKazan = g.side === 0 ? session.layout.blackKazan : session.layout.whiteKazan;

  return (
    <div className="fixed inset-0 z-50 bg-bg-base flex flex-col">
      {/* Top bar — back + score + settings */}
      <header
        className="shrink-0 flex items-center justify-between px-3 py-2 bg-bg-raised border-b border-bg-border"
        style={{ paddingTop: "max(0.5rem, env(safe-area-inset-top))" }}
      >
        <Link
          to="/lobby"
          className="p-1.5 rounded hover:bg-bg-inset text-fg-secondary"
          aria-label={t("lobby.newGame")}
        >
          <ArrowLeft size={20} />
        </Link>

        <ScoreBadge mine={myKazan} opp={oppKazan} />

        <button
          type="button"
          onClick={openSettings}
          aria-label={t("settings.title")}
          className="p-1.5 rounded hover:bg-bg-inset text-fg-secondary"
        >
          <Settings size={20} />
        </button>
      </header>

      {/* Board fills the rest. min-h-0 forces flex child to allow shrinking. */}
      <div className="flex-1 min-h-0 flex items-center justify-center p-2 overflow-y-auto overscroll-contain">
        <BoardMobile
          layout={session.layout}
          side={g.side}
          sideToMove={g.sideToMove}
          onMove={session.onMove}
          disabled={session.boardDisabled}
          activePit={session.activePit}
        />
      </div>

      {/* Compact bottom action row */}
      <nav
        className="shrink-0 flex items-center justify-around bg-bg-raised border-t border-bg-border py-1.5"
        style={{ paddingBottom: "max(0.375rem, env(safe-area-inset-bottom))" }}
      >
        <ActionBtn
          icon={<HelpCircle size={20} />}
          label={t("game.hint")}
          onClick={session.doHint}
          disabled={session.actionsDisabled || g.hintsUsed >= g.hintsLimit}
          accent="gold"
          badge={g.hintsLimit - g.hintsUsed}
        />
        <ActionBtn
          icon={<Undo2 size={20} />}
          label={t("game.undo")}
          onClick={session.doUndo}
          disabled={session.actionsDisabled}
        />
        <ActionBtn
          icon={<Handshake size={20} />}
          label={t("game.draw")}
          onClick={session.doDraw}
          disabled={session.actionsDisabled}
        />
        <ActionBtn
          icon={<Flag size={20} />}
          label={t("game.resign")}
          onClick={session.doResign}
          disabled={session.actionsDisabled}
          accent="loss"
        />
      </nav>

      {/* Game-finished overlay */}
      <AnimatePresence>
        {g.status === "finished" && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 z-10 bg-bg-base/85 backdrop-blur-sm flex items-center justify-center p-6"
          >
            <motion.div
              initial={{ scale: 0.85, y: 20, opacity: 0 }}
              animate={{ scale: 1, y: 0, opacity: 1 }}
              transition={{ type: "spring", stiffness: 380, damping: 26 }}
              className="bg-bg-raised border border-bg-border rounded-2xl p-6 w-full max-w-sm shadow-2xl text-center"
            >
              <p className="text-2xl font-semibold mb-1">
                {t(`game.result.${g.result}`)}
              </p>
              {g.finalScore && (
                <p className="text-fg-secondary text-lg font-mono">{g.finalScore}</p>
              )}
              <Link
                to="/lobby"
                className="inline-block mt-5 bg-accent-amber text-bg-base px-5 py-2.5 rounded-lg font-semibold"
              >
                {t("lobby.newGame")}
              </Link>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {session.wsStatus !== "connected" && (
        <div className="absolute top-12 inset-x-3 text-center text-xs text-state-warn bg-bg-raised/90 rounded px-2 py-1">
          {t("game.wsStatus", { status: session.wsStatus })}
        </div>
      )}
    </div>
  );
}

function ScoreBadge({ mine, opp }: { mine: number; opp: number }) {
  return (
    <div className="flex items-center gap-1 font-mono text-base font-semibold tabular-nums">
      <motion.span
        key={`opp-${opp}`}
        initial={{ scale: 1.5, color: "var(--color-accent-amber)" }}
        animate={{ scale: 1, color: "var(--color-fg-secondary)" }}
        transition={{ duration: 0.45 }}
      >
        {opp}
      </motion.span>
      <span className="text-fg-muted">—</span>
      <motion.span
        key={`me-${mine}`}
        initial={{ scale: 1.5, color: "var(--color-accent-amber)" }}
        animate={{ scale: 1, color: "var(--color-fg-primary)" }}
        transition={{ duration: 0.45 }}
      >
        {mine}
      </motion.span>
    </div>
  );
}

function ActionBtn({
  icon,
  label,
  onClick,
  disabled,
  accent,
  badge,
}: {
  icon: React.ReactNode;
  label: string;
  onClick: () => void;
  disabled?: boolean;
  accent?: "gold" | "loss";
  badge?: number;
}) {
  const accentCls =
    accent === "gold"
      ? "text-accent-gold"
      : accent === "loss"
        ? "text-state-loss"
        : "text-fg-secondary";
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-label={label}
      className={`relative flex flex-col items-center gap-0.5 px-3 py-1 rounded transition disabled:opacity-40 ${accentCls} active:scale-95`}
    >
      {icon}
      <span className="text-[10px] uppercase tracking-wider">{label}</span>
      {badge != null && badge > 0 && (
        <span className="absolute top-0 right-1 bg-accent-amber text-bg-base text-[9px] rounded-full w-4 h-4 flex items-center justify-center font-bold">
          {badge}
        </span>
      )}
    </button>
  );
}

function BoardSkeleton() {
  // Subtle shimmer placeholder while the game loads
  return (
    <div className="w-full max-w-md p-4 space-y-3">
      <div className="h-32 bg-bg-raised rounded-xl animate-pulse" />
      <div className="h-12 bg-bg-raised rounded-xl animate-pulse" />
      <div className="h-32 bg-bg-raised rounded-xl animate-pulse" />
    </div>
  );
}
