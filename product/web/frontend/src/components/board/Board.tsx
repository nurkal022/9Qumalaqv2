import { motion } from "framer-motion";
import { BoardLayout } from "../../domain/board";
import Hole from "./Hole";
import Pebbles from "./Pebbles";
import { useTranslation } from "react-i18next";
import { useUI } from "../../stores/ui";

type ActivePit = { side: 0 | 1; pit: number; actor: "human" | "engine" } | null;

type Props = {
  layout: BoardLayout;
  /** viewer's side (0 = white at bottom of board) */
  side: 0 | 1;
  sideToMove: 0 | 1;
  onMove: (pitIndex: number) => void;
  disabled?: boolean;
  /** The pit currently being acted on by the sowing animation
   *  (source on pickup, then each landing pit in sequence) */
  activePit?: ActivePit;
  turnLabel?: string | null;
};

export default function Board({
  layout, side, sideToMove, onMove, disabled, activePit, turnLabel,
}: Props) {
  const { t } = useTranslation();
  const showCoordinates = useUI((s) => s.showCoordinates);

  const myRow = side === 0 ? layout.bottomRow : layout.topRow;
  const oppRow = side === 0 ? layout.topRow : layout.bottomRow;
  const myKazan = side === 0 ? layout.whiteKazan : layout.blackKazan;
  const oppKazan = side === 0 ? layout.blackKazan : layout.whiteKazan;
  const oppSide: 0 | 1 = side === 0 ? 1 : 0;
  const mySide: 0 | 1 = side;

  const sideToMoveLabel = turnLabel
    ?? t(sideToMove === 0 ? "board.turnWhite" : "board.turnBlack");

  return (
    <div className="flex gap-2 sm:gap-3 w-full">
      {/* Side nameplate — turn indicator + score */}
      <div className="nameplate flex flex-col items-center justify-between rounded-md py-3 px-2 sm:py-4 sm:px-3 min-w-[2.75rem] sm:min-w-[3.25rem]">
        <span
          className="text-[11px] sm:text-xs font-medium uppercase tracking-[0.2em] whitespace-nowrap"
          style={{ writingMode: "vertical-rl", transform: "rotate(180deg)" }}
        >
          {sideToMoveLabel}
        </span>
        <div className="font-mono text-lg sm:text-xl tracking-wide mt-2 flex flex-col items-center gap-0.5">
          <motion.span
            key={`opp-${oppKazan}`}
            initial={{ scale: 1.3, color: "var(--color-accent-amber)" }}
            animate={{ scale: 1, color: "var(--color-pebble-light)" }}
            transition={{ duration: 0.5 }}
          >
            {oppKazan}
          </motion.span>
          <span className="text-fg-muted text-xs">—</span>
          <motion.span
            key={`my-${myKazan}`}
            initial={{ scale: 1.3, color: "var(--color-accent-amber)" }}
            animate={{ scale: 1, color: "var(--color-pebble-light)" }}
            transition={{ duration: 0.5 }}
          >
            {myKazan}
          </motion.span>
        </div>
      </div>

      <div className="board-wood relative flex-1 rounded-xl p-3 sm:p-5 select-none">
        {/* Opponent row — numerals ABOVE, visually reversed */}
        <div className="grid grid-cols-9 gap-2 sm:gap-3" data-testid="opp-row">
          {oppRow.slice().reverse().map((h, displayPos) => {
            const isActive = activePit?.side === oppSide && activePit?.pit === h.index;
            return (
              <Hole
                key={`opp-${h.index}`}
                pebbles={h.pebbles}
                interactive={false}
                isTuz={h.isTuz}
                displayIndex={displayPos + 1}
                numeralPosition="above"
                showNumeral={showCoordinates}
                isActive={isActive}
                activeActor={isActive ? activePit?.actor : undefined}
              />
            );
          })}
        </div>

        {/* Two kazan grooves */}
        <div className="my-3 sm:my-4 space-y-2">
          <KazanGroove count={oppKazan} side="left" />
          <KazanGroove count={myKazan} side="right" />
        </div>

        {/* Player row — numerals BELOW, natural left-to-right */}
        <div className="grid grid-cols-9 gap-2 sm:gap-3" data-testid="my-row">
          {myRow.map((h) => {
            const isActive = activePit?.side === mySide && activePit?.pit === h.index;
            return (
              <Hole
                key={`mine-${h.index}`}
                pebbles={h.pebbles}
                interactive={!disabled && sideToMove === side && h.pebbles > 0 && !h.isTuz}
                isTuz={h.isTuz}
                displayIndex={h.index + 1}
                numeralPosition="below"
                showNumeral={showCoordinates}
                isActive={isActive}
                activeActor={isActive ? activePit?.actor : undefined}
                onClick={() => onMove(h.index)}
              />
            );
          })}
        </div>
      </div>
    </div>
  );
}

function KazanGroove({ count, side }: { count: number; side: "left" | "right" }) {
  const minH = count > 36 ? "h-16 sm:h-20" : count > 18 ? "h-14 sm:h-16" : "h-12 sm:h-14";
  return (
    <motion.div
      // The motion `key={count}` retriggers the boxShadow animation each time
      // the kazan count changes — visible "capture" pulse so the player sees
      // stones arriving in the kazan rather than the score just jumping.
      key={count}
      animate={{
        boxShadow: [
          "0 0 0 rgba(245,166,35,0)",
          "0 0 22px rgba(245,166,35,0.5), inset 0 7px 16px rgba(0,0,0,0.7)",
          "inset 0 7px 16px rgba(0,0,0,0.7)",
        ],
      }}
      transition={{ duration: 0.6 }}
      className={`kazan-groove rounded-3xl ${minH} flex items-center px-3 relative overflow-hidden transition-[height]`}
    >
      {side === "left" ? (
        <>
          <CountBadge value={count} />
          <div className="flex-1 h-full flex items-center">
            <Pebbles count={count} variant="kazan" />
          </div>
        </>
      ) : (
        <>
          <div className="flex-1 h-full flex items-center">
            <Pebbles count={count} variant="kazan" />
          </div>
          <CountBadge value={count} />
        </>
      )}
    </motion.div>
  );
}

function CountBadge({ value }: { value: number }) {
  return (
    <motion.span
      key={value}
      initial={{ scale: 1.4, color: "var(--color-accent-amber)" }}
      animate={{ scale: 1, color: "var(--color-pebble-mid)" }}
      transition={{ duration: 0.4 }}
      className="font-mono font-semibold text-sm sm:text-base shrink-0 w-7 text-center"
    >
      {value}
    </motion.span>
  );
}
