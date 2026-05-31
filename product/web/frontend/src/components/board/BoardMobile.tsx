import { motion, AnimatePresence } from "framer-motion";
import { BoardLayout } from "../../domain/board";
import HoleMobile from "./HoleMobile";
import { useUI } from "../../stores/ui";

type ActivePit = { side: 0 | 1; pit: number; actor: "human" | "engine" } | null;

type Props = {
  layout: BoardLayout;
  side: 0 | 1;
  sideToMove: 0 | 1;
  onMove: (pitIndex: number) => void;
  disabled?: boolean;
  activePit?: ActivePit;
};

/**
 * Mobile portrait board.
 *
 *   ┌──────────────┬────────┬──────────────┐
 *   │ opp pit 1   │ KAZ_OPP │   pit 1 me  │
 *   │ opp pit 2   │  ──     │   pit 2 me  │
 *   │ ...         │  ──     │   ...       │
 *   │ opp pit 9   │ KAZ_ME  │   pit 9 me  │
 *   └──────────────┴────────┴──────────────┘
 *
 * Two outer columns of horizontal-orientation pits flank a centre column
 * containing the two kazans stacked vertically.
 */
export default function BoardMobile({
  layout, side, sideToMove, onMove, disabled, activePit,
}: Props) {
  const showCoordinates = useUI((s) => s.showCoordinates);

  const myRow = side === 0 ? layout.bottomRow : layout.topRow;
  const oppRow = side === 0 ? layout.topRow : layout.bottomRow;
  const myKazan = side === 0 ? layout.whiteKazan : layout.blackKazan;
  const oppKazan = side === 0 ? layout.blackKazan : layout.whiteKazan;
  const oppSide: 0 | 1 = side === 0 ? 1 : 0;
  const mySide: 0 | 1 = side;

  return (
    <div className="board-wood relative w-full max-w-md mx-auto rounded-xl p-2.5 select-none">
      <div className="flex gap-2" data-testid="mobile-rows">
        {/* Left: opponent column (mirrored — pit 1 visible at top) */}
        <div className="flex-1 flex flex-col gap-1.5" data-testid="opp-row">
          {oppRow.slice().reverse().map((h, displayPos) => {
            const isActive = activePit?.side === oppSide && activePit?.pit === h.index;
            return (
              <HoleMobile
                key={`opp-${h.index}`}
                pebbles={h.pebbles}
                interactive={false}
                isTuz={h.isTuz}
                displayIndex={displayPos + 1}
                numeralPosition="left"
                showNumeral={showCoordinates}
                isActive={isActive}
                activeActor={isActive ? activePit?.actor : undefined}
              />
            );
          })}
        </div>

        {/* Centre: two vertical kazan strips */}
        <div className="flex flex-col gap-2 w-12 sm:w-14 shrink-0">
          <KazanVertical count={oppKazan} owner="opp" />
          <KazanVertical count={myKazan} owner="me" />
        </div>

        {/* Right: player column (natural — pit 1 at top) */}
        <div className="flex-1 flex flex-col gap-1.5" data-testid="my-row">
          {myRow.map((h) => {
            const isActive = activePit?.side === mySide && activePit?.pit === h.index;
            return (
              <HoleMobile
                key={`mine-${h.index}`}
                pebbles={h.pebbles}
                interactive={!disabled && sideToMove === side && h.pebbles > 0 && !h.isTuz}
                isTuz={h.isTuz}
                displayIndex={h.index + 1}
                numeralPosition="right"
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

/**
 * Tall vertical kazan strip. Pebbles stack from the bottom upward in 2 cols.
 * Each capture pulses the strip golden — visible feedback that stones just
 * went into the kazan.
 */
function KazanVertical({ count, owner }: { count: number; owner: "opp" | "me" }) {
  // Visible cap so the strip doesn't overflow on big captures
  const visible = Math.min(count, 30);
  const PEBBLE_SIZE = 11;

  return (
    <motion.div
      key={count}
      animate={{
        boxShadow: [
          "0 0 0 rgba(245,166,35,0)",
          "0 0 22px rgba(245,166,35,0.55), inset 0 7px 16px rgba(0,0,0,0.7)",
          "inset 0 7px 16px rgba(0,0,0,0.7)",
        ],
      }}
      transition={{ duration: 0.6 }}
      className="kazan-groove rounded-2xl flex-1 min-h-0 relative overflow-hidden flex flex-col items-center py-2"
      aria-label={`${owner} kazan: ${count}`}
    >
      {/* Count badge at the very top of the strip */}
      <motion.span
        key={`count-${count}`}
        initial={{ scale: 1.4, color: "var(--color-accent-amber)" }}
        animate={{ scale: 1, color: "var(--color-pebble-mid)" }}
        transition={{ duration: 0.4 }}
        className="font-mono font-semibold text-base shrink-0 text-center mb-1"
        style={{ textShadow: "0 1px 3px rgba(0,0,0,0.85)" }}
      >
        {count}
      </motion.span>

      {/* Pebble pile — fills the rest of the strip from bottom up */}
      <div
        className="grid w-full px-1 pb-1 mt-auto"
        style={{
          gridTemplateColumns: `repeat(2, ${PEBBLE_SIZE}px)`,
          gap: "2px",
          justifyContent: "center",
          alignContent: "end",
        }}
      >
        <AnimatePresence initial={false}>
          {Array.from({ length: visible }).map((_, i) => (
            <motion.span
              key={i}
              className="pebble rounded-full block"
              style={{ width: PEBBLE_SIZE, height: PEBBLE_SIZE }}
              initial={{ scale: 0, opacity: 0, y: -4 }}
              animate={{ scale: 1, opacity: 1, y: 0 }}
              exit={{ scale: 0, opacity: 0 }}
              transition={{ duration: 0.25 }}
            />
          ))}
        </AnimatePresence>
      </div>
    </motion.div>
  );
}
