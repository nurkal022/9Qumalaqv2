import { BoardLayout } from "../../domain/board";
import Hole from "./Hole";

type Props = {
  layout: BoardLayout;
  side: 0 | 1;                      // viewer's side (0 = white at bottom)
  sideToMove: 0 | 1;
  onMove: (pitIndex: number) => void;
  disabled?: boolean;
  highlightLastMoveTo?: number;
};

export default function Board({
  layout, side, sideToMove, onMove, disabled, highlightLastMoveTo,
}: Props) {
  // White viewer: white at bottom (their row), black at top.
  // Black viewer: flip — black at bottom, white at top.
  const myRow = side === 0 ? layout.bottomRow : layout.topRow;
  const oppRow = side === 0 ? layout.topRow : layout.bottomRow;
  const myKazan = side === 0 ? layout.whiteKazan : layout.blackKazan;
  const oppKazan = side === 0 ? layout.blackKazan : layout.whiteKazan;

  return (
    <div className="bg-board-wood rounded-2xl p-4 shadow-xl select-none">
      <div className="grid grid-cols-9 gap-2" data-testid="opp-row">
        {/* Opponent row: visually reversed (right-to-left from viewer's perspective)
            because togyzkumalak is played on a circular path */}
        {oppRow.slice().reverse().map((h) => (
          <Hole
            key={`opp-${h.index}`}
            pebbles={h.pebbles}
            interactive={false}
            highlight={h.index === highlightLastMoveTo}
            isTuz={h.isTuz}
          />
        ))}
      </div>

      <div className="flex justify-between items-center my-3 text-fg-secondary text-sm font-mono">
        <div className="px-3 py-1 bg-bg-inset rounded">
          <span className="text-fg-muted text-xs">opp</span> {oppKazan}
        </div>
        <div className="px-3 py-1 bg-bg-inset rounded">
          {myKazan} <span className="text-fg-muted text-xs">you</span>
        </div>
      </div>

      <div className="grid grid-cols-9 gap-2" data-testid="my-row">
        {myRow.map((h) => (
          <Hole
            key={`mine-${h.index}`}
            pebbles={h.pebbles}
            interactive={!disabled && sideToMove === side && h.pebbles > 0 && !h.isTuz}
            highlight={h.index === highlightLastMoveTo}
            isTuz={h.isTuz}
            onClick={() => onMove(h.index)}
          />
        ))}
      </div>
    </div>
  );
}
