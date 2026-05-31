export type BoardLayout = {
  bottomRow: Array<{ index: number; pebbles: number; isTuz: boolean }>; // indices 0..8 (player's row)
  topRow: Array<{ index: number; pebbles: number; isTuz: boolean }>;     // indices 0..8 (opponent's row, displayed reversed)
  whiteKazan: number;
  blackKazan: number;
  sideToMove: 0 | 1;
  whiteTuz: number;  // -1 if none
  blackTuz: number;
};

export function parsePos(pos: string): BoardLayout {
  // "w0,..,w8/b0,..,b8/kw,kb/tw,tb/side"
  const parts = pos.split("/");
  if (parts.length !== 5) {
    throw new Error(`invalid pos format: ${pos}`);
  }
  const [wStr, bStr, kStr, tStr, sideStr] = parts;
  const w = wStr.split(",").map((n) => parseInt(n, 10));
  const b = bStr.split(",").map((n) => parseInt(n, 10));
  const [kw, kb] = kStr.split(",").map((n) => parseInt(n, 10));
  const [tw, tb] = tStr.split(",").map((n) => parseInt(n, 10));
  const side = parseInt(sideStr, 10) as 0 | 1;

  return {
    // bottomRow is white's row; black has tuzdyk on white's pit means tb === i
    bottomRow: w.map((p, i) => ({ index: i, pebbles: p, isTuz: tb === i })),
    // topRow is black's row; white has tuzdyk on black's pit means tw === i
    topRow: b.map((p, i) => ({ index: i, pebbles: p, isTuz: tw === i })),
    whiteKazan: kw,
    blackKazan: kb,
    sideToMove: side,
    whiteTuz: tw,
    blackTuz: tb,
  };
}

export type PitRef = { side: 0 | 1; pit: number };

export const pitKey = (side: 0 | 1, pit: number) => `${side}-${pit}`;

/**
 * Compute the sowing path for a togyzkumalak move.
 *
 * Rules (mirrors engine/src/board.rs:make_move):
 *  - Both sides increment pit_index. Direction: pit++ always.
 *  - When pit overflows past 8, jump to opposite side's pit 0.
 *  - For stones >= 2: the FIRST stone goes back into the source pit itself.
 *  - For stones == 1: the single stone goes to source+1 (no return to source).
 *
 * Ring direction visually with white at bottom + opponent row displayed
 * reversed (black 8 leftmost, black 0 rightmost on top row):
 *   bottom L→R (white 0..8) → top R→L visually (black 0..8) → wrap to white 0
 *   = perfect counter-clockwise circle on the board.
 */
export function computeSowingPath(source: PitRef, stones: number): PitRef[] {
  const path: PitRef[] = [];
  if (stones <= 0) return path;

  if (stones === 1) {
    // Single stone: skip source, advance one pit
    let curPit = source.pit + 1;
    let curSide: 0 | 1 = source.side;
    if (curPit > 8) {
      curPit = 0;
      curSide = (1 - curSide) as 0 | 1;
    }
    path.push({ side: curSide, pit: curPit });
    return path;
  }

  // First stone returns to source pit itself
  path.push({ side: source.side, pit: source.pit });

  // Remaining (stones - 1) stones advance pit++ each
  let curSide: 0 | 1 = source.side;
  let curPit = source.pit;
  for (let i = 0; i < stones - 1; i++) {
    curPit++;
    if (curPit > 8) {
      curPit = 0;
      curSide = (1 - curSide) as 0 | 1;
    }
    path.push({ side: curSide, pit: curPit });
  }
  return path;
}
