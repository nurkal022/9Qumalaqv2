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
