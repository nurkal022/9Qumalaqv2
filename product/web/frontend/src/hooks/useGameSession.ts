import { useCallback, useEffect, useRef, useState } from "react";
import { useGameQuery } from "./useGameQuery";
import { useGameSocket } from "./useGameSocket";
import { playApi, type GameState, type Move } from "../api/play";
import type { ServerMsg } from "../api/ws";
import { parsePos, computeSowingPath, type BoardLayout } from "../domain/board";
import { useUI } from "../stores/ui";

// Base timings (multiplied by 1/sowingSpeed at runtime)
const SOW_STEP_MS = 280;
const PICKUP_BUFFER_MS = 320;
const MOVE_GAP_MS = 450;

export type ActivePit = { side: 0 | 1; pit: number; actor: "human" | "engine" } | null;

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

/**
 * Owns all the per-game state, animation lifecycle, REST/WS plumbing, and
 * derived display state for a single game session. Both the desktop and
 * mobile views consume this hook so the rendering layer is purely cosmetic.
 */
export function useGameSession(gid: number) {
  const sowingSpeed = useUI((s) => s.sowingSpeed);
  const { data: g, isLoading, applyServerSnapshot, refetch } = useGameQuery(gid);

  const onWsMsg = useCallback((m: ServerMsg) => {
    if (m.type === "snapshot") {
      applyServerSnapshot(m.game);
    } else if (
      m.type === "engine_move" ||
      m.type === "move_applied" ||
      m.type === "game_finished" ||
      m.type === "engine_thinking"
    ) {
      refetch();
    }
  }, [applyServerSnapshot, refetch]);

  const { status: wsStatus } = useGameSocket(gid, onWsMsg);

  const [animatedLayout, setAnimatedLayout] = useState<BoardLayout | null>(null);
  const [activePit, setActivePit] = useState<ActivePit>(null);
  const [pendingPick, setPendingPick] = useState<{ side: 0 | 1; pit: number } | null>(null);
  const prevGameRef = useRef<GameState | null>(null);
  const animTokenRef = useRef(0);

  useEffect(() => {
    if (!g) return;
    const prev = prevGameRef.current;
    prevGameRef.current = g;

    if (!prev || g.moves.length <= prev.moves.length) return;
    const newMoves = g.moves.slice(prev.moves.length);
    const myToken = ++animTokenRef.current;
    void runSowAnimation(prev, newMoves, sowingSpeed, myToken);

    async function runSowAnimation(
      prevG: GameState,
      moves: Move[],
      speed: number,
      token: number,
    ) {
      const k = 1 / Math.max(0.1, speed);
      const pickupMs = PICKUP_BUFFER_MS * k;
      const stepMs = SOW_STEP_MS * k;
      const gapMs = MOVE_GAP_MS * k;

      let working = parsePos(prevG.currentFen);
      setAnimatedLayout(working);

      for (const move of moves) {
        if (animTokenRef.current !== token) return;
        const sourceSide = move.side as 0 | 1;
        const oppSide = (1 - sourceSide) as 0 | 1;
        const sourcePit = parseInt(move.moveUci, 10);
        if (Number.isNaN(sourcePit)) continue;
        const actor: "human" | "engine" = move.actor === "engine" ? "engine" : "human";

        const sourceRow = sourceSide === 0 ? working.bottomRow : working.topRow;
        const stones = sourceRow[sourcePit]?.pebbles ?? 0;
        if (stones <= 0) continue;

        // Pickup — clear source, highlight it
        setActivePit({ side: sourceSide, pit: sourcePit, actor });
        working = pickupSource(working, sourceSide, sourcePit);
        setAnimatedLayout(working);
        await sleep(pickupMs);
        if (animTokenRef.current !== token) return;

        // Sow stones one at a time. Each landing on a tuzdyk pit goes
        // straight to the corresponding kazan rather than into the pit.
        const path = computeSowingPath({ side: sourceSide, pit: sourcePit }, stones);
        for (const p of path) {
          if (animTokenRef.current !== token) return;

          const moverTuzPit = sourceSide === 0 ? working.whiteTuz : working.blackTuz;
          const oppTuzPit = sourceSide === 0 ? working.blackTuz : working.whiteTuz;
          const isMoverTuzdyk = p.side === oppSide && moverTuzPit === p.pit;
          const isOppTuzdyk = p.side === sourceSide && oppTuzPit === p.pit;

          if (isMoverTuzdyk) {
            working = addToKazan(working, sourceSide, 1);
          } else if (isOppTuzdyk) {
            working = addToKazan(working, oppSide, 1);
          } else {
            working = addStone(working, p.side, p.pit);
          }
          setActivePit({ side: p.side, pit: p.pit, actor });
          setAnimatedLayout(working);
          await sleep(stepMs);
        }

        // End-of-move: capture or tuzdyk creation. Mirrors engine/src/board.rs.
        const lastP = path[path.length - 1];
        const moverTuzPit = sourceSide === 0 ? working.whiteTuz : working.blackTuz;
        const oppTuzPit = sourceSide === 0 ? working.blackTuz : working.whiteTuz;
        const lastWasTuzdyk =
          (lastP.side === oppSide && moverTuzPit === lastP.pit) ||
          (lastP.side === sourceSide && oppTuzPit === lastP.pit);

        if (lastP.side === oppSide && !lastWasTuzdyk) {
          const oppRow = lastP.side === 0 ? "bottomRow" : "topRow";
          const count = working[oppRow][lastP.pit].pebbles;

          if (count === 3 && canCreateTuzdyk(working, sourceSide, lastP.pit)) {
            // Tuzdyk creation — claim the pit, transfer stones to mover's kazan
            working = createTuzdyk(working, sourceSide, lastP.pit);
            setActivePit({ side: lastP.side, pit: lastP.pit, actor });
            setAnimatedLayout(working);
            await sleep(stepMs);
          } else if (count > 0 && count % 2 === 0) {
            // Even-count capture — all stones go to mover's kazan
            working = captureFromPit(working, lastP.side, lastP.pit, sourceSide);
            setActivePit({ side: lastP.side, pit: lastP.pit, actor });
            setAnimatedLayout(working);
            await sleep(stepMs);
          }
        }

        await sleep(gapMs);
      }

      if (animTokenRef.current !== token) return;
      setAnimatedLayout(null);
      setActivePit(null);
      setPendingPick(null);
    }
  }, [g, sowingSpeed]);

  // Render-time derived state. The "pending" branch prevents a one-frame
  // flash of the FINAL state between REST return and animation start.
  const prev = prevGameRef.current;
  const animationPending =
    !animatedLayout && prev != null && (g?.moves.length ?? 0) > prev.moves.length;
  const layout: BoardLayout | null = animatedLayout
    ?? (animationPending && prev
      ? parsePos(prev.currentFen)
      : (g ? parsePos(g.currentFen) : null));
  const displayActivePit: ActivePit = activePit
    ?? (pendingPick ? { ...pendingPick, actor: "human" } : null);

  const onMove = useCallback(async (pitIndex: number) => {
    if (pendingPick || animatedLayout || !g) return;
    setPendingPick({ side: g.side, pit: pitIndex });
    try {
      const r = await playApi.move(gid, String(pitIndex));
      applyServerSnapshot(r.game);
    } catch (e) {
      console.error("move failed", e);
      setPendingPick(null);
    }
  }, [pendingPick, animatedLayout, g, gid, applyServerSnapshot]);

  const doResign = useCallback(async () => {
    const r = await playApi.resign(gid);
    applyServerSnapshot(r.game);
  }, [gid, applyServerSnapshot]);

  const doDraw = useCallback(async () => {
    const r = await playApi.drawOffer(gid);
    applyServerSnapshot(r.game);
  }, [gid, applyServerSnapshot]);

  const doUndo = useCallback(async () => {
    const r = await playApi.undo(gid);
    applyServerSnapshot(r.game);
  }, [gid, applyServerSnapshot]);

  const doHint = useCallback(async () => {
    await playApi.hint(gid);
    refetch();
  }, [gid, refetch]);

  return {
    g,
    isLoading,
    layout,
    activePit: displayActivePit,
    /** True when the board should reject clicks (during animation, pending REST, or finished) */
    boardDisabled:
      !g
      || g.status !== "active"
      || g.engineThinking
      || animatedLayout !== null
      || animationPending
      || pendingPick !== null,
    /** True while the action buttons should be inactive (game already finished) */
    actionsDisabled: !g || g.status !== "active",
    wsStatus,
    onMove,
    doResign,
    doDraw,
    doUndo,
    doHint,
  };
}

function pickupSource(layout: BoardLayout, side: 0 | 1, pit: number): BoardLayout {
  const target = side === 0 ? "bottomRow" : "topRow";
  return {
    ...layout,
    [target]: layout[target].map((h, i) => i === pit ? { ...h, pebbles: 0 } : h),
  };
}
function addStone(layout: BoardLayout, side: 0 | 1, pit: number): BoardLayout {
  const target = side === 0 ? "bottomRow" : "topRow";
  return {
    ...layout,
    [target]: layout[target].map((h, i) => i === pit ? { ...h, pebbles: h.pebbles + 1 } : h),
  };
}
function addToKazan(layout: BoardLayout, side: 0 | 1, n: number): BoardLayout {
  return side === 0
    ? { ...layout, whiteKazan: layout.whiteKazan + n }
    : { ...layout, blackKazan: layout.blackKazan + n };
}
function captureFromPit(
  layout: BoardLayout,
  oppRowSide: 0 | 1,
  pit: number,
  moverSide: 0 | 1,
): BoardLayout {
  const target = oppRowSide === 0 ? "bottomRow" : "topRow";
  const count = layout[target][pit].pebbles;
  let next: BoardLayout = {
    ...layout,
    [target]: layout[target].map((h, i) => i === pit ? { ...h, pebbles: 0 } : h),
  };
  next = addToKazan(next, moverSide, count);
  return next;
}
function canCreateTuzdyk(layout: BoardLayout, mover: 0 | 1, pit: number): boolean {
  if (pit === 8) return false;
  const myTuz = mover === 0 ? layout.whiteTuz : layout.blackTuz;
  if (myTuz !== -1) return false;
  const oppTuz = mover === 0 ? layout.blackTuz : layout.whiteTuz;
  if (oppTuz === pit) return false;
  return true;
}
function createTuzdyk(layout: BoardLayout, mover: 0 | 1, pit: number): BoardLayout {
  const oppRowKey = mover === 0 ? "topRow" : "bottomRow";
  const count = layout[oppRowKey][pit].pebbles;
  let next: BoardLayout = {
    ...layout,
    [oppRowKey]: layout[oppRowKey].map((h, i) =>
      i === pit ? { ...h, pebbles: 0, isTuz: true } : h,
    ),
  };
  if (mover === 0) next = { ...next, whiteTuz: pit };
  else next = { ...next, blackTuz: pit };
  next = addToKazan(next, mover, count);
  return next;
}
