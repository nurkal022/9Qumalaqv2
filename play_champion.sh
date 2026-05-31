#!/bin/bash
# Interactive play vs the strongest Togyz Kumalak engine we have on this machine.
# Engine: Gen7-baseline (Mar 18 build) — empirically beats Gen7-current 99-1 in our match-up.
# Includes NNUE eval (40-256-32-1), EGTB (3.97M endgame positions), opening book (21k positions).

cd /home/nurlykhan/9QumalaqV2/engine

ENGINE=/home/nurlykhan/9QumalaqV2/engine/target/release/togyzkumalaq-engine-baseline

if [ ! -x "$ENGINE" ]; then
    echo "Engine binary missing: $ENGINE"
    exit 1
fi

echo "============================================================"
echo "  Togyz Kumalak champion-play — Gen7-baseline"
echo "  NNUE + EGTB + opening book (must run from engine dir)"
echo "============================================================"
echo ""
echo "Commands inside the prompt:"
echo "  1-9          — make a move (your pit number)"
echo "  undo         — take back the last move"
echo "  quit         — exit"
echo ""

exec "$ENGINE" play
