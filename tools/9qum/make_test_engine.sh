#!/usr/bin/env bash
# make_test_engine.sh -- assemble a runnable engine directory beside a candidate
# NNUE weights file, for A/B matches and gates (tools/9qum/match.py, ab_match.py, etc).
#
# WHY THIS EXISTS: a test engine assembled under /tmp was wiped by a machine restart
# partway through a 24-game run, which then died instantly with FileNotFoundError --
# losing the whole measurement. /tmp is NOT persistent across reboots on this machine
# and MUST NOT be used for a test engine's destination directory; this script refuses
# to write there (see the /tmp check below) and defaults to a location that survives a
# restart: models/nets/nnue_v2/eng_* (already gitignored -- rebuildable, not tracked).
#
# Usage:
#   tools/9qum/make_test_engine.sh <weights.bin> <dest-dir> [--force]
#
# What it does:
#   1. copies target/release/togyzkumalaq-engine, egtb.bin and opening_book.txt (the
#      latter two from models/engine/, the production assets -- same tablebase/book
#      every candidate is compared under) into <dest-dir>;
#   2. installs <weights.bin> as <dest-dir>/nnue_weights.bin;
#   3. starts the assembled engine (serve mode) and VERIFIES it actually starts and
#      reports loading the weights from <dest-dir> specifically (not some other
#      nnue_weights.bin found earlier on its own asset search path);
#   4. prints the sha256 of the installed weights, so the resulting directory's
#      identity is unambiguous (see tools/9qum/match.py's engine provenance fields,
#      which record this same sha256 into every game record).
#
# Refuses to:
#   - write into models/engine/ (the production engine directory);
#   - write under /tmp (does not survive a reboot -- see above);
#   - overwrite a non-empty destination unless --force is given.
#
# Example:
#   tools/9qum/make_test_engine.sh models/nets/nnue_v2/v2_e12_v3.bin \
#       models/nets/nnue_v2/eng_v2_e12
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENGINE_BIN="$REPO_ROOT/target/release/togyzkumalaq-engine"
ASSETS_DIR="$REPO_ROOT/models/engine"

usage() {
    cat >&2 <<EOF
usage: $(basename "$0") <weights.bin> <dest-dir> [--force]

Assembles a runnable engine dir: release binary + egtb.bin + opening_book.txt (from
models/engine/) + the given weights installed as nnue_weights.bin.

dest-dir must NOT be under /tmp (a machine restart wipes /tmp and previously killed a
24-game run instantly with FileNotFoundError). Default destinations:
models/nets/nnue_v2/eng_* (already gitignored, rebuildable by this script).
EOF
    exit 2
}

FORCE=0
POSITIONAL=()
for arg in "$@"; do
    case "$arg" in
        --force) FORCE=1 ;;
        -h|--help) usage ;;
        *) POSITIONAL+=("$arg") ;;
    esac
done
[ "${#POSITIONAL[@]}" -eq 2 ] || usage

WEIGHTS="${POSITIONAL[0]}"
DEST="${POSITIONAL[1]}"

[ -f "$WEIGHTS" ] || { echo "error: weights file not found: $WEIGHTS" >&2; exit 1; }
[ -x "$ENGINE_BIN" ] || {
    echo "error: release binary not found or not executable: $ENGINE_BIN" >&2
    echo "  (run 'cargo build --release' from the repo root first)" >&2
    exit 1
}
[ -f "$ASSETS_DIR/egtb.bin" ] || { echo "error: missing $ASSETS_DIR/egtb.bin" >&2; exit 1; }
[ -f "$ASSETS_DIR/opening_book.txt" ] || { echo "error: missing $ASSETS_DIR/opening_book.txt" >&2; exit 1; }

# Reject /tmp BEFORE resolving to a real path (an unresolvable dest may not exist yet).
case "$DEST" in
    /tmp|/tmp/*)
        echo "error: destination '$DEST' is under /tmp, which does not survive a machine" >&2
        echo "  restart (this is exactly the defect this script exists to prevent -- see" >&2
        echo "  the header comment). Use models/nets/nnue_v2/eng_* instead." >&2
        exit 1
        ;;
esac

mkdir -p "$DEST"
DEST_REAL="$(cd "$DEST" && pwd)"
ASSETS_REAL="$(cd "$ASSETS_DIR" && pwd)"

case "$DEST_REAL" in
    /tmp|/tmp/*)
        echo "error: destination resolves to '$DEST_REAL', under /tmp -- see above" >&2
        exit 1
        ;;
esac
case "$DEST_REAL" in
    "$ASSETS_REAL"|"$ASSETS_REAL"/*)
        echo "error: refusing to write into models/engine/ (the production engine dir)." >&2
        echo "  Pick a destination under models/nets/nnue_v2/eng_* instead." >&2
        exit 1
        ;;
esac

if [ -n "$(ls -A "$DEST_REAL" 2>/dev/null)" ] && [ "$FORCE" -ne 1 ]; then
    echo "error: destination $DEST_REAL is not empty; pass --force to overwrite" >&2
    exit 1
fi

cp "$ENGINE_BIN" "$DEST_REAL/togyzkumalaq-engine"
cp "$ASSETS_DIR/egtb.bin" "$DEST_REAL/egtb.bin"
cp "$ASSETS_DIR/opening_book.txt" "$DEST_REAL/opening_book.txt"
cp "$WEIGHTS" "$DEST_REAL/nnue_weights.bin"
chmod +x "$DEST_REAL/togyzkumalaq-engine"

SHA=$(sha256sum "$DEST_REAL/nnue_weights.bin" | cut -d' ' -f1)
echo "assembled: $DEST_REAL"
echo "  weights:  $WEIGHTS -> $DEST_REAL/nnue_weights.bin"
echo "  sha256:   $SHA"

# Verify the assembled engine actually starts and loads THESE weights (not some other
# nnue_weights.bin it might find first on its own asset search path) -- the assembled
# directory is useless as test evidence if it silently fell back to a stale net.
STDERR_FILE="$(mktemp)"
trap 'rm -f "$STDERR_FILE"' EXIT
FIRST_LINE=$(printf 'quit\n' | TT_SIZE_MB=16 "$DEST_REAL/togyzkumalaq-engine" serve 2>"$STDERR_FILE" | head -n1)
if [ "$FIRST_LINE" != "ready" ]; then
    echo "error: engine did not respond 'ready' (got: '$FIRST_LINE')" >&2
    echo "--- engine stderr ---" >&2
    cat "$STDERR_FILE" >&2
    exit 1
fi
EXPECT="NNUE loaded from $DEST_REAL/nnue_weights.bin"
if ! grep -qF "$EXPECT" "$STDERR_FILE"; then
    echo "error: engine did not report loading NNUE weights from $DEST_REAL/nnue_weights.bin" >&2
    echo "  expected a line containing: $EXPECT" >&2
    echo "--- engine stderr ---" >&2
    cat "$STDERR_FILE" >&2
    exit 1
fi
echo "verified: engine starts and loads weights from $DEST_REAL/nnue_weights.bin"
echo "test engine ready: $DEST_REAL"
