#!/usr/bin/env bash
# Tests for tools/9qum/make_test_engine.sh, without any network access.
#
# The defect this guards: a test engine assembled under /tmp was wiped by a machine
# restart mid-run, killing a 24-game measurement instantly with FileNotFoundError.
# These tests check that make_test_engine.sh (a) actually assembles a working engine
# directory and verifies it starts, and (b) refuses every path back to that defect:
# a /tmp destination, models/engine/ itself, and a silent overwrite of existing files.
#
# Runs the real (committed, small) models/engine/nnue_weights.bin as the "candidate"
# weights -- no large/generated files are created by this test beyond the assembled
# engine directories themselves, which are removed at the end. No network access;
# CPU cost is a handful of `cp` calls plus one `serve`+`quit` round trip per case
# (negligible next to the live 24-game benchmark running concurrently).
#
# Run: bash tools/9qum/test_make_test_engine.sh
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT="$REPO_ROOT/tools/9qum/make_test_engine.sh"
WEIGHTS="$REPO_ROOT/models/engine/nnue_weights.bin"

PASS=0
FAIL=0
CLEANUP_DIRS=()

ok() { PASS=$((PASS + 1)); echo "  ok: $1"; }
bad() { FAIL=$((FAIL + 1)); echo "  FAIL: $1"; }

cleanup() {
    for d in "${CLEANUP_DIRS[@]:-}"; do
        [ -n "$d" ] && rm -rf "$d"
    done
}
trap cleanup EXIT

[ -x "$REPO_ROOT/target/release/togyzkumalaq-engine" ] || {
    echo "SKIP: target/release/togyzkumalaq-engine not built -- run 'cargo build --release' first" >&2
    exit 77
}
[ -f "$WEIGHTS" ] || { echo "SKIP: $WEIGHTS not found" >&2; exit 77; }

# NOTE: called as `dest="$(new_dest)"` -- that command substitution forks a subshell,
# so appending to CLEANUP_DIRS *inside* this function would be invisible to the parent
# shell. Callers must append the returned path to CLEANUP_DIRS themselves.
new_dest() {
    local d
    d="$(mktemp -d "$REPO_ROOT/models/nets/nnue_v2/eng_test_XXXXXX")"
    rmdir "$d"  # make_test_engine.sh must be able to create it fresh (mkdir -p)
    echo "$d"
}

# --------------------------------------------------------------------------
# 1) successful assembly: binary + assets + weights installed, engine verified to start
#    and report loading weights from THIS destination; sha256 printed and correct.
# --------------------------------------------------------------------------
test_successful_assembly() {
    local dest out rc want_sha
    dest="$(new_dest)"
    CLEANUP_DIRS+=("$dest")
    out="$("$SCRIPT" "$WEIGHTS" "$dest" 2>&1)"
    rc=$?
    if [ $rc -ne 0 ]; then bad "successful assembly: exit $rc, output: $out"; return; fi
    [ -x "$dest/togyzkumalaq-engine" ] && ok "binary installed and executable" \
        || bad "binary missing/not executable"
    [ -f "$dest/egtb.bin" ] && ok "egtb.bin installed" || bad "egtb.bin missing"
    [ -f "$dest/opening_book.txt" ] && ok "opening_book.txt installed" || bad "opening_book.txt missing"
    [ -f "$dest/nnue_weights.bin" ] && ok "nnue_weights.bin installed" || bad "nnue_weights.bin missing"
    cmp -s "$WEIGHTS" "$dest/nnue_weights.bin" && ok "installed weights match the source file byte-for-byte" \
        || bad "installed weights differ from source"
    want_sha="$(sha256sum "$dest/nnue_weights.bin" | cut -d' ' -f1)"
    echo "$out" | grep -qF "$want_sha" && ok "printed sha256 matches the installed file" \
        || bad "printed output did not include sha256 $want_sha:\n$out"
    echo "$out" | grep -qF "verified: engine starts and loads weights from $dest/nnue_weights.bin" \
        && ok "reported verification of engine startup + weights load from THIS dest" \
        || bad "missing startup verification line:\n$out"
}

# --------------------------------------------------------------------------
# 2) refuses to write into models/engine/ (production dir)
# --------------------------------------------------------------------------
test_refuses_models_engine() {
    local out rc
    out="$("$SCRIPT" "$WEIGHTS" "$REPO_ROOT/models/engine" 2>&1)"
    rc=$?
    [ $rc -ne 0 ] && ok "refuses models/engine/ as destination (exit $rc)" \
        || bad "should have refused models/engine/ as destination"
    echo "$out" | grep -qi "models/engine" && ok "error message names models/engine/" \
        || bad "error message doesn't explain why:\n$out"
    # must not have touched the real production engine dir
    [ -f "$REPO_ROOT/models/engine/nnue_weights.bin" ] && ok "production nnue_weights.bin untouched" \
        || bad "production nnue_weights.bin missing after refused write!"
}

# --------------------------------------------------------------------------
# 3) refuses a /tmp destination
# --------------------------------------------------------------------------
test_refuses_tmp_destination() {
    local dest out rc
    dest="/tmp/make_test_engine_should_refuse_$$"
    out="$("$SCRIPT" "$WEIGHTS" "$dest" 2>&1)"
    rc=$?
    [ $rc -ne 0 ] && ok "refuses a /tmp destination (exit $rc)" || bad "should have refused /tmp destination"
    echo "$out" | grep -qi "/tmp" && ok "error message explains the /tmp restriction" \
        || bad "error message doesn't mention /tmp:\n$out"
    [ ! -e "$dest" ] && ok "/tmp destination was never created" || { bad "/tmp destination was created!"; rm -rf "$dest"; }
}

# --------------------------------------------------------------------------
# 4) refuses a non-empty destination unless --force
# --------------------------------------------------------------------------
test_refuses_nonempty_without_force_then_force_overwrites() {
    local dest out rc
    dest="$(new_dest)"
    CLEANUP_DIRS+=("$dest")
    mkdir -p "$dest"
    echo "pre-existing" > "$dest/leftover.txt"

    out="$("$SCRIPT" "$WEIGHTS" "$dest" 2>&1)"
    rc=$?
    [ $rc -ne 0 ] && ok "refuses a non-empty destination without --force (exit $rc)" \
        || bad "should have refused a non-empty destination without --force"
    [ -f "$dest/leftover.txt" ] && ok "leftover file untouched by the refused run" \
        || bad "leftover file disappeared even though the run was refused"
    [ ! -f "$dest/nnue_weights.bin" ] && ok "nothing was installed on the refused run" \
        || bad "weights were installed despite being refused"

    out="$("$SCRIPT" "$WEIGHTS" "$dest" --force 2>&1)"
    rc=$?
    [ $rc -eq 0 ] && ok "--force overwrites a non-empty destination (exit $rc)" \
        || bad "--force run failed: $out"
    [ -f "$dest/nnue_weights.bin" ] && ok "--force run installed the weights" \
        || bad "--force run did not install the weights"
}

test_successful_assembly
test_refuses_models_engine
test_refuses_tmp_destination
test_refuses_nonempty_without_force_then_force_overwrites

echo
if [ "$FAIL" -eq 0 ]; then
    echo "OK: make_test_engine.sh assembly + all refusals ($PASS/$PASS)"
    exit 0
else
    echo "FAILED: $FAIL check(s) failed, $PASS passed"
    exit 1
fi
