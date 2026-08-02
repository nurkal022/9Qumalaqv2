#!/usr/bin/env python3
"""Tests for tools/ab_match.py's --save game recording (task 15).

Why this exists: tools/ab_match.py plays our own engines against each other and,
before this change, kept only the aggregate score -- when a diagnosis needed to
replay games and watch how the material lead evolved, that was only possible for
games played against the external 9qum opponent (tools/9qum/match.py stores full
move lists), never for our own internal A/B matches. --save fixes that; this file
is the check that a saved record is trustworthy: every required field is present,
and -- the one that would have caught a mis-recorded game -- replaying the saved
move list through a SECOND, independent implementation (tools/playok/engine.py's
pure-Python board transition) reproduces the exact final position and kazans that
tools/ab_match.py recorded via the actual game referee (alphazero-code's
TogyzQumalaq).

No network, no subprocess, no real engine binary: play_game() is driven by tiny
in-process fake "engines" that just pick a legal move, so this stays fast and CPU-
light (a monitor job is using a core on this machine). The real end-to-end proof
(spawning actual engine binaries with --save) is a separate short smoke run, not
part of this automated suite -- see the task-15 report.

Run: python3.12 tools/test_ab_match.py
"""
import hashlib
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ab_match  # noqa: E402

REPO = os.path.dirname(os.path.abspath(__file__)) + "/.."
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "playok"))
from engine import Engine as PlayokEngine, START_POSITION, _parse_pos  # noqa: E402


TMP_DIR = None


# --------------------------------------------------------------------------
# Fakes: in-process move choosers, no subprocess, driving the real TogyzQumalaq
# referee inside play_game() exactly like a real engine would.
# --------------------------------------------------------------------------
class FirstLegalMoveEngine:
    """Always plays its lowest-indexed legal move. Deterministic and cheap; good
    enough to make play_game() actually advance a real game through several plies
    via the real rules referee, without spawning any engine subprocess."""

    def bestmove(self, g, time_ms):
        moves = g.get_valid_moves_list()
        return min(moves) if moves else None


class AlwaysIllegalEngine:
    """Always returns an out-of-range pit -- used to exercise the illegal-move
    forfeit path (moves_out must NOT record a move that was never applied)."""

    def bestmove(self, g, time_ms):
        return 99


def _make_fake_engine_dir(name, weights_bytes=b"not-a-real-nnu2-file-just-bytes-to-hash"):
    """A throwaway 'engine directory': a fake, never-executed binary path plus a real
    nnue_weights.bin beside it, so compute_engine_meta has something real to hash."""
    d = os.path.join(TMP_DIR, name)
    os.makedirs(d, exist_ok=True)
    weights_path = os.path.join(d, "nnue_weights.bin")
    with open(weights_path, "wb") as f:
        f.write(weights_bytes)
    return os.path.join(d, "togyzkumalaq-engine"), weights_path, weights_bytes


# --------------------------------------------------------------------------
# play_game(): moves_out / final_state_out are opt-in and back-compatible
# --------------------------------------------------------------------------
def test_play_game_default_params_are_backward_compatible():
    result = ab_match.play_game(FirstLegalMoveEngine(), FirstLegalMoveEngine(), time_ms=1, max_plies=20)
    assert result in (0, 1, 2)


def test_moves_out_and_final_state_out_are_noops_when_not_requested():
    # same call as above but with the params explicitly None -- must not raise, must
    # not change the result, and existing callers (main() without --save) never pass
    # these, so this is exactly their code path.
    result = ab_match.play_game(FirstLegalMoveEngine(), FirstLegalMoveEngine(), time_ms=1,
                                 max_plies=20, moves_out=None, final_state_out=None)
    assert result in (0, 1, 2)


def test_illegal_move_forfeit_is_not_appended_to_moves():
    moves = []
    final_state = {}
    result = ab_match.play_game(AlwaysIllegalEngine(), FirstLegalMoveEngine(), time_ms=1,
                                 max_plies=10, moves_out=moves, final_state_out=final_state)
    assert result == 1, "white's illegal first move must forfeit the game to black"
    assert moves == [], "an illegal move must never be recorded as if it were played"
    # the referee never advanced past the empty start position
    assert final_state["kazan"] == [0, 0]


# --------------------------------------------------------------------------
# THE key check: the recorded move list, replayed through a SECOND, independent
# implementation (tools/playok/engine.py), must reproduce the exact final position
# and kazans that play_game() recorded via the real referee (TogyzQumalaq). This is
# the check that would catch a mis-recorded game.
# --------------------------------------------------------------------------
def test_moves_replay_to_recorded_final_position_and_kazan():
    moves = []
    final_state = {}
    result = ab_match.play_game(FirstLegalMoveEngine(), FirstLegalMoveEngine(), time_ms=1,
                                 max_plies=60, moves_out=moves, final_state_out=final_state)
    assert len(moves) >= 10, "expected a non-trivial number of plies to actually replay"
    assert all(0 <= m <= 8 for m in moves), "wire-protocol moves must be 0-8 pit indices"

    pos = START_POSITION
    for mv in moves:
        pos = PlayokEngine.apply_move(pos, mv)
    white, black, kaz, tuz, side = _parse_pos(pos)

    assert kaz == final_state["kazan"], f"replayed kazan {kaz} != recorded {final_state['kazan']}"
    assert [white, black] == final_state["pits"], "replayed pits != recorded pits"
    assert side == final_state["side"], "replayed side-to-move != recorded side"
    assert pos == final_state["pos"], "replayed position string != recorded position string"
    assert result in (0, 1, 2)


def test_final_state_and_moves_are_json_serializable():
    moves = []
    final_state = {}
    ab_match.play_game(FirstLegalMoveEngine(), FirstLegalMoveEngine(), time_ms=1, max_plies=20,
                        moves_out=moves, final_state_out=final_state)
    json.dumps({"moves": moves, "final_state": final_state})  # must not raise


# --------------------------------------------------------------------------
# make_game_record(): every field the brief requires must be present, and the
# per-engine weights sha256 must come from the SAME compute_engine_meta() that
# tools/9qum/match.py uses (tools/engine_provenance.py), not a re-implementation.
# --------------------------------------------------------------------------
def test_make_game_record_has_all_required_fields():
    engine_a_path, weights_a_path, weights_a_bytes = _make_fake_engine_dir("engineA")
    engine_b_path, weights_b_path, weights_b_bytes = _make_fake_engine_dir(
        "engineB", weights_bytes=b"different-weights-bytes-for-engine-b")
    meta_a = ab_match.compute_engine_meta(engine_a_path)
    meta_b = ab_match.compute_engine_meta(engine_b_path)

    moves = [3, 5, 0, 8, 1]
    final_state = {"pos": "9,9,9,9,9,9,9,9,0/9,9,9,9,9,9,9,9,9/1,0/-1,-1/1",
                    "kazan": [1, 0], "tuzdyk": [-1, -1], "side": 1,
                    "pits": [[9, 9, 9, 9, 9, 9, 9, 9, 0], [9] * 9]}
    rec = ab_match.make_game_record(meta_a, meta_b, game_index=7, a_white=True, moves=moves,
                                     final_state=final_state, result_a="W", time_ms=200, tt_mb=64,
                                     a_nobook=False, b_nobook=True, ts=1700000000)

    # two engine paths
    assert rec["engine_a"]["path"] == meta_a["engine_path"]
    assert rec["engine_b"]["path"] == meta_b["engine_path"]
    # which engine had which colour
    assert rec["white"] == "A" and rec["black"] == "B"
    # weights sha256 per engine, reusing compute_engine_meta -- not re-hashed here
    assert rec["engine_a"]["weights_sha256"] == hashlib.sha256(weights_a_bytes).hexdigest()
    assert rec["engine_b"]["weights_sha256"] == hashlib.sha256(weights_b_bytes).hexdigest()
    assert rec["engine_a"]["weights_sha256"] != rec["engine_b"]["weights_sha256"]
    # full move list: 0-8 wire indices, and the cheap 1-9 human-readable labels
    assert rec["moves"] == moves
    assert rec["moves_1based"] == [m + 1 for m in moves]
    # final position + both kazans
    assert rec["final_position"] == final_state["pos"]
    assert rec["kazan"] == final_state["kazan"]
    # result from engine A's perspective
    assert rec["result_a"] == "W"
    # number of plies
    assert rec["plies"] == len(moves)
    # time control
    assert rec["time_control_ms"] == 200
    # git commit
    assert rec["git_commit"] == meta_a["git_commit"]
    assert rec["schema"] == "ab_match_game_v1"
    assert rec["ts"] == 1700000000

    json.dumps(rec)  # the whole record must be JSON-serializable as-is


def test_make_game_record_colour_flips_when_b_is_white():
    engine_a_path, _, _ = _make_fake_engine_dir("engineA2")
    engine_b_path, _, _ = _make_fake_engine_dir("engineB2")
    meta_a = ab_match.compute_engine_meta(engine_a_path)
    meta_b = ab_match.compute_engine_meta(engine_b_path)
    final_state = {"pos": "x", "kazan": [0, 0], "tuzdyk": [-1, -1], "side": 0, "pits": [[9] * 9] * 2}
    rec = ab_match.make_game_record(meta_a, meta_b, game_index=1, a_white=False, moves=[],
                                     final_state=final_state, result_a="L", time_ms=100, tt_mb=64,
                                     a_nobook=False, b_nobook=False)
    assert rec["white"] == "B" and rec["black"] == "A"
    assert rec["result_a"] == "L"


def test_make_game_record_handles_missing_weights_file():
    d = os.path.join(TMP_DIR, "engine_no_weights")
    os.makedirs(d, exist_ok=True)
    engine_a_path = os.path.join(d, "togyzkumalaq-engine")  # no nnue_weights.bin beside it
    engine_b_path, _, _ = _make_fake_engine_dir("engineB3")
    meta_a = ab_match.compute_engine_meta(engine_a_path)
    meta_b = ab_match.compute_engine_meta(engine_b_path)
    final_state = {"pos": "x", "kazan": [0, 0], "tuzdyk": [-1, -1], "side": 0, "pits": [[9] * 9] * 2}
    rec = ab_match.make_game_record(meta_a, meta_b, game_index=0, a_white=True, moves=[],
                                     final_state=final_state, result_a="D", time_ms=100, tt_mb=64,
                                     a_nobook=False, b_nobook=False)
    assert rec["engine_a"]["weights_sha256"] is None
    assert rec["engine_a"]["weights_path"] is None
    assert rec["engine_b"]["weights_sha256"] is not None


# --------------------------------------------------------------------------
# End-to-end (still no subprocess): drive main()'s exact write path -- build a
# record, append it as JSONL, read it back -- and confirm it round-trips and
# replays, exactly as a real --save run would leave on disk after a crash.
# --------------------------------------------------------------------------
def test_saved_jsonl_record_round_trips_and_replays():
    moves = []
    final_state = {}
    ab_match.play_game(FirstLegalMoveEngine(), FirstLegalMoveEngine(), time_ms=1, max_plies=40,
                        moves_out=moves, final_state_out=final_state)
    engine_a_path, _, _ = _make_fake_engine_dir("engineA4")
    engine_b_path, _, _ = _make_fake_engine_dir("engineB4")
    meta_a = ab_match.compute_engine_meta(engine_a_path)
    meta_b = ab_match.compute_engine_meta(engine_b_path)
    rec = ab_match.make_game_record(meta_a, meta_b, game_index=0, a_white=True, moves=moves,
                                     final_state=final_state, result_a="D", time_ms=200, tt_mb=64,
                                     a_nobook=False, b_nobook=False)

    save_path = os.path.join(TMP_DIR, "ab_match_test.jsonl")
    with open(save_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with open(save_path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert len(rows) == 1
    loaded = rows[0]
    assert loaded["moves"] == moves

    pos = START_POSITION
    for mv in loaded["moves"]:
        pos = PlayokEngine.apply_move(pos, mv)
    _, _, kaz, _, _ = _parse_pos(pos)
    assert kaz == loaded["kazan"]
    assert pos == loaded["final_position"]


TESTS = [
    test_play_game_default_params_are_backward_compatible,
    test_moves_out_and_final_state_out_are_noops_when_not_requested,
    test_illegal_move_forfeit_is_not_appended_to_moves,
    test_moves_replay_to_recorded_final_position_and_kazan,
    test_final_state_and_moves_are_json_serializable,
    test_make_game_record_has_all_required_fields,
    test_make_game_record_colour_flips_when_b_is_white,
    test_make_game_record_handles_missing_weights_file,
    test_saved_jsonl_record_round_trips_and_replays,
]


if __name__ == "__main__":
    TMP_DIR = tempfile.mkdtemp(prefix="ab_match_test_")
    try:
        for t in TESTS:
            t()
        print(f"OK: ab_match --save record building + replay-back verification ({len(TESTS)}/{len(TESTS)})")
    finally:
        shutil.rmtree(TMP_DIR, ignore_errors=True)
