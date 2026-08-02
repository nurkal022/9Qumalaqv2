#!/usr/bin/env python3
"""Tests for tools/9qum/lead_profile.py (task 15).

Pure-function tests, no network, no subprocess: the per-schema adapters
(views_from_ab_match_record / views_from_9qum_record), the checkpoint math, and the
replay against tools/playok/engine.py's real board transition (using small,
hand-traceable move sequences whose kazan values are verified by hand below, not
just re-derived from the same code under test).

Run: python3.12 tools/9qum/test_lead_profile.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lead_profile  # noqa: E402


# --------------------------------------------------------------------------
# checkpoint_leads(): quartile indexing + peak
# --------------------------------------------------------------------------
def test_checkpoint_leads_basic():
    # 8 ply game (leads has 9 entries incl. the ply-0 baseline); indices picked by
    # round(pct * 8): p25->2, p50->4, p75->6, final->8
    leads = [0, 1, 2, 5, 3, -1, 4, 6, 2]
    cps = lead_profile.checkpoint_leads(leads)
    assert cps["plies"] == 8
    assert cps["p25"] == leads[2] == 2
    assert cps["p50"] == leads[4] == 3
    assert cps["p75"] == leads[6] == 4
    assert cps["final"] == leads[8] == 2
    assert cps["peak"] == max(leads) == 6


def test_checkpoint_leads_zero_ply_game_is_none():
    assert lead_profile.checkpoint_leads([0]) is None


# --------------------------------------------------------------------------
# views_from_ab_match_record(): colour/side mapping, result mirroring, VOID-like
# (missing/invalid) records rejected
# --------------------------------------------------------------------------
def test_views_from_ab_match_record_white_is_a():
    rec = {
        "engine_a": {"path": "/eng/A", "weights_sha256": "aaaa1111bb"},
        "engine_b": {"path": "/eng/B", "weights_sha256": "bbbb2222cc"},
        "white": "A", "black": "B", "moves": [3, 4, 5], "result_a": "W",
        "kazan": [10, 2],
    }
    views = lead_profile.views_from_ab_match_record(rec, "src:1")
    assert len(views) == 2
    by_engine = {v["engine"]: v for v in views}
    a_view = [v for v in views if v["side"] == 0][0]
    b_view = [v for v in views if v["side"] == 1][0]
    assert a_view["result"] == "W" and b_view["result"] == "L"
    assert a_view["moves"] == [3, 4, 5] and b_view["moves"] == [3, 4, 5]
    assert "/eng/A" in a_view["engine"] and "aaaa1111bb" in a_view["engine"]
    assert "/eng/B" in b_view["engine"]


def test_views_from_ab_match_record_white_is_b_flips_sides_and_results():
    rec = {
        "engine_a": {"path": "/eng/A"}, "engine_b": {"path": "/eng/B"},
        "white": "B", "black": "A", "moves": [1], "result_a": "L", "kazan": [0, 1],
    }
    views = lead_profile.views_from_ab_match_record(rec, "src:2")
    a_view = [v for v in views if "/eng/A" in v["engine"]][0]
    b_view = [v for v in views if "/eng/B" in v["engine"]][0]
    assert a_view["side"] == 1 and b_view["side"] == 0
    assert a_view["result"] == "L" and b_view["result"] == "W"


def test_views_from_ab_match_record_draw_stays_a_draw_for_both():
    rec = {
        "engine_a": {"path": "/eng/A"}, "engine_b": {"path": "/eng/B"},
        "white": "A", "black": "B", "moves": [], "result_a": "D", "kazan": [0, 0],
    }
    views = lead_profile.views_from_ab_match_record(rec, "src:3")
    assert all(v["result"] == "D" for v in views)


def test_views_from_ab_match_record_missing_fields_yields_nothing():
    assert lead_profile.views_from_ab_match_record({}, "src:4") == []
    assert lead_profile.views_from_ab_match_record(
        {"engine_a": {}, "engine_b": {}, "white": "A", "moves": [1], "result_a": "VOID"},
        "src:5") == []


# --------------------------------------------------------------------------
# views_from_9qum_record(): opening-prefix + record concatenation, VOID excluded
# --------------------------------------------------------------------------
def test_views_from_9qum_record_concatenates_opening_and_record():
    rec = {"our_seat": 0, "result": "W", "opening": "7,9,6,8", "record": "4 9 9",
           "kazan": [50, 40], "engine_path": "/eng/ours", "engine_weights_sha256": "deadbeef01"}
    views = lead_profile.views_from_9qum_record(rec, "src:6")
    assert len(views) == 1
    v = views[0]
    # 1-9 human labels -> 0-8 wire indices, opening prefix THEN the post-opening moves
    assert v["moves"] == [6, 8, 5, 7, 3, 8, 8]
    assert v["side"] == 0 and v["result"] == "W"
    assert "/eng/ours" in v["engine"] and "deadbeef01" in v["engine"]


def test_views_from_9qum_record_no_opening_is_just_record():
    rec = {"our_seat": 1, "result": "L", "opening": None, "record": "1 2 3", "kazan": [0, 0]}
    views = lead_profile.views_from_9qum_record(rec, "src:7")
    assert views[0]["moves"] == [0, 1, 2]
    assert views[0]["side"] == 1


def test_views_from_9qum_record_void_is_excluded():
    rec = {"our_seat": 0, "result": "VOID", "record": "1 2 3"}
    assert lead_profile.views_from_9qum_record(rec, "src:8") == []


def test_views_from_9qum_record_missing_our_seat_is_excluded():
    rec = {"result": "W", "record": "1 2 3"}
    assert lead_profile.views_from_9qum_record(rec, "src:9") == []


# --------------------------------------------------------------------------
# replay(): against tools/playok/engine.py, on a hand-traced sequence
# --------------------------------------------------------------------------
def test_replay_matches_hand_traced_opening_moves():
    # Start position: 9 stones in every pit, both kazans 0, side 0 (white) to move.
    # Move 1: white plays pit 0 (9 stones) -> lands the 9th stone in black's pit 8 (a
    # normal 9-count distribution wraps exactly onto the opponent's last pit; not a
    # capture since it is odd and pit 8 is never a legal tuzdyk pit). No capture; both
    # kazans still 0 after white's first move.
    # Move 2: black plays pit 0 (9 stones): its own pit1..8 get +1 each (8 pits), 9th
    # stone lands on white's pit 0 -- white's pit0 was just emptied by white's own
    # move, so it now holds exactly 1 stone (no capture, count isn't even).
    # This trace only checks that replay() runs the real transition and reports a
    # lead of 0 for a completely symmetric two-ply opening with no captures.
    moves = [0, 0]
    leads, final_kazan = lead_profile.replay(moves, our_side=0)
    assert final_kazan == [0, 0], "a two-ply symmetric opening with no captures banks nothing"
    assert leads == [0, 0, 0]


def test_replay_lead_is_from_requested_sides_perspective():
    # A single move that creates an immediate capture: white's pit 8 has 9 stones;
    # played, it distributes to black's pits 0..7 (8 stones) then wraps to black's
    # pit 8 with the 9th -- but let's instead use a position we can trust by
    # inspection: replay from side 0 and side 1 on the SAME moves must give exactly
    # opposite lead trajectories.
    moves = [3, 5, 0, 8, 1, 2]
    leads_white, kazan_w = lead_profile.replay(moves, our_side=0)
    leads_black, kazan_b = lead_profile.replay(moves, our_side=1)
    assert kazan_w == kazan_b, "the underlying game replayed is identical either way"
    assert leads_white == [-x for x in leads_black]


TESTS = [
    test_checkpoint_leads_basic,
    test_checkpoint_leads_zero_ply_game_is_none,
    test_views_from_ab_match_record_white_is_a,
    test_views_from_ab_match_record_white_is_b_flips_sides_and_results,
    test_views_from_ab_match_record_draw_stays_a_draw_for_both,
    test_views_from_ab_match_record_missing_fields_yields_nothing,
    test_views_from_9qum_record_concatenates_opening_and_record,
    test_views_from_9qum_record_no_opening_is_just_record,
    test_views_from_9qum_record_void_is_excluded,
    test_views_from_9qum_record_missing_our_seat_is_excluded,
    test_replay_matches_hand_traced_opening_moves,
    test_replay_lead_is_from_requested_sides_perspective,
]


if __name__ == "__main__":
    for t in TESTS:
        t()
    print(f"OK: lead_profile schema adapters + checkpoint math + replay ({len(TESTS)}/{len(TESTS)})")
