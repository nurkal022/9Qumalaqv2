#!/usr/bin/env python3
"""NNUE v2 feature layout — the Python side of the contract in engine/src/nnue.rs.

292 features, 23 active. Keep this file and build_features_v2() in nnue.rs in lockstep;
test_features_v2.py fails if they drift.
"""
NUM_FEATURES = 292
NUM_BUCKETS = 4


def count_bucket(c: int) -> int:
    if c <= 9:
        return c
    if c <= 12:
        return 10
    if c <= 16:
        return 11
    if c <= 24:
        return 12
    return 13


def _tuz_rel(tuzdyk, side):
    """9qum stores tuzdyk[p] as an absolute pit index on the opponent's side; the engine
    wants the index inside that side's row, or 9 for none."""
    t = tuzdyk[side]
    if t is None or t < 0:
        return 9
    return t - 9 if side == 0 else t


def build_features(pits, kazan, tuzdyk, to_move):
    me, opp = to_move, 1 - to_move
    rows = [pits[0:9], pits[9:18]]
    f = []
    for i in range(9):
        f.append(i * 14 + count_bucket(rows[me][i]))
    for i in range(9):
        f.append(126 + i * 14 + count_bucket(rows[opp][i]))
    f.append(252 + min(8, kazan[me] // 10))
    f.append(261 + min(8, kazan[opp] // 10))
    f.append(270 + _tuz_rel(tuzdyk, me))
    f.append(280 + _tuz_rel(tuzdyk, opp))
    f.append(290 + (sum(pits) % 2))
    assert len(f) == 23
    return f


def phase_bucket(pits) -> int:
    total = sum(pits)
    if total >= 121:
        return 0
    if total >= 81:
        return 1
    if total >= 41:
        return 2
    return 3


def pos_string(pits, kazan, tuzdyk, to_move) -> str:
    tw = -1 if tuzdyk[0] is None else tuzdyk[0] - 9
    tb = -1 if tuzdyk[1] is None else tuzdyk[1]
    return (",".join(map(str, pits[0:9])) + "/" + ",".join(map(str, pits[9:18])) +
            f"/{kazan[0]},{kazan[1]}/{tw},{tb}/{to_move}")


TOTAL_STONES = 162

# tuzdyk convention shared by build_features/pos_string above (== 9qum's own wire
# format, see tools/9qum/README.md's TFEN note): tuzdyk[0] (white's tuzdyk) is an
# ABSOLUTE pit index 9..17 -- a pit on BLACK's row; tuzdyk[1] (black's tuzdyk) is an
# absolute pit index 0..8 -- a pit on WHITE's row. `None` or a negative value means "no
# tuzdyk on that side".
_TUZDYK_BOUNDS = ((9, 17), (0, 8))


class InvalidPositionError(ValueError):
    """A hand-written or harvested position violates the game's physical invariants.

    This guards against exactly the incident recorded in
    .superpowers/sdd/2026-07-31-beat-9qum-phase-a/progress.md: a hand-typed position
    summing to 105 stones (not 162) was used as evidence for a scale-mismatch
    diagnosis and had to be retracted after the fact. validate_position() below makes
    that class of mistake fail loudly at the point a position is constructed, not
    after it has already been used as evidence.
    """


def validate_position(pits, kazan, tuzdyk, to_move=None):
    """Check a togyzkumalaq position against the game's physical invariants:

      1. all 18 pit counts plus both kazans sum to exactly TOTAL_STONES (162) --
         stones only ever move between pits and kazans, never created or destroyed;
      2. every count (each of the 18 pits, each of the 2 kazans) is non-negative;
      3. each tuzdyk index is either absent (None or negative) or a legal pit on the
         side it claims to belong to (see _TUZDYK_BOUNDS above) -- e.g. a tuzdyk index
         of 3 for tuzdyk[0] (white's tuzdyk, which must be a pit on BLACK's row, 9..17)
         is impossible, even though 3 would be a perfectly legal pit for tuzdyk[1].

    Raises InvalidPositionError with a specific, human-readable message describing the
    first violation found. Returns None (does nothing) when the position is valid.
    `to_move`, if given, must be 0 or 1; pass None to skip that check (e.g. when the
    caller doesn't track it).

    This is the ONE place both the Python tooling (tools/9qum/match.py's opening
    suite, ad-hoc diagnostics) and its tests validate a position -- see
    test_features_v2.py for the acceptance/rejection cases.
    """
    if len(pits) != 18:
        raise InvalidPositionError(f"expected 18 pit counts, got {len(pits)}: {pits!r}")
    if len(kazan) != 2:
        raise InvalidPositionError(f"expected 2 kazan counts, got {len(kazan)}: {kazan!r}")
    for i, c in enumerate(pits):
        if c < 0:
            raise InvalidPositionError(f"pit {i} has a negative count: {c}")
    for side, c in enumerate(kazan):
        if c < 0:
            raise InvalidPositionError(f"kazan {side} has a negative count: {c}")

    total = sum(pits) + sum(kazan)
    if total != TOTAL_STONES:
        raise InvalidPositionError(
            f"stones sum to {total}, not {TOTAL_STONES} (pits sum={sum(pits)}, "
            f"kazan sum={sum(kazan)}) -- this position cannot arise from real play"
        )

    tuz = list(tuzdyk) if tuzdyk is not None else [None, None]
    if len(tuz) != 2:
        raise InvalidPositionError(f"expected 2 tuzdyk entries, got {len(tuz)}: {tuz!r}")
    for side, t in enumerate(tuz):
        if t is None or t < 0:
            continue
        lo, hi = _TUZDYK_BOUNDS[side]
        if lo <= t <= hi:
            continue
        other_lo, other_hi = _TUZDYK_BOUNDS[1 - side]
        if other_lo <= t <= other_hi:
            raise InvalidPositionError(
                f"tuzdyk[{side}] = {t} is not a legal pit for side {side} "
                f"(expected {lo}..{hi}) -- looks like it belongs on the OTHER side "
                f"(tuzdyk[{1 - side}], legal range {other_lo}..{other_hi})"
            )
        raise InvalidPositionError(
            f"tuzdyk[{side}] = {t} is not a legal pit for side {side} "
            f"(expected {lo}..{hi} or None/negative for 'no tuzdyk')"
        )

    if to_move is not None and to_move not in (0, 1):
        raise InvalidPositionError(f"to_move must be 0 or 1, got {to_move!r}")
