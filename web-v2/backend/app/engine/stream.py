"""
Parse output lines from the Togyzkumalaq engine in serve mode.

Protocol notes (see protocol_notes.md):
- The engine does NOT emit `info` lines during search (silent mode).
- A `go` command yields exactly ONE response line: either `bestmove ...` or `terminal ...`.
- `bestmove` carries: move_index (0-based int), score, depth, nodes, time, nps
- `terminal` carries: result string (white_win / black_win / draw / unknown)
- `ready` and `pong` are handshake responses, not search results.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class InfoLine:
    """Placeholder for UCI-style `info` lines.

    The engine does not emit info lines in serve mode.  This dataclass is kept
    for structural compatibility with the plan's interface; it will never be
    returned by parse_line() under normal operation but can be used if a
    future engine version adds streaming info.
    """

    depth: int | None = None
    cp: int | None = None
    pv: list[str] = field(default_factory=list)
    nodes: int | None = None
    time_ms: int | None = None


@dataclass
class BestMove:
    """Engine's chosen move after a `go` command.

    Attributes
    ----------
    move:
        0-based pit index (0–8).  Add 1 to get the human-readable pit number.
    score:
        Search score in centipawn-like units (positive = good for side to move).
    depth:
        Search depth reached.
    nodes:
        Nodes searched.
    time_ms:
        Elapsed search time in milliseconds.
    nps:
        Nodes per second.
    """

    move: int
    score: int = 0
    depth: int = 0
    nodes: int = 0
    time_ms: int = 0
    nps: int = 0


@dataclass
class TerminalResult:
    """Engine detected a terminal position (game over).

    Attributes
    ----------
    result:
        One of: ``"white_win"``, ``"black_win"``, ``"draw"``, ``"unknown"``.
    """

    result: str


def parse_line(line: str) -> InfoLine | BestMove | TerminalResult | None:
    """Parse one text line from the engine's stdout.

    Returns
    -------
    BestMove
        When the line starts with ``bestmove``.
    TerminalResult
        When the line starts with ``terminal``.
    InfoLine
        When the line starts with ``info`` (future-proofing; not emitted in
        practice).
    None
        For ``ready``, ``pong``, empty lines, or any unrecognised token.
    """
    line = line.strip()
    if not line:
        return None

    if line.startswith("bestmove "):
        return _parse_bestmove(line)

    if line.startswith("terminal "):
        parts = line.split(maxsplit=1)
        result = parts[1] if len(parts) > 1 else "unknown"
        return TerminalResult(result=result)

    if line.startswith("info "):
        return _parse_info(line)

    # ready, pong, error — not search results
    return None


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _parse_bestmove(line: str) -> BestMove:
    """Parse: bestmove <idx> score <s> depth <d> nodes <n> time <t> nps <nps>"""
    tokens = line.split()
    bm = BestMove(move=0)
    i = 0
    while i < len(tokens) - 1:
        tok = tokens[i]
        nxt = tokens[i + 1]
        if tok == "bestmove":
            try:
                bm.move = int(nxt)
            except ValueError:
                pass
            i += 2
        elif tok == "score":
            try:
                bm.score = int(nxt)
            except ValueError:
                pass
            i += 2
        elif tok == "depth":
            try:
                bm.depth = int(nxt)
            except ValueError:
                pass
            i += 2
        elif tok == "nodes":
            try:
                bm.nodes = int(nxt)
            except ValueError:
                pass
            i += 2
        elif tok == "time":
            try:
                bm.time_ms = int(nxt)
            except ValueError:
                pass
            i += 2
        elif tok == "nps":
            try:
                bm.nps = int(nxt)
            except ValueError:
                pass
            i += 2
        else:
            i += 1
    return bm


def _parse_info(line: str) -> InfoLine:
    """Parse a UCI-style `info` line (not currently emitted by the engine)."""
    tokens = line.split()
    out = InfoLine()
    i = 1
    while i < len(tokens):
        tok = tokens[i]
        if tok == "depth" and i + 1 < len(tokens):
            try:
                out.depth = int(tokens[i + 1])
            except ValueError:
                pass
            i += 2
        elif tok == "score" and i + 2 < len(tokens) and tokens[i + 1] == "cp":
            try:
                out.cp = int(tokens[i + 2])
            except ValueError:
                pass
            i += 3
        elif tok == "nodes" and i + 1 < len(tokens):
            try:
                out.nodes = int(tokens[i + 1])
            except ValueError:
                pass
            i += 2
        elif tok == "time" and i + 1 < len(tokens):
            try:
                out.time_ms = int(tokens[i + 1])
            except ValueError:
                pass
            i += 2
        elif tok == "pv":
            out.pv = tokens[i + 1:]
            break
        else:
            i += 1
    return out
