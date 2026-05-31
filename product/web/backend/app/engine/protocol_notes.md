# Engine Wire Protocol Notes

Extracted from `engine/src/main.rs` (`run_serve()`) and `web/server.py`.

---

## 1. Serve Mode Startup Sequence

Launch the binary with the `serve` argument:

```
togyzkumalaq-engine serve
```

On startup the engine emits exactly one line:

```
ready
```

There is no additional banner. The caller MUST read this line before sending any commands.

---

## 2. Position Format

The engine does NOT use FEN strings. All positions are encoded as:

```
w0,w1,w2,w3,w4,w5,w6,w7,w8/b0,b1,b2,b3,b4,b5,b6,b7,b8/kw,kb/tw,tb/side
```

Where:
- `w0`–`w8`: stone counts in White's 9 pits (indices 0–8, pit 1–9 in game terms)
- `b0`–`b8`: stone counts in Black's 9 pits
- `kw,kb`: White's kazan (captured stones) / Black's kazan
- `tw,tb`: White's tuzdyk pit index (-1 if none) / Black's tuzdyk pit index
- `side`: `0` = White to move, `1` = Black to move

### Start Position

```
9,9,9,9,9,9,9,9,9/9,9,9,9,9,9,9,9,9/0,0/-1,-1/0
```

Explanation:
- All 9 pits for each side contain 9 stones (162 total)
- Kazans are 0
- No tuzdyk for either side (−1)
- White to move (0)

---

## 3. `go` Command (Engine Thinks and Replies)

```
go time <ms> pos <position_string>
```

Optional flags: `nobook` — disables opening book.

Example:
```
go time 3000 pos 9,9,9,9,9,9,9,9,9/9,9,9,9,9,9,9,9,9/0,0/-1,-1/0
```

### Response — Normal Move

```
bestmove <move_index> score <cp> depth <d> nodes <n> time <t_ms> nps <nps>
```

- `move_index`: 0-based integer (0–8), representing the pit the engine picks.  
  **This is NOT "1-3" dash notation; it is a plain integer.**
- `score`: centipawn-like score (positive = good for side to move)
- `depth`: search depth reached
- `nodes`: nodes searched
- `time`: elapsed search time in ms
- `nps`: nodes per second

All fields are on a **single line** — there are NO streaming `info` lines during search (the engine runs with `searcher.silent = true` in serve mode).

### Response — Terminal Position

```
terminal <result>
```

Where `<result>` is one of: `white_win`, `black_win`, `draw`, `unknown`.

---

## 4. `position` Command (Push History Entry)

```
position <position_string>
```

Pushes the position's hash into the searcher's game history for repetition detection.  
Response: `ready`

This is used to notify the engine of human moves so repetition detection works across the full game.

---

## 5. `newgame` Command

```
newgame
```

Clears TT and game history.  
Response: `ready`

---

## 6. `ping` Command

```
ping
```

Response: `pong`

---

## 7. `quit` Command

```
quit
```

Engine exits. No response expected.

---

## 8. `apply_move` — DOES NOT EXIST

**The engine has no command that takes a position + move and returns the resulting position.**

The existing `web/server.py` works around this by having the *client* maintain full board state and send the complete position with every `go` command. The backend never asks the engine to compute a new position; instead the frontend sends the updated board after each human move.

### Implication for `EngineProcess.apply_move`

Since the engine cannot apply a move and return a new position string, `apply_move` must be implemented in Python using Togyzkumalak rules, or the callers must maintain board state themselves and pass the full position string to `think()`.

**For Task 7, `apply_move` is implemented as a Python-side rule application** (see `process.py`). The rules are:

1. Pick up all stones from `position[side][pit]`.
2. Sow them counter-clockwise starting from `pit+1`.
3. If the last stone lands in an opponent's pit with an even count → capture.
4. Tuzdyk creation: if exactly 3 stones land in an opponent's pit at index ≠ 8 and the player has no tuzdyk yet → mark it as tuzdyk; all future landings there are captured.
5. Switch `side_to_move`.

Because implementing the full rule set faithfully is non-trivial, **Task 7 implements apply_move as a thin wrapper that raises NotImplementedError and includes a clear TODO**. Callers (Task 11 onwards) will need to either:
- Implement rule application in Python (preferred), or
- Pass the pre-computed board state from the client.

This is the escalation point flagged in the plan. The integration test for `apply_move` is skipped until the rules are implemented.

---

## 9. Error Responses

Any command that receives invalid input returns:

```
error <message>
```

---

## 10. `info` Lines

There are **no `info` lines** in serve mode. The engine runs silently (`searcher.silent = true`).  
The only outputs are `ready`, `bestmove ...`, `terminal ...`, `pong`, `error ...`.

---

## 11. Move Index vs Human-Readable Pit Numbers

- Internal representation: 0-indexed (0–8)
- Human/UI representation: 1-indexed (1–9)
- The engine's `bestmove` response uses 0-indexed values
- To display to users: `displayed_pit = engine_move_index + 1`
