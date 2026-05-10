# Web-v2 Smoke Test Report

**Date:** 2026-05-10  
**Branch:** rust-mcts  
**Task:** 22 — End-to-end smoke test (automated programmatic verification)

---

## Results Summary

| Step | Check | Result |
|------|-------|--------|
| 1 | Engine binary exists | PASS |
| 2 | Backend pytest | PASS — 75 passed |
| 3 | Frontend build | PASS (after tsconfig fix — see note) |
| 4 | Frontend tests | PASS — 18 passed |
| 5 | Backend health endpoint | PASS |
| 6 | End-to-end curl flow | PASS |

**Overall: DONE**

---

## Detail

### Step 1: Engine binary
```
-rwxrwxr-x 2 nurlykhan nurlykhan 677824 Apr 29 13:48
/home/nurlykhan/9QumalaqV2/engine/target/release/togyzkumalaq-engine
```

### Step 2: Backend pytest
```
75 passed, 9 warnings in 16.27s
```
9 warnings are non-fatal (short JWT secret key in test config, httpx cookie deprecation).

### Step 3: Frontend build

**Bug found and fixed:** `tsconfig.app.json` was missing `"types": ["vitest/globals"]`, causing
`tsc -b` (used by `npm run build`) to fail with TS2582 errors on test files. The `typecheck`
script (`tsc --noEmit`) passed because it uses a different invocation path.

**Fix applied:** Added `"types": ["vitest/globals"]` to `compilerOptions` in
`web-v2/frontend/tsconfig.app.json`.

After fix:
```
✓ 228 modules transformed.
dist/index.html                   0.46 kB │ gzip:   0.30 kB
dist/assets/index-BQ3fugls.css   12.52 kB │ gzip:   3.41 kB
dist/assets/index-Usj73e33.js   432.01 kB │ gzip: 136.56 kB
✓ built in 1.20s
```

### Step 4: Frontend dist/ size
```
456K    web-v2/frontend/dist/
```
Contents: `index.html`, `vite.svg`, `assets/` (1 CSS + 1 JS bundle).

### Step 5: Frontend tests
```
Test Files  5 passed (5)
      Tests  18 passed (18)
   Duration  1.31s
```
Non-fatal stderr warnings about `act(...)` in useGameSocket tests (React Testing Library advisory).

### Step 6: Backend health
```json
{"status": "ok"}
```

### Step 7: End-to-end curl flow

All sub-steps passed:

1. **Anon session init** — `{"kind":"anon","user":null,"anonId":"..."}` ✓
2. **Create game** — `id=3, status=active, currentPly=0` ✓
3. **Play move 0** — `currentPly=1, engineThinking=true` ✓
4. **Reload (GET game)** — `id=3, currentPly=1, status=active` (persisted) ✓
5. **Resign** — `status=finished, result=win_black, resultReason=resign` ✓

---

## Errors / Notes

- `npm run build` failed before the tsconfig fix (see Step 3). The fix is minimal and correct.
- `npm run typecheck` was already passing, so this was a build-only gap.
- 9 pytest warnings are non-blocking (test JWT key length, httpx deprecation).
- React `act(...)` warnings in socket hook tests are advisory, not failures.
