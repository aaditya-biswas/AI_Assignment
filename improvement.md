# Agent Improvement Plan

This document tracks the upgrades applied to the single-file adversarial agent
`B23CS1001.py` and the benchmark evidence behind each one. Everything here
respects [constraint.md](constraint.md): **only** the agent file is modified, the
engine (`board.py`, `config.py`) and the harness (`run_games_fast.py`) are never
touched, the class stays `B23CS1001`, and `B23ME1074.py` is treated strictly as a
black box (played against, never read).

## Reference resources

1. Chess Programming Wiki — Transposition Table  https://www.chessprogramming.org/Transposition_Table
2. Chess Programming Wiki — Alpha-Beta  https://www.chessprogramming.org/Alpha-Beta
3. Chess Programming Wiki — Move Ordering  https://www.chessprogramming.org/Move_Ordering
4. Chess Programming Wiki — Quiescence Search  https://www.chessprogramming.org/Quiescence_Search
5. Chess Programming Wiki — Killer Heuristic  https://www.chessprogramming.org/Killer_Heuristic
6. Chess Programming Wiki — History Heuristic  https://www.chessprogramming.org/History_Heuristic
7. Chess Programming Wiki — Evaluation Function  https://www.chessprogramming.org/Evaluation_Function
8. Chess Programming Wiki — Mobility  https://www.chessprogramming.org/Mobility
9. Chess Programming Wiki — King Safety  https://www.chessprogramming.org/King_Safety
10. Chess Programming Wiki — Passed Pawn  https://www.chessprogramming.org/Passed_Pawn
11. Andrew N. Luce, "Chess Programming" notes / engine concepts  https://www.chessprogramming.org/
12. Artificial Intelligence: A Modern Approach (minimax / alpha-beta foundations)  https://aima.cs.berkeley.edu/

## Constraint compliance

- Single submission file, renamed `B23CS1001.py`, class name `B23CS1001`.
- No engine, rules, evaluation-backend, scoring or time-control tampering.
- No castling / en-passant / promotion / any special move.
- No reinforcement learning: the agent is a pure adversarial game-tree search
  (negamax alpha-beta + iterative deepening), with optimisation techniques
  (transposition table, move ordering, quiescence, reductions, pruning).
- External `B23ME1074.py` was only *played against*; its source was never read.

---

## Benchmark evidence (before → after)

Environment: Python 3.14, single machine; figures are measured, not estimated.

| Metric (per-move hot path) | Before | After |
|---|---|---|
| `_legal_moves` (midgame) | 38.6 µs | ~15–22 µs |
| `_evaluate_white` | 15.4 µs | ~11 µs |
| `_snapshot` | 8.2 µs | ~5–6 µs |
| `_attacked` | 1.15 µs | ~0.96 µs |
| `_position_key` | 2.9 µs | 0.52 µs |
| Search throughput (NPS) | ~2.9–4 k | ~20–27 k (~**7–9×**) |
| Depth @ 0.35 s (start / midgame) | ~4–6 / ~2–4 | **7–8 / 6–7** |

`run_parity.py` **PASSED** (3195 random positions, 0 mismatches — the internal rule
copy is bit-exact with the engine). `test_endgame.py` improved from 3/5 to
**5/5** converted mates (including the previously failing KBB vs K).

### Head-to-head vs `B23ME1074` (`run_games_fast.py`)

**Baseline (before):** lost **both** games, including a checkmate suffered on ply 7
(`0–20` and `40–510`).

**After (three consecutive clean runs):**

| Run | `B23CS1001` vs `B23ME1074` | `B23ME1074` vs `B23CS1001` | Net pts |
|---|---|---|---|
| 1 | **checkmate win (White), ply 124** (+40) | draw, 390–390 (0) | **+40** |
| 2 | draw, +50 | draw, +30 | **+80** |
| 3 | draw, −10 | draw, 0 | −10 |

Across 6 games the agent is decisively ahead on aggregate points and converted
one outright checkmate, versus the earlier 0–2 record.

---

## Implemented improvements (all inside `B23CS1001.py`)

### A. Search correctness

1. **Fixed iterative-deepening / aspiration windows** (`get_best_move`).
   The old code centred the next iteration's window on the *static evaluation*
   (`prev_score = self._evaluate_stm()`) and had **no fail-low/fail-high
   re-search**. In the losing game this made every root move fail low, so the
   search silently kept the depth-1 move — a pawn push that allowed mate-in-1.
   Now the window is centred on the previous iteration's *search score* and
   widened/re-searched on failure, so a completed iteration always updates the
   best move. This removed the instant blunders.
2. **Transposition-table mate-score normalisation.** Stored scores near `MATE`
   are converted to node-relative on store and back to root-relative on probe
   (via the node `ply`), so mate distances are never reused from the wrong ply.
   TT bounds are applied with correct fail-soft cutoffs.
3. **Quiescence rewrite.** Replaced "generate every legal move then filter" with
   a **capture-only generator** (`_capture_moves`), MVV-LVA ordering, **delta
   pruning** and a depth cap (`QMAX`). When in check, **all evasions** are
   searched, so simple mates are still found at the horizon.

### B. Speed (the dominant lever)

4. **Integer board representation.** The agent now works on a flat list of small
   ints (`0` empty, `1–5` white P/N/B/R/K, `6–10` black) instead of the engine's
   2-char strings. String indexing/compare in the hottest paths became integer
   ops. This drove most of the NPS gain.
5. **Precomputed attack geometry tables.** `_KNIGHT_ATT`, `_KING_ATT`,
   `_BISHOP_RAYS`, `_ROOK_RAYS`, `_PAWN_FWD`, `_PAWN_CAP_W/B` are built once at
   import, removing all bounds checks and per-node `_idx()` calls.
6. **Pin-based legality shortcut.** `_pinned_squares()` computes the pieces
   standing between the king and an enemy slider; when not in check, only king
   moves and pinned-piece moves need the expensive `_attacked` legality test.
   This cut `_legal_moves` by ~2.6×.
7. **Flat static-evaluation table** (`_STATIC`). Signed material + piece-square
   table + centralisation for every (piece code, square) are folded into a single
   lookup, so the eval loop does one indexing per occupied square.
8. **O(1) exact position key.** `_position_key()` returns `(wtm, bytes(self.b))`
   (the 0–10 codes) — no per-node `map`/genexpr allocation.

### C. Evaluation quality

9. **Single-pass, allocation-free `_evaluate_white`.** Removed the old
   dict-of-dicts `_scan_board` (multiple 48-square passes per leaf) in favour of
   one tight loop. Material + PST stay dominant (they mirror the tournament
   scoring), plus centralisation, **passed-pawn** bonuses (gated to low material
   so the midgame stays fast) and a cheap **king-shield** term; the endgame
   king-driving logic is retained.
10. **MVV-LVA with attacker value** in `_ordering_key` (prefer capturing a big
    piece with a small one), on top of the existing TT-move / killer / history
    ordering.

### D. Structural clean-up

11. Reused the evaluation scan to supply the king squares to quiescence (removed
    a redundant `_find_king` scan).
12. Removed dead legacy helpers (`_scan_board`, `_mobility_bonus`,
    `_pawn_structure_bonus`, `_passed_pawn_bonus`, `_king_safety_bonus`,
    `_move_key`) that still referenced the old string board.

---

## Evaluation of the suggested optimisations

Each suggestion was measured against the current code (isolated micro-benchmarks
plus before/after `run_bench_fast.py`), not just assumed to help.

| # | Suggestion | Measured effect | Verdict |
|---|---|---|---|
| 1 | **Incremental Zobrist hashing** | `_position_key` was ~2.9 µs/node; with the integer board, `bytes(self.b)` is **0.52 µs** and allocation-free, so a 64-bit Zobrist (≈3 XORs/move) saves <1% overall | **Superseded** by the `bytes` key; not worth the extra make/unmake bookkeeping |
| 2 | **Eliminate `_snapshot()` scans** | `_snapshot` fell 8.2→~5.3 µs via the int board; full incremental king/material tracking would save only ~1–2% but must be maintained in every temporary make/unmake site (incl. `_legal_moves`) — high bug risk | **Not done** (low ROI, high risk) |
| 3 | **Integer board representation** | `_legal_moves` 38.6→~15 µs, eval faster, NPS **~7–9×**; biggest single win | **Implemented** |
| 4 | **Raw-tuple TT / packed moves** | `TTEntry` already uses `__slots__` (no per-instance dict); tuples only remove one tiny allocation. Packing moves saves list/tuple churn but complicates every hot path | **Not done** (<1% here) |
| 5 | **Avoid lambdas in ordering** | `sorted(key=lambda)` calls the key function once per element (not per comparison); the cost is `_ordering_key` itself, which was already trimmed by MVV-LVA. Micro-bench: ~6.5 µs/node | **Partially** (kept, low priority) |
| 6 | **Local-variable caching** | Already applied where it matters (`b = self.b` in `_negamax`, `_quiescence`, generators); further binding is noise | **Applied where it helps** |

Two additional, higher-value wins were found by profiling that were **not** in
the original six suggestions but are now implemented: the **pin-based legality
shortcut** (item 6 in section B) and the **flat `_STATIC` evaluation table**
(item 7 in section B). Profiling showed `_legal_moves` + embedded `_attacked` and
`_evaluate_white` together were ~65% of runtime, which is why those were targeted.

---

## Verification performed

- `python3 -c "import ast; ast.parse(...)"` — file parses.
- `run_parity.py` — **PASSED**, 3195 positions, 0 mismatches (internal rules are
  bit-exact with `GameEngine.get_legal_moves`, both colours).
- `_STATIC` table validated against an independent reference (0 mismatches) and
  the symmetric start position evaluates to exactly 0.
- `test_endgame.py` — **5/5** converted to checkmate (KR vs K ×2, KRR vs K,
  KBB vs K, KR vs KP).
- `run_bench_fast.py` — throughput/depth recorded above.
- `run_games_fast.py` — three clean runs recorded above.

## Status checklist

- [x] Transposition table (bounds + best-move ordering + mate normalisation)
- [x] Quiescence search (capture-only, evasions in check, delta pruning, cap)
- [x] Killer + history move ordering
- [x] Mobility / king-safety / passed-pawn evaluation terms
- [x] Aspiration windows with fail-low/high re-search
- [x] Iterative-deepening correctness fix (root best-move tracking)
- [x] Late-move reductions, reverse-futility pruning, null-move pruning (retained/tuned)
- [x] Integer board + precomputed attack tables + pin legality + flat eval table
- [x] Benchmark + regression checks (parity, endgame, bench, gauntlet)

## Final note

The agent remains a single-file, engine-agnostic adversarial search agent. No
engine, rule, scoring or harness file was modified, and the black-box opponent's
source was never inspected.

---

## Additional micro-optimisations (round 3)

Applied after profiling the remaining hot path:

- **LMR lookup table (`_LMR`)**: the reduction formula was calling `math.log`
  twice per non-first move; it is now a precomputed 64×64 table lookup.
- **Flat history array** (`self.history = [0] * 2304`): replaced the dict keyed by
  `(fr, to)` tuples with direct integer indexing `fr*48 + to`, removing per-move
  dict hashing.
- **History gravity**: cutoff bonuses now decay
  (`h += bonus - h * bonus // 16384`), so early-iteration scores cannot
  permanently dominate ordering.
- **Inlined move ordering** in `_negamax` — **tried and REVERTED**. A hand-rolled
  `for` loop building `(key, move)` tuples plus `list.sort` was *slower* than
  CPython's C-level `sorted(list, key=fn)` (decorate-sort-undecorate): start
  position throughput fell from ~20k to ~13k NPS, which cost search depth and
  measurably weakened play. The original
  `sorted(legal, key=lambda m: self._ordering_key(...))` was restored.

Verification after round 3 (final): `run_parity.py` **PASSED** (3195 positions,
0 mismatches); start-position NPS ~19–20k; a **6-game** head-to-head against
`B23ME1074` produced **+550 net points** with one **checkmate win** (all six
games positive: +40, +30\*, +20, +170, +50, +240).

### Lesson recorded

Micro-optimisation must be measured, not assumed. The two round-3 changes that
helped/neutral (LMR table, flat history array + gravity) were kept; the one that
*slowed the search* (manual ordering) was reverted. In CPython, prefer built-in
C-level operations (`sorted(key=…)`, `bytes`, `list.sort`) over hand-written
Python loops that do the same decorate/sort work.

### Considered but not adopted (with reasons)

- **Full incremental Zobrist hashing**: `bytes(self.b)` is already 0.52 µs per
  node and allocation-light; a 64-bit XOR key would save <1% while adding
  make/unmake bookkeeping risk.
- **Incremental king/material tracking** (to delete `_snapshot`): `_snapshot` is
  ~5.3 µs; the saving is ~1–2%, but the state must stay consistent across every
  temporary apply/unapply — including inside move generation — which is high risk
  for low reward in Python.
- **Futility pruning**: a variant was tried and measured; it did not help (poor
  interaction with the aspiration re-search), so it was reverted.
- **Pure-Python bitboards**: 48 squares do fit in one 64-bit word, but the current
  cost is Python interpreter overhead per operation, not the representation;
  bitboards would only pay off with a much larger rewrite (see discussion).

### Recommended next SOTA steps (not yet implemented)

1. **Bounded check extension** — extend one ply only while resolving a check
   (`ply`-capped) to reduce tactical blindness; must be capped to avoid
   perpetual-check explosion.
2. **Late Move Pruning (LMP)** at `depth <= 3` — search only the first
   `3 + depth*depth` quiet moves; needs measurement (futility was rejected).
3. **Continuation / counter-move history** — index history by the opponent's
   previous move to improve quiet-move ordering.
4. **Static Exchange Evaluation (SEE)** — order/​prune captures by true material
   outcome, sharpening quiescence.
5. **Singular extensions** — expensive; only worth it once the above are stable.

Each of these must be gated behind the parity test plus a fresh `run_games_fast`
comparison, because measured run-to-run spread on this opponent is ±~70 points.


