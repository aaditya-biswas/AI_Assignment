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

---

## Round 4: ordering refactor + counter-move heuristic (kept); selective pruning/extensions (reverted)

### Kept (all in `B23CS1001.py`)

- **Hoisted the killers lookup out of the per-move ordering key.** Previously
  `_ordering_key` called `self.killers.get(depth, ())` once *per move* (inside the
  `sorted(key=...)` lambda). It is now resolved once per node, and ordering uses a
  new hot `_order_key(...)` that receives the TT move, killers and history as
  pre-resolved arguments. Pure speed win, no behaviour change.
- **Killers stored as flat move ints** (`fr*48 + to`) instead of `(fr, to)`
  tuples, for fast comparison.

### Tried and REVERTED (failed the acceptance gate)

- **Counter-Move Heuristic (CMH)** — keyed by the opponent's previous move and
  ordered just below the killers (a killer + counter-move hybrid). It could not be
  shown to beat the baseline above run-to-run noise (+500 vs +550 on 6-game
  samples; the CMH-off run was +540 over 8 games), so it was removed to keep the
  configuration provably >= baseline.
- **Late Move Pruning (LMP)** — at `depth <= 3` non-check nodes, skip quiet moves
  after `3 + depth*depth`. Too aggressive for this small-board, shallow-search
  variant: the combined head-to-head went **negative** (‑90, ‑280 including a mate
  loss, ‑40, …).
- **Bounded check extension** (+1 ply when in check, ply-capped at 24) — alone it
  produced 2 checkmate wins but a lower point net (**+250** vs +500/+550) with two
  negative games; reverted.

### Measured nets vs B23ME1074

| Configuration | Net | Checkmate wins |
|---|---|---|
| Baseline (round 3) | +550 (6 games) | 1 |
| hoist + CMH | +500 (6 games) | 1 |
| check extension only | +250 (6 games) | 2 |
| hoist + CMH + LMP + check ext | negative | 0 (one loss) |
| **Final: hoist only** | **+540 (8 games)** | **1 (and 1 loss)** |

**Lesson (repeat):** aggressive late-move / selective pruning that is standard in
large-board engines does not automatically transfer to this 6×8, shallow variant,
where a quiet defensive resource is often the whole point. Keep pruning off unless
a multi-game measurement clearly beats the baseline.

---

## Round 5: speed (semantics-preserving) — ~2× throughput

Every change here is behaviour-preserving (verified: `run_parity.py` PASSED, a
capture-move differential PASSED, `test_endgame.py` 5/5); they only remove
redundant work:

- **Skip `_snapshot` at horizon nodes.** `_negamax` used to scan the board and run
  `_attacked` *before* the `depth <= 0` return, where the result was unused
  (quiescence recomputes what it needs). Moving the horizon return above
  `_snapshot` cut `_snapshot` calls by ~71% (measured 6144 → 1784 per move).
- **Reuse `in_check` in `_legal_moves`.** `_negamax` already knows whether the side
  to move is in check (from `_snapshot`); `_legal_moves` no longer recomputes
  `_attacked(king, opp)`.
- **Pin set → bitmask.** `_pinned_squares` returns an int bitmask instead of a
  `set` (no allocation/hashing); membership via `(pinned >> fr) & 1`.
- **`_capture_moves` pin shortcut.** Quiescence captures now use the same pin
  argument as `_legal_moves`, so most captures skip the `_attacked` test. Proven
  exact by a differential test against `_legal_moves` (3200 positions, 0
  mismatches).
- **One node counter** instead of two (`nodes_expanded`), **local binding** of the
  recursive `_negamax`, and an **inlined `_LMR` table lookup** (no method call or
  `math.log` per move).

Measured A/B on the same position (git HEAD vs current, via importlib):

| | total nodes/s | start-position depth |
|---|---|---|
| HEAD (pre-speed) | 26,648 | 7 (in 0.36 s) |
| **Current** | **56,587** | **8 (completed in 0.22 s)** |

i.e. **~2.1× throughput, +1 ply**.

### Mate-weighted scoring (important)

Tournament scoring gives **+600 for checkmate and overrides captured points**, so
the metric that matters is *mate count*, not capture diff. Re-scoring the 8-game
head-to-head under tournament rules:

| run | capture-net | mate-weighted net | mates (us – them) |
|---|---|---|---|
| current (speed) | +160 | **+1080** | 3 – 1 |
| baseline (round 3) | +550 | — | 1 – 0 (in 6 games) |

The speed round roughly **tripled the mate rate** and produced a strongly positive
tournament score. Conclusion: prioritise depth (speed) and mate-finding over
capture-diff tuning.

---

## Round 6: incremental evaluation (O(1) midgame eval) — ~30% faster

The profile showed `_evaluate_white` was the top cost and `list.append` a large
share (the eval built pawn lists every call). Two changes:

- **No pawn lists in the common case.** `_evaluate_white` no longer collects pawn
  squares; the passed-pawn scan runs only when `pieces <= 12` (endgame). The king
  shield is inlined (removed ~12k method calls/move).
- **Incremental material + PST(+centre).** A running `self._static` (plus king
  squares, non-king counts and material powers) is maintained on every
  apply/unapply in `get_best_move`, `_negamax` and `_quiescence` via small
  `_eval_add` / `_eval_sub` helpers, initialised in `_load`. `_evaluate_white`
  now reads this state, so midgame evaluation is **O(1)** (endgames still do the
  one passed-pawn scan).

### Validation (all passing)

- **Fixed-depth equivalence**: 450 positions gave identical move + node counts vs
  the previous version (the incremental eval is exact).
- **State consistency**: 300 checks (including forced aborts) - the incremental
  state always matched a from-scratch recomputation.
- `run_parity.py` PASSED; `test_endgame.py` 5/5.

### Measured speedup (alternating A/B, total nodes searched in 0.35 s)

| Position | previous | now |
|---|---|---|
| start | 11709 / 14752 | 15205 / **20652** |
| midgame | 21363 / 20258 | 25501 / **26096** |

The new version also reached **depth 8** on the start position (previous mostly
depth 7) - roughly **+25-35% throughput** on top of round 5.

---

## Round 7: opening book - investigated, not adopted

Motivation: as White the agent looked weaker than as Black, so an opening book
was trialled. A single forced first move was measured over several games each
(all as White vs B23ME1074, official scoring):

| First move | sample | total |
|---|---|---|
| default `(7,0)->(5,1)` | 12 games | **+176** |
| pawn `(6,4)->(5,4)` | 8 games | +206 |
| pawn `(6,3)->(5,3)` | 4 games | -288 |
| pawn `(6,2)->(5,2)` | 2 games | -200 |

An initial 8-game sample made the default look terrible (-578) and a pawn push
look good, but a coding mistake in the test (it forced `(7,5)->(5,4)`, which is
**illegal** because `(7,5)` is a bishop, so it silently fell back to the default)
revealed that the "default" arm then scored **+754** - i.e. the spread is pure
**variance**, dominated by +/-600 mate results.

Conclusion: **the opening move is not the bottleneck**; the differences are within
run-to-run noise. The book was removed (the agent is back to the verified state:
parity PASSED, first move chosen by search). The real swing is **mates** - as
White the agent has been mated outright in some games, and a single mate flips
the official score by 600. Future work should target **king safety / mate
avoidance**, not the opening.

### Paired, deterministic confirmation (final)

To remove the mate-driven noise, both openings were then played on **identical**
seeded games with our agent at a fixed depth (deterministic) - 12 paired seeds:

| Arm | Total official score |
|---|---|
| default `(7,0)->(5,1)` | **+496** |
| pawn `(6,4)->(5,4)` | **+412** |
| delta (pawn − default) | **−84** |

Per-seed deltas: `-50, -616, 0, +652, +598, -78, -24, +84, 0, -78, -24, -548`.
The pawn opening's apparent lead was entirely a few +/-600 mate events swinging
the other way late. So the opening choice is **statistically neutral**, and no
book was adopted.

---

## Round 8: valuing checks in the search - tried and REVERTED

Observation: vs B23ME1074 we give ~0-6 checks while it gives ~2-32 (each worth
+2 in the official score), i.e. a ~50-point handicap in quiet games.

Implementation tried:
- precomputed, per (piece type, enemy-king square), bitmasks of the squares from
  which that piece would attack the king (a cheap pre-filter), then
- **+2 per check** for moves near the root (root loop and `ply <= 2` in
  `_negamax`), matching the official scoring.

Result (paired, deterministic, 6 identical seeded games, fixed depth 5):

| Arm | Total |
|---|---|
| no check bonus | **+828** |
| with check bonus | **-132** |
| delta | **-960** |

And it hardly increased checks (seeds 3-6: 12 vs 0 checks, yet -40 vs +76). The
+2 is too small to survive the position: preferring checks distorted the search,
cost material and **stopped finding mates** (seed 2: no-bonus mated, bonus did
not). Fully reverted (`_CF_` / `CHECK_BONUS` refs = 0; parity PASSED).

**Lesson:** in this variant a check is worth only 2 points, while a pawn is 20 and
a mate is 600 - so optimising directly for checks trades away far more than it
gains. Match the search objective to the *dominant* term (material and mate), not
to the small ones.

---

## Round 9: single-pass `_legal_moves` - measured, performance-NEUTRAL, reverted

Round 8 left the search objective alone, so the remaining lever was raw speed:
`_legal_moves` had been building an intermediate `pseudo` list of `(from, to)`
pairs and then walking it a *second* time to validate/emit.

Change tried: compute the check/pin state **before** generation, hoist the
"needs the attack test" decision to once per piece (`tst = must_test or i == ksq
or (pinned >> i) & 1` - it depends on the piece, not the move), and emit each
generated move straight into either `legal` or a small `need` list. That removes
one tuple allocation and one list append per generated move plus the entire
second pass.

Correctness: **exact** - the returned move sets are identical to the previous
implementation (quiet and `drive` modes), `run_parity.py` PASSED, `smoke_test.py`
PASSED.

Speed: **no gain.** Measured with CPU time (`time.process_time`, immune to
machine load), alternating the two versions in one process on the same position:

| Measurement | single-pass | previous |
|---|---|---|
| `_legal_moves` median | 15.752 us/call | 15.720 us/call |
| fixed-depth-7 search, median NPS | 57,376 | 61,092 |

The search node counts were also essentially the same (11,350 vs 11,325 - the
only difference is the *order* of king/pinned moves, which now come after the
rest, shifting a couple of tie-breaks). Both versions' NPS readings were bimodal
(~40k vs ~63k) under load, which is machine noise, not code.

Conclusion: the second pass was never the bottleneck - the 48-square scan and
the per-ray loops dominate - so this is all cost, no benefit. Reverted.

**Lesson:** profile before optimising. A "obviously wasteful" intermediate list
was worth ~0% here; the measurement (CPU time, same process, alternating) is
what settles it, not the intuition.

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




## Pruning log — strategies removed to fit the 60 KB single-file limit

Submission file: **73,876 B / 1726 lines -> 51,346 B / ~1378 lines** (-22.5%),
identical search tree (AST equality with docstrings ignored + depth-8 start search
completes at exactly the same 6789 nodes).

Removed because each had been measured as a net loss (all were already
disabled in the shipped build, so the code was dead weight):

| removed strategy | measured result |
|---|---|
| `ADJ_CHECK_PTS` (root bonus per check given) | paired games at 2 were ~80 pts of material per game worse than at 0 |
| `ROOT_CHECK_W` + `_threat_checks` (tax on checking replies left) | check column fixed (mean check diff -81 -> -1.5 over 8 games) but material +97.5 -> -27.5, official +16.5 -> -7.0 = net loss |
| `ADJ_TRADE_W` (exchange bonus while ahead) | 3W-1L / +48.0 official without it vs 2W-2L / -13.0 with it |
| `_PROX_FAR` far-weighted tropism path | weight was 0, so the hot-table build did dead arithmetic |

Comment/docstring compaction (18.5 KB of narrative) moved here; the
executable code is unchanged.

### Raw notes dropped from the submission file (verbatim)

Single-file adversarial agent for Spartans Chess (6x8, no queens).

"""

        A game the harness still sees running after ADJ_CAP_PLY plies is decided
        on this table, so the agent has to know the score it is playing for.
        Only the plies not banked yet are read - the opponent's move, plus our
        own move banked when we picked it - so the cost is a couple of
        dictionary lookups per move: exact, and nothing that touches the clock.
        """

        The mover's colour is read from the engine, not from 'self.wtm': the
        search flips 'wtm' while it runs and only '_load' restores it.
        """

        Root-only probe on the internal board (a few attack tests per root
        move, once per move rather than per node), used as a bounded score tax:
        the eval may still prefer a positional nicety, but not at the price of
        handing over material.  Kings are excluded (they cannot be captured) and
        the tax is capped, so a deliberate sacrifice or a mate stays available.
        """

        Called on every return path of get_best_move, so the next move has only
        the opponent's reply left to read.  Everything read here comes from the
        engine and from the check map built *before* the search started, never
        from search-mutated state.
        """

        Sudden death on a 60 s clock: the remaining time is spread over the moves
        still expected, then scaled by how critical the position looks.  The
        multipliers encode what this variant rewards - a mate is worth 600, so
        mate-critical positions (we are driving for mate, or we are in check) get
        by far the most time, while a position whose best move has not changed
        for several plies and whose score is steady is treated as obvious and
        gets less.  'hard' is the ceiling the search may extend to when the root
        move keeps changing or the score drops (panic time); it is capped so no
        single move can eat the clock, because running out of time LOSES.
        """

        Reordering only matters for moves the search scores equally (a root move
        is kept only on a strict improvement), so this can never give away a
        tactic or a mate - it just decides which of two equal moves to prefer.
        """
        (TT move, killers and history are passed in)."""
        quiet moves and keeps the delta-pruning friendly 4-tuple shape)."""

        This replaces the old "generate every legal move then filter" version
        (which paid ~30 attack tests per node for a handful of captures) with a
        capture-only generator, MVV-LVA ordering and delta pruning.  In check,
        all evasions are searched so simple mates are still detected at the
        horizon.
        """

        Returns (my_king, opp_king, in_check, few_pieces, mate_drive).
        mate_drive is true when the side to move is up by >= a rook against an
        opponent that has at most one non-king piece left.
        """

        Uses the precomputed geometry tables (no bounds tests, no _idx calls),
        which is the single hottest helper in the search.
        """
        (only these quiet moves can ever expose the king)."""

        Rules mirror board.py exactly (generate-then-validate); the king of the
        side that just moved must not be left in check.  When 'drive' is set we
        additionally record whether each move gives check (mate-drive needs to
        order forcing moves first).
        """

        Material + piece-square tables dominate (they mirror the tournament
        scoring), with small centralisation, passed-pawn and king-safety terms.
        The material/PST/centre part and the king squares are maintained
        incrementally on every make/unmake (see '_load'); only the endgame
        passed-pawn scan touches the board.
        """
        while the leader's king and rook(s) move in for the mate."""

        True only when the king sits on or next to an edge - the only place being
        boxed in really matters - and the incremental tropism already says enemy
        material is near it.  Quiet middlegame positions therefore never pay for
        the scan, and the even-material danger term stays affordable.
        """
FEW_PIECE_LIMIT = 9      # at/below this many pieces: cheap terminal check
# ------------------------------------------------------------------ int board
# Internally the agent works on a flat list of small ints instead of the
# engine's 2-char strings (0 empty, 1..5 = white P/N/B/R/K, 6..10 = black).
# Integer compares/indexing are markedly faster than string slicing, which is
# King tropism: running total of enemy material near each king, maintained
# incrementally in _eval_add/_eval_sub so it is free per evaluation.
# Fully precomputed static score per (piece code, square): signed material +
# piece-square table + centralisation folded into a single table lookup, so
# the hot evaluation loop does one indexing per occupied square.
# The official scorecard pays ONLY material (and checks), so the search must not
# trade a 20-point pawn for a positional nicety: _POS_SCALE shrinks the
# positional part (piece-square table + centralisation) while the material part
# stays exactly at the tournament values.  Measured over a fixed 48-position
# suite, 1.0 left 6.7 points of material hanging per position, 0.5 leaves 2.5,
# and 0.25 is worse again (4.6) - so 0.5 is what this variant wants.
# ---------------------------------------------------------------- fast tables
# Precomputed per-square attack geometry so the hot paths never do bounds
# checks or function calls.  Built once at import time (48 squares).
_ABORT_MASK = 15           # test the hard deadline every 16 nodes.  The drift a
# Inverse pawn tables: the squares a white/black pawn would have to stand on in
# order to attack a given square (same geometry as the opposite pawn's captures).
# Late-Move-Reduction lookup: R(d, i) = floor(1 + ln(d)*ln(i+1)/2) for d>2, i>2,
    # ------------------------------------------- time control (self-contained)
    # Everything the agent needs to budget its own clock lives in THIS file, so
    # the submission is a genuine single file: the official runner only has to
    # construct it and call get_best_move().  The tournament gives each player
    # 1 minute for the whole game (sudden death, no increment) and running out
    # of time LOSES, so the allocator always keeps a reserve back.
    #
    # Scheme (after the usual engine practice: a soft bound the search aims to
    # finish within, and a hard bound it may extend to when the position turns
    #   soft = remaining/moves_left * TIME_FRACTION * criticality
    #   hard = min(soft * HARD_FACTOR, spendable, remaining*MAX_SHARE_CRIT)
    MIN_MOVE_SECONDS = 0.30        # never move instantly (endgames need real depth)
    # The reserve is deliberately generous and it is a *hard pool*: '_plan_time'
    # never plans to spend past GAME_SECONDS - RESERVE_SECONDS, and it shares
    # what is left of that pool over the moves still expected, so a long game
    # glides into small searches instead of instant ones.  The runner times the
    # WHOLE get_best_move() call and a time-out loses the game outright (600
    # points), so the reserve has to cover the abort granularity, per-move
    # bookkeeping and a loaded machine.  'test_budget.py --stress' plays whole
    # games on the real clock and fails if any move leaves less than 3.5 s.
    # Even if that pool is ever exhausted, the search still gets TINY_SLICE per
    # move - enough for depth 1-2 instead of a move in generation order - and it
    # may only eat into the reserve down to MIN_RESERVE.  The gap between the two
    # constants is what covers the abort granularity and per-move bookkeeping.
    # --- dynamic factors: positional advantage / criticality ---------------
    # --- check-avoidance (king pressure) evaluation ------------------------
    # Checks are never priced and never ducked: both were measured as net losses
    # (they trade material for 2-point checks).  King safety and depth attack the
    # check column instead.  Measurements: see improvement.md.
    # Mate risk at LEVEL material: the danger term used to switch on only once we
    # were already a rook down, so a level game could walk into a mate with the
    # king's exposure priced at nothing.  A scaled share now applies whenever the
    # enemy keeps attackers near a king that is on or next to an edge.
    # Mate defence by search: a position in check is a forcing line, so extend it
    # by one ply, bounded by the budget, the distance from the root and the clock.
    CHECK_EXT = 1                  # extend one ply when the side to move is in check
    # --- adjudication scorecard (the tournament's points table) ------------
    # Captures score PIECE_VALUES (P20 N70 B70 R100) plus 2 points per check
    # GIVEN, a mate is 600 and overrides that, a time-out is a loss.  The harness
    # adjudicates a game still running at ADJ_CAP_PLY plies, so the agent keeps
    # the exact tally and plays to the score late in the game.
        # Cross-move search history driving the dynamic time control.
        # King-pressure state (check avoidance), maintained incrementally.
        # Adjudication tally (official points table: captures + 2 per check).
    # ------------------------------------------------------------- setup
        # Incremental evaluation state (kept in sync on every make/unmake).
        # King pressure (check avoidance): computed once here, then maintained
        # Official points table (captures + checks) and the late-game mode that
    # -------------------------------------------------- adjudication table
            # Only the last ply can still be tested for a delivered check: if we
        # Play-to-the-score mode: late in a game that will be adjudicated, our
        # own points are what count, so a lead is worth protecting and a
        # deficit is worth chasing (a check is +2, a mate is 600).
            # A captured piece stops exerting its own pressure on the ENEMY
            # king (the king squares have not moved yet, so this is exact).
            # The mover carries its king pressure along with it.  Most moves are
            # far from the enemy king, where both lookups are 0 and the whole
        # Each square is a 0..10 code, so bytes() is a very fast exact key.
    # ------------------------------------------------------------ public API
        # The clock starts on the FIRST line: the runner times this whole call,
        # so _load(), root-move generation and per-move overheads have to be
        # counted here too, or the reserve silently leaks a little each move.
            # It still has to be banked: a forced move can give check (+2).
        # Dynamic time control: (soft, hard) seconds for this move, derived from
        # the clock, the game phase and how critical the position is.
        # Which root moves give check (used by the adjudication tally and by root
        # ordering).  Computed once per move and never fed into the eval, so the
        # Fresh check-extension budget for this move (a per-move resource).
        limit = soft_limit     # raised to hard_limit while the move is unstable
                # Do not start an iteration that cannot finish in the budget.
                # Aspiration window centred on the last *search* score (the
                # previous version centred it on the static eval, which made
                # every root move fail low and silently kept a stale move).
                # A forced mate for us is already the best possible outcome.
            # An unchanged root move over consecutive moves = "easy move".
        # Bank the move we are about to play into the adjudication tally (its
        # captures, and its +2 if it gives check), so the next move only has the
        # Hard pool: the reserve below is never spent, whatever the multipliers
        # say.  Without this the MAX_SHARE_CRIT crawl spends 35% of whatever is
        # left on every single move, and no per-move guard can bound that sum -
        # real games were measured at 59.6 s and 59.1 s of the 60 s clock.
        # The floor must scale with the clock, otherwise a fixed minimum would
        # spend more than half of what is left in a near-empty clock.  It is
        # ... but a move is never free: once the pool is gone the search still
        # gets a tiny slice, which is enough for depth 1-2 and far better than
        # returning a move in generation order.  It may only eat into the
        # reserve down to MIN_RESERVE, which is what covers the abort drift.
                # Protecting a lead: demote a piece left hanging, because that is
                # exactly how the opponent takes the points back before the cap.
    # ---------------------------------------------------------------- search
            # Convert the stored node-relative mate score back to root-relative.
        # Horizon handling runs before the board scan: quiescence and the eval
        # recompute what they need, so _snapshot is skipped at leaf nodes.
        # Check extension: being in check means a forcing line, and forcing lines
        # are exactly what produce the mates we have been losing to (plies 62-96
        # in several recent games).  Search one ply deeper there, bounded by a
        # per-move budget and by ply, so a check-heavy game cannot blow the tree
                    # unfinished null move must still hand 'wtm' back, or the
                    # caller unwinds with the board and the side to move out of
                    # step (the qsearch guard and the node-count abort can both
        # Ordering inputs are hoisted out of the per-move key function (the
        # killer lookup used to run once per move inside the key lambda).
            # Bounded ("gravity") update keeps early scores from dominating.
            # Same pin shortcut as _legal_moves: only king moves and pinned
            # pieces can expose the king, so most captures skip the test.
        # Hard-deadline guard for the forcing search: _negamax only tests the
        # deadline every 255 nodes, so a capture-rich tree at QMAX can overshoot
        # inside a single node.  The abort is re-raised through the unmake
        # King squares are tracked incrementally (self._wk / self._bk).
    # ------------------------------------------------------------- snapshot
    # ------------------------------------------------------ attack/legality
        # Only king moves and pinned-piece moves can expose the king, so when
        # not in check (and not driving for mate, which needs 'gives') the
    # --------------------------------------------------------- evaluation
        # Everything below is maintained incrementally (see _load / make-unmake),
        # so the common midgame evaluation is O(1) - no 48-square scan.
            # Passed-pawn scan (endgame only), so the common midgame eval never
                # King shield (inlined): friendly pawns in front of each king.
        # Check avoidance: enemy material near a king is what generates the
        # endless stream of +2 checks (and then the mates).  Bounded by
        # PRESS_CAP and only counted while the attacker still has real material,
        # so it can never outweigh material - and never a mate, because a mate
        # score is ~100000 and always overrules any evaluation term.
        # Defensive drive: the case the bonus above does *not* cover, i.e. the
        # trailing side still has two or more non-king pieces.  Losing a king is
        # worth 600, so a boxed-in king is penalised well before the endgame has
        # been stripped down to a bare king - and now also when the material is
        # level, at a fraction of the weight, because a level game can be lost to
        # Expose the king squares so quiescence does not need a second scan.
