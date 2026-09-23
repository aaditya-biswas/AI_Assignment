# `B23CS1001.py` — part-by-part viva walkthrough

Line numbers refer to the shipped file (1,404 lines / 53 KB).

## 0. The 60-second answer

> It is a single-file adversarial search agent. The engine (`board.py`) stays the
> source of truth: I only read it and ask it for the legal root moves. Internally
> I copy the board into 48 integers once per move, keep an **incremental
> evaluation** that updates in O(1) per make/unmake, and run **iterative-deepening
> negamax** with a transposition table, principal-variation search, null-move
> pruning, late-move reductions, reverse futility and a **quiescence search** that
> also searches real evasions while in check. A **dynamic time control** spreads
> the 60-second clock over the expected 75 moves and reserves ~5 seconds, because
> a time-out is an automatic loss. Ordering (TT move, MVV-LVA, killers, history)
> exists to make alpha-beta actually prune; the king-pressure, mate-drive and
> adjudication terms exist because the official score card pays 600 for mate and
> +2 per check.

## 1. Map of the file

| lines | part |
|---|---|
| 1-32 | imports and constants (piece codes, values, tropism weights) |
| 35-81 | precomputed tables: king tropism `_PROX`, folded static eval `_STATIC` |
| 84-91 | `TTEntry` — one transposition-table record |
| 94-151 | attack/ray tables built once at import time |
| 154-159 | `_AbortSearch` and `_idx` |
| 162-204 | class constants: time control, multipliers, eval knobs |
| 206-231 | `__init__` — per-game and per-move state |
| 233-355 | loading + official-points tracker (`_load`, `_load_adj`, `_adj_points`, `_gives_check`, `_root_hang`, `_adj_bank`) |
| 357-437 | incremental eval + king pressure (`_pressure_on`, `_press_add`, `_eval_add`, `_eval_sub`) |
| 439-588 | `get_best_move` — the per-move driver |
| 590-629 | `_plan_time` — dynamic time control |
| 631-855 | search: `_root_key`, `_null_move_allowed`, `_negamax`, `_order_key`, `_ordering_key`, `_record_killer` |
| 857-1008 | quiescence: `_capture_moves`, `_quiescence` |
| 1010-1214 | primitives: `_snapshot`, `_find_king`, `_attacked`, `_pinned_squares`, `_legal_moves` |
| 1216-1398 | evaluation: `_evaluate_stm`, `_evaluate_white`, `_drive_bonus`, `_edge_danger`, `_king_danger`, `evaluate_board` |

## 2. Constants (lines 12-32)

- `MATE = 100000` — a *search* sentinel, not the tournament value. It only has to
  be bigger than any material total (~600 + checks) so the search always prefers
  mating; the tournament's 600 is scored by the harness, not by me.
- `FEW_PIECE_LIMIT = 9` — "few pieces" threshold used to raise the time multiplier
  and to allow the more expensive endgame eval terms.
- `TT_EXACT / TT_LOWER / TT_UPPER = 0 / 1 / 2` — transposition-table bounds.
- `E, WP..WK, BP..BK = 0..10` — **integer piece codes**, white 1-5, black 6-10.
  Consequences used everywhere: `p < 6` tests "white", `p != E` tests "occupied";
  integers index lists and compare much faster than the `'wP'` strings the engine
  uses, which is why the board is converted once per move (`_load`).
- `_CODE` string→code, `_TYPE` code→piece type (1P 2N 3B 4R 5K), `_VAL` unsigned
  material (matches the official 20/70/70/100), `_VAL_SIGN` white-positive,
  `_PTVAL` by type with king = 0 (kings are never captured in this variant, so a
  king "capture" must never look attractive).
- `_WT` and `_PROX_RANGE = 3` — king-tropism weights per code (pawn 1, knight 2,
  bishop 2, rook 3, king 0) and the radius beyond which a piece does not count.

## 3. Precomputed tables (lines 35-151)

**`_build_prox` (35-49)** builds a 48×48 table: `_PROX[k][s]` = 3/2/1 when the
Chebyshev distance between king square `k` and square `s` is 1/2/3, else 0. This is
the king-tropism kernel used by the pressure term. Built once at import, so the
search pays one list index instead of arithmetic.

**`_build_static` (56-81)** is the performance centrepiece. For every piece code it
precomputes a 48-entry table

```
_STATIC[code][sq] = material_sign + int(0.5 * (PST + centre_bonus))
```

- `_PST_OF` supplies the official PSTs; black is vertically mirrored
  (`_mr = BH - 1 - r`) and negated **at build time**, so no per-node colour
  branch exists.
- For N/B/R a centre bonus `3 - (|r-3| + |c-2|)` is added when `_d < 3`
  (distance from the 6×8 centre), which encourages centralisation.
- `_POS_SCALE = 0.5` halves every positional term. **Measured**: with full-weight
  positional terms the agent hung ~6.7 points of material per position at equal
  depth; at 0.5 it was ~2.5. Material is what the score card pays for, so
  material-first is the rational choice — this is a defensible, measured decision,
  not a guess.
- `int(...)` keeps everything integer, so make/unmake is exact and TT comparisons
  never suffer float noise.

**Attack tables (94-151)** are built by nested loops at import:
`_KNIGHT_ATT[sq]`, `_KING_ATT[sq]` (target tuples, edges filtered),
`_BISHOP_RAYS[sq]`, `_ROOK_RAYS[sq]` (four rays per square, each a tuple of squares
out to the edge), `_PAWN_FWD[sq]` (single push; -1 off-board) and
`_PAWN_CAP_W/B[sq]` (the two diagonal capture targets). The variant has no
castling, en-passant or promotion, so that is the *complete* move grammar.

`_PAWN_CAP_W_inv = _PAWN_CAP_B` (151-152) looks like a bug but is not: to test
"is `sq` attacked by a *white* pawn" I need the squares from which a white pawn
could capture onto `sq`, and that is exactly the set of squares a *black* pawn
attacks from `sq`. So the two tables are each other's inverse.

**`_LMR` (145-151)** precomputes `1 + int(log(depth)*log(move_index+1)/2)` for every
(depth, index) pair, so the hot search path never calls `math.log`.


## 4. `TTEntry` (84-91)

A transposition-table record with `__slots__` (less memory, faster attribute
access) holding `depth`, `flag` (EXACT/LOWER/UPPER), `value` and `best_move` as an
`(from, to)` tuple. Storing the move means the TT doubles as the best
move-ordering hint, which is where most of alpha-beta's pruning power comes from.

## 5. Class constants (162-204)

**Time control** — `GAME_SECONDS = 60` (official clock), `ASSUMED_MOVES = 75` (the
harness adjudicates at ply 150, i.e. 75 moves each), `TIME_FRACTION = 0.90` (never
plan to spend the last 10%), `MIN_MOVES_LEFT = 8`, `MIN_MOVE_SECONDS = 0.30`,
`HARD_FACTOR = 2.5`, `MAX_SHARE = 0.25`, `MAX_SHARE_CRIT = 0.35`,
`RESERVE_SECONDS = 5.0`, `TINY_SLICE = 0.02`, `MIN_RESERVE = 4.5`,
`ITER_PREDICT = 2.5` (projected cost of the next iteration).

**Situational multipliers** — `DRIVE_MULT = 3.0` (a finishable mate is worth 600),
`CHECK_MULT = 1.5` (in check you either find the escape or lose the king),
`FEW_MULT = 1.6`, `WIDE_MULT = 1.1` (`WIDE_ROOT = 25`), `PANIC_MULT = 1.5`
(`PANIC_DROP = 100`), `EASY_MULT = 0.5` (`EASY_STABILITY = 3`, `EASY_DROP = 30`).

**Evaluation knobs** — `PRESS_SCALE = 1`, `PRESS_CAP = 30` (1.5 pawns: the
king-pressure term can never outbid material), `DANGER_EVEN = 0.5`,
`DANGER_PRESS_MIN = 8`.

**Bounded check extension** — `CHECK_EXT = 1`, `CHECK_EXT_PLY = 5`,
`CHECK_EXT_BUDGET = 6`. Measured: at full strength (ply 12, budget 16) it cost a
whole ply and dropped endgame conversion from 5/5 to 3/5; the tightened version
keeps 5/5 and even improves depth at 0.3 s, so the bound is not cosmetic.

**Adjudication tracker** — `ADJ_CAP_PLY = 150`, `ADJ_LATE_PLY = 40`,
`ADJ_LEAD_PTS = 20`, `ADJ_SAFE_PTS = 20`. The harness decides a 150-ply game on
points, so the agent knows the official score it is playing for.

## 6. `__init__` (206-231)

Harness-visible attributes: `engine`, `nodes_expanded` (cumulative, printed by
`game_runner`), `depth` (last completed iteration), and `max_depth = 12` as a cap.
`test_budget` is `None` in real play and is set by the test harness to force an
exact per-move budget (deterministic tests). `tt` and `history` persist across
moves; `history` is *not* cleared but decays through the
`hval + bonus - hval*bonus//16384` update, and `killers` is keyed by depth.
`_press_w/_press_b` are the king-pressure values, `_adj_*` the official-points
tracker, `_root_ck` a per-move cache of "does this root move give check?", and
`_ext_left` the remaining check-extension budget for this move.

## 7. `_load` (233-270)

Called at the start of every `get_best_move`:

1. Flatten `engine.board` into `self.b` via `_CODE`; record `self.wtm`.
2. Sum `_STATIC[p][i]` over the 48 squares → `self._static` (the O(48) part, once
   per move instead of once per node).
3. Record king squares (`_wk`, `_bk`), non-king piece counts (`wnk`, `bnk`) and
   material (`wpow`, `bpow`).
4. Compute both king-pressure values from scratch (kings move rarely) —
   everything afterwards is incremental.
5. `_load_adj()` for the official tally.

This is the boundary "read the engine's truth, then work on my own fast copy": the
agent never mutates the engine, so it cannot corrupt the official game state.

## 8. Official-points tracker (272-355)

`_load_adj` (272-298) rebuilds the score the harness will use from
`engine.move_log`:

- captures: `_VAL[_CODE[move.piece_captured]]` credited to `i & 1` — ply 0 is
  White's move, so even indices are White.
- checks: the engine logs no check flags, so only the **last** move can be
  credited — if the side to move is in check right now, that move gave it. Every
  earlier move was credited the same way on the previous turn, and my own moves
  are banked live by `_adj_bank`, so the tally stays exact. An undo or a new game
  is detected with `len(log) < self._adj_ply` and resets the tally.
- then it sets `_adj_mode`: late in the game (`ADJ_CAP_PLY - n <= ADJ_LATE_PLY`)
  `+1` if we lead by `ADJ_LEAD_PTS`, `-1` if we are that far behind, else `0`.

Why it exists: if the game is adjudicated on points, a safe +2 check can be worth
more than a small positional plus. Be ready for the honest caveat — the
*aggressive* versions of this idea (a root tax on checking replies, pricing checks
inside the search, a bonus for exchanges while ahead) were implemented and
**measured net-negative** and removed. Only the late-game mode survives.

`_adj_points` (300-304) is a read accessor for the tally and is currently unused
(the live paths touch the arrays directly) — see §15.

`_gives_check` (306-320) applies a root move on `self.b`, tests whether the enemy
king square is attacked, and undoes it. It is root-only: 48 tests per move is
irrelevant, and it tells ordering which moves check.

`_root_hang` (322-341) returns the value of the piece a candidate move leaves both
attacked and undefended (kings excluded), used by the "protect a lead" ordering.

`_adj_bank` (343-355) banks *our chosen* move into the tally — its capture value
and +2 if it gives check — and advances `_adj_ply`, so the tally stays exact
without re-reading the log.

## 9. Incremental evaluation and king pressure (357-437)

`_pressure_on(ksq, by_white)` (357-370) sums `_WT[piece] * _PROX[ksq][sq]` over all
enemy pieces — i.e. "how much material is standing close to this king", weighted
by type and by distance (3/2/1 for adjacent/near/nearer). It is O(48) and is only
called when a king moves or once in `_load`.

`_press_add(code, sq, sign)` (372-383) applies the ±contribution of one piece to
the opposing king's pressure — the incremental counterpart.

`_eval_add` (385-411) / `_eval_sub` (413-437) are the symmetric make/unmake pair the
whole search is built on. For a move `fr → to` with `piece` and (possibly) a
`captured` piece:

- `_static += _STATIC[piece][to] - _STATIC[piece][fr]` — material *and*
  positional change in one lookup;
- if a piece was captured, subtract its static entry and decrement the material
  counts (and remove its pressure contribution);
- if a **king** moved, store the new king square and recompute that king's
  pressure from scratch (kings are rare, so this is cheap and exact);
- otherwise update both kings' pressure incrementally — but only when the
  `_PROX` lookup is non-zero, so pieces far from both kings cost nothing.

`_eval_sub` is the exact inverse, which is what makes search abort handling safe.
Every call site wraps make/unmake in `try/except _AbortSearch` so that an abort
restores the board, the side to move and the incremental eval before unwinding —
if that were asymmetric, the evaluation would silently drift for the rest of the
game.

**Why this matters for the viva**: a from-scratch evaluation scans 48 squares per
node; here it is a handful of list lookups. Together with the integer board, this
is the single biggest constant-factor win in the program, and it is also what lets
the same code be used in the quiescence search without a separate evaluator.

## 10. `get_best_move` (439-588) — the per-move driver

1. `t0 = time.monotonic()`, `_load()`, then `root = engine.get_legal_moves()`.
   If there are no legal moves, return `None` (the engine will report
   checkmate/stalemate). **The engine, not my code, defines legality** — the move
   I return is always one the engine itself produced.
2. A single legal move is returned immediately (still banked in the tally): no
   point burning clock on a forced move.
3. `_snapshot()` gives kings, `in_check`, `few` (≤9 pieces) and `drive` (opponent
   down to ≤1 non-king piece and we are ≥100 ahead) → `_plan_time` returns
   `(soft, hard)` deadlines. `test_budget` can override both for tests.
4. `_root_ck` is rebuilt: `_gives_check` for every root move — used by ordering and
   by `_adj_bank`.
5. **Iterative deepening** `d = 1 … cap`, where
   `cap = max_depth + (10 if drive else 8 if few else 0)`: the search is allowed to
   go deeper when a mate is available or in the endgame. The loop stops when the
   hard deadline is reached, or when the *projected* next iteration
   (`iter_secs * ITER_PREDICT`) would not fit inside the soft budget.
6. Each iteration re-orders the root: previous best move first, then `_root_key`
   (mate drive, adjudication policy, MVV-LVA).
7. **Aspiration windows**: iteration 1 uses the full window; later iterations
   search `prev_score ± 80`. If the result fails low or high the window is widened
   ×4 (up to ±MATE) and the *same depth* is re-searched.
8. Root moves are searched with PVS (`-negamax(d-1, -beta, -alpha, 1)`).
9. After each depth: `self.depth = d` (the harness reads this), and
   - if the best move **changed**, or the score dropped by ≥`PANIC_DROP`,
     `limit = hard_limit` — "panic time": an unstable root move means the position
     is critical, so it may spend up to the hard bound;
   - if the score exceeds `MATE_BOUND`, stop early (the mate is found);
   - `_last_drop` and `_last_stability` are updated for the next move's planning.
10. `_AbortSearch` from deeper in the tree is caught here; the `finally` block
    clears the deadlines and accumulates `time_used` with `time.monotonic()`.
11. If the search was aborted before scoring any move, fall back to `root[0]` —
    `get_best_move` never returns `None` while moves exist.
12. `_adj_bank(best, opk)` then return.

## 11. `_plan_time` (590-629) — the dynamic time control

```
remaining   = 60 - time_used
moves_left  = max(MIN_MOVES_LEFT, ASSUMED_MOVES - moves_done)
base        = remaining * 0.90 / moves_left        # an even share of the rest
mult        = 1.0 * (DRIVE|CHECK) * (FEW) * (WIDE) * (PANIC|EASY)
reserve     = min(RESERVE_SECONDS, remaining/2)
room        = max(0, remaining - RESERVE_SECONDS)
floor       = min(MIN_MOVE_SECONDS, remaining*0.1, room)   # then TINY_SLICE
spendable   = max(floor, min(remaining - reserve, room))
soft        = max(floor, min(base*mult, spendable, remaining*0.25))
hard        = max(soft, min(soft*2.5, spendable, remaining*0.35))
hard        = min(hard, remaining - MIN_RESERVE)   # absolute floor
```

Points to make in the viva:

- The share is recomputed **every move** from the time actually used, so a slow
  start is paid back automatically; it is not a fixed per-move budget.
- The multipliers encode chess sense: mating is worth 600 (`DRIVE_MULT=3`), being
  in check is critical (`CHECK_MULT=1.5`), the endgame needs precision
  (`FEW_MULT=1.6`), a wide choice is harder (`WIDE_MULT=1.1`), a score collapse
  means panic time, and a stable, unchanging best move means "move along"
  (`EASY_MULT=0.5`).
- `MIN_RESERVE = 4.5` is the point of the whole design: a time-out is not a small
  loss, it is an automatic zero (600-point swing). The reserve is never spendable
  — `hard` is clamped by it — and `TINY_SLICE` guarantees that even with ~1 s left
  a legal move gets produced (the last-resort allowance).
- `soft` vs `hard`: soft is a normal stopping point, hard is the emergency bound
  used only in panic time; the abort check fires against `hard`, so the normal
  case never even reaches it.


## 12. Move ordering (631-646, 813-855)

`_root_key` (631-646) orders root moves with the *strategy* baked in:

- mate drive: if a mate is finishable and this move gives check → `-200000`;
- chase mode (`_adj_mode < 0`): checks get `-20000`, because a check is +2 official
  points;
- protect mode (`_adj_mode > 0`): a quiet move that does **not** hang a piece
  (`_root_hang < ADJ_SAFE_PTS`) gets `+30000 + piece value` — with a lead, prefer
  solid moves;
- then normal MVV-LVA captures: `-100000 - _VAL[captured]`.

Inside the tree, `_order_key` (829-844) is the hot-path version: the caller
pre-resolves the lookups (`b`, TT move, both killers, `history`) and passes them in,
so the sort key is pure arithmetic. Priorities: TT move (`-10^9`), mate-drive
checks, MVV-LVA captures (`100000 + victim*100 - attacker`), killers (`-50000`),
then the history table.

`_ordering_key` (813-827) is the same idea for quiescence, where there is no TT or
killer context.

`_record_killer` (846-855) keeps two killer slots per depth (a killer is a quiet
move that caused a beta cutoff at the same depth in a sibling node).

Why ordering matters: alpha-beta's pruning depends entirely on searching the best
move first; with perfect ordering the tree is O(b^(d/2)) instead of O(b^d). This is
also why the TT stores a best move.

## 13. `_negamax` (663-811) — the search core, line by line

- Bind `self.b` and `self._negamax` to locals (Python attribute/global lookups
  dominate node cost); `nodes_expanded += 1`.
- **Abort check:** every 16 nodes (`nodes_expanded & _ABORT_MASK == 0`) compare
  `time.monotonic()` with the hard deadline and `raise _AbortSearch`. Reading the
  clock in every node would cost a measurable percentage of run time; every 16
  nodes keeps the overshoot far below the 5 s reserve.
- **TT probe:** `key = (wtm, bytes(self.b))` — `bytes` of the 48-integer board is a
  compact, hash-friendly, collision-free key. On a hit with sufficient depth the
  entry is used, and the stored value is **re-based by ply** before being returned
  as EXACT or used to tighten `alpha`/`beta`.
- **Mate-score normalisation:** the search returns `±(MATE - ply)` so a faster mate
  scores higher. A mate distance is only meaningful *relative to the node*, so the
  score is adjusted by `ply` when storing and when reading. Without this the engine
  stores "mate in 5" near the root and later compares it with "mate in 1" found
  deeper — the classic bug that causes missed short mates and spurious draws.
- **Depth 0 → quiescence**; **`ply > max_ply` → static eval** (safety valve).
- **Snapshot + check extension:** `_snapshot()` gives kings, `in_check`, `few`,
  `drive`. If in check, within `CHECK_EXT_PLY` of the root and the per-move budget
  has room, `depth += 1` — "never let a check run out of depth", but bounded,
  because an unbounded extension explodes on perpetual checks.
- **Reverse futility (static null move):** when not in check and `depth <= 3`, if
  the static score is already `120*depth` above `beta`, return it — at low depth a
  huge static margin is assumed to be real. The cheapest pruning here.
- **Null-move pruning:** `_null_move_allowed` (655-661) requires "not in check,
  depth ≥ 2, at least one non-king piece" (the piece test avoids zugzwang).
  Passing the turn and searching `depth-2` with a minimal window: if the opponent
  cannot beat `beta` even with a free move, this node is too good to be true.
  The side-to-move flag is restored inside the `except _AbortSearch` path too —
  that is exactly the line the comment-repair pass nearly deleted (see §17).
- **Terminal test:** legal moves; none → `-(MATE - ply)` if in check (mate) else `0`
  (stalemate). Mate *distance* enters here.
- **Move loop:** make → incremental eval → **PVS**: first move full window, later
  moves a null-window scout (`-alpha-1`) and, only on an improvement, a full-window
  re-search. **LMR**: late, quiet, non-checking moves while not in check at depth
  ≥ 3 are searched at `depth - _LMR[depth][index]` (never below 0); the
  scout-then-re-search pattern corrects a reduction that was too aggressive.
- **Killer/history update** on a beta cutoff: the move is recorded as a killer and
  the history table gets `depth*depth` with the classic
  `hval + bonus - hval*bonus//16384` decay, so old evidence fades instead of
  overflowing.
- **TT store:** EXACT if the score stayed inside the window, UPPER on a fail low,
  LOWER on a fail high; mate scores are re-based by `ply` on the way in.


## 14. Quiescence (857-1008)

`_capture_moves` (857-924) is a **capture-only** generator (quiescence never needs
quiet moves). It uses the same precomputed tables as `_legal_moves`, then applies
the legality shortcut: when not in check and the king square is known,
`_pinned_squares` (1095-1128) says which of our pieces stand between the king and
an enemy slider, so only king moves and pinned pieces need the expensive
`_attacked` test — most candidates are declared legal for free.

`_quiescence` (926-1008):

- abort check every 16 quiescence nodes (`self.qnodes`), as in `_negamax`;
- recompute `in_check` for the side to move;
- **in check:** generate *all* legal moves (real evasions, not just captures),
  return `-(MATE-1)` if there are none, cap recursion at `QMAX`, and search them
  all. Searching only captures here would let the program "see" mates that do not
  exist and would miss blocking moves — a classic source of blunders;
- **not in check:** stand-pat (`stand_pat >= beta` → cutoff), then captures sorted
  by `_ordering_key`, with **delta pruning** (`stand_pat + victim + 200 < alpha` →
  skip: even winning that piece cannot raise alpha) and the `QMAX` cap.

Why quiescence at all: without it, the static evaluation is applied in the middle
of a capture sequence, so the search hangs pieces one ply beyond the horizon.
`QMAX` was measured — raising 6 → 10 changed nothing, so 6 stayed.

## 15. Primitives (1010-1214)

`_snapshot` (1010-1044) is one 48-square scan returning `(my king, opp king,
in_check, few, drive)`. `drive = opp_nk <= 1 and pow_my - pow_opp >= 100`: the
opponent is down to a lone king and we are at least a rook up, i.e. a mating
attempt is worth the tournament's 600. It is recomputed per node because it also
drives the depth bonus and the mate ordering.

`_find_king` (1046-1052) is currently **unused** — king squares are cached in
`_wk`/`_bk` and updated by `_eval_add/_eval_sub`. It is a harmless helper; say so
plainly if asked rather than inventing a use.

`_attacked` (1054-1093) is the hot primitive: to test whether `sq` is attacked by
colour `by`, it checks the inverse pawn tables, then the knight and king offset
tables (single list lookups), then walks the bishop and rook rays from `sq` and
stops at the first occupied square. Reverse lookups like this are much faster than
generating the opponent's moves, and this primitive is reused by legality, check
detection, mate scoring and the danger terms.

`_pinned_squares` (1095-1128) walks every rook/bishop ray out of our king: the
first own piece on a ray is pinned if an enemy rook/bishop is behind it on the same
ray. It returns a 48-bit mask (`1 << square`), which is why legality can be decided
by two bit tests.

`_legal_moves` (1130-1214) is a staged generator: pseudo-legal moves first (single
pawn pushes and diagonal captures, knight/king offsets, sliding rays that stop at
blockers), then a legality filter. Optimisations: (a) when not in check and not in
drive mode, only king moves and pinned pieces need the full attack test; (b) in
drive mode it also computes a `gives` check flag per move. It returns
`(from, to, captured, gives)` tuples, a compact form the search consumes without
touching `Move` objects on every node.

Note the deliberate absence of castling, en-passant and promotion — the assignment
forbids them, so the generator is complete without special-case state.

## 16. Evaluation (1216-1398)

`_evaluate_stm` (1216) is the negamax convention: scores are always from the
side-to-move's point of view, so no search code ever flips a sign.

`_evaluate_white` (1220-1317) starts from the incremental `_static` (material plus
half-weight positional) and adds only the terms that cannot be maintained
incrementally:

1. **Passed pawns, endgame only** (≤12 non-king pieces): a pawn with no enemy pawn
   ahead of it on its file or an adjacent file scores `10 + 4*(rank advance)`.
   The variant has no promotion, so a runner is rewarded for approaching the back
   rank, where it creates mate threats (which is what actually pays).
2. **King placement:** with ≤6 pieces both kings score the config's
   `KING_PST_LATE_GAME` (centralise when few pieces); otherwise a small **pawn
   shield** term (+6 per friendly pawn directly in front of the king, mirrored).
3. **King pressure:** `_press_b - _press_w`, scaled by `PRESS_SCALE` and hard-capped
   at `PRESS_CAP = 30` (1.5 pawns) so it can never outbid material, applied only
   when the attacking side still has ≥2 non-king pieces.
4. **Mate drive** (the +600 objective): if we are ≥100 up and the opponent has ≤1
   non-king piece, add `_drive_bonus`; mirrored when we are behind.
5. **King danger** (the defensive half): when the opponent has ≥2 non-king pieces
   and is either ≥100 ahead or our king is near an edge with enough local pressure
   (`_edge_danger` gate), subtract `_king_danger` — full weight when material-down,
   half weight (`DANGER_EVEN = 0.5`) at level material. That half-weight choice was
   measured: it makes the eval aware of mate threats without distorting equal
   positions.

`_drive_bonus` (1319-1349): enemy king in a corner +140, on an edge +80, near-edge
+30; king proximity +50/+25/+10; and +40 per own rook that cuts the enemy king's
rank or file (the standard "rook cuts the king off" mating pattern).

`_edge_danger` (1351-1358) is a cheap O(1) gate: if the king is safely central,
return False immediately; otherwise require the pressure to reach
`DANGER_PRESS_MIN`. This keeps `_king_danger` (1360-1392) — which counts covered
flight squares (`escapes == 0` → +200, a mating net) and rook cuts — off the vast
majority of nodes.

`evaluate_board(game_state)` (1394-1398) is the public hook the tournament runner's
`AIPlayer` interface expects: ±MATE for checkmate (from the side to move), 0 for
stalemate, otherwise the static evaluation. It makes the class a drop-in
replacement for the template agent.


## 17. Rule compliance (worth stating up front in the viva)

- **One file, one class, roll-number naming**: `B23CS1001.py`, class `B23CS1001`,
  ~53 KB, imports only `config` (plus Python's `math` and `time`). No third-party
  packages, no data files.
- **No engine modification**: `board.py`, `config.py`, `AIPlayer.py` are untouched;
  the agent reads `engine.board`, `engine.white_to_move`, `engine.move_log` and
  calls `engine.get_legal_moves()` / `engine.is_in_check()`. It never calls
  `engine.make_move`, so my search cannot corrupt the official game state.
- **No special moves**: no castling, en-passant or promotion anywhere.
- **Not reinforcement learning**: a classical adversarial search (negamax over a
  game tree with alpha-beta pruning and a hand-written evaluation), which is
  exactly what the brief allows.
- **Time limit respected**: 4.5-5 s reserve, deadline checks every 16 nodes, and
  `test_budget.py` measures the clock margin under simulated CPU contention.
- **Legality guaranteed**: the returned move comes from the engine's own
  `get_legal_moves()`, and the internal generator is validated against the engine
  by the parity test.

## 18. Measured evidence to quote

| claim | evidence |
|---|---|
| search depth | depth 6 at 0.3 s and depth 8 at 1.0 s from the opening position |
| scale | ~15-20k nodes/s, ~1M nodes per game against a strong opponent |
| correctness | smoke passed; parity passed; incremental passed (160 checks over 4 games, incl. forced aborts, which is what proves make/unmake symmetry); endgame 5/5 mate conversions |
| clock safety | `test_budget.py --stress` (contended CPU) keeps a margin of several seconds |
| vs B23CS1082, both colours | checkmate in both games, official 600-0 twice; clocks used 50.2 s and 30.3 s of 60 s |
| trade-offs measured, not guessed | `_POS_SCALE = 0.5` cut hung material from ~6.7 to ~2.5 points per position; `QMAX` 6→10 was a no-op; full-strength check extension cost a ply and broke 5/5 endgame conversion; check-pricing, the root check tax and exchange bonuses were net-negative and were removed |


## 19. Likely viva questions and crisp answers

1. **Why negamax instead of separate min/max functions?** One code path and one
   score convention (always side-to-move). Min/max is the same thing with a sign
   flip; negamax halves the code and removes a class of sign bugs.
2. **What makes alpha-beta prune?** Good move ordering. Perfect ordering gives
   O(b^(d/2)); bad ordering degenerates to minimax. Hence TT move → captures
   (MVV-LVA) → killers → history.
3. **Why is the evaluation incremental?** A full scan is 48 squares per node; with
   `_STATIC` a move's delta is two lookups. It also lets quiescence reuse the same
   evaluator cheaply.
4. **Why `bytes(self.b)` as the TT key?** Compact (48 bytes), hashes fast, and
   cannot collide: one byte per square, values 0-10, plus the side to move.
5. **Why do mate scores need normalisation?** Mate distance is relative to the
   node, so a stored score must be re-based by ply; otherwise deep mates are
   compared with shallow ones and short mates get missed.
6. **Why is the null move safe here?** It is not a real move, only a pruning
   assumption: disabled in check, at depth < 2, and when only kings remain
   (zugzwang risk).
7. **What is LMR and why is it safe?** Late quiet moves are searched at reduced
   depth; if a reduced search still beats alpha the move is re-searched at full
   depth (PVS pattern), so a bad reduction cannot silently drop a good move.
8. **Why search evasions in quiescence?** Otherwise "mate" scores appear out of
   capture-only sequences and blocking moves are never found — the agent would see
   mates that do not exist.
9. **How do you know your own move generator is correct?** The parity test compares
   it with the engine's `get_legal_moves()`, and the incremental test re-derives
   the evaluation from scratch and compares it (160 checks).
10. **How do you avoid a time-out?** Untouchable 4.5-5 s reserve, soft/hard
    deadlines recomputed each move from the time actually used, a deadline check
    every 16 nodes, panic time when the root move becomes unstable, and a 0.02 s
    last-resort allowance so a legal move always exists.
11. **What happens if the search is aborted mid-node?** `_AbortSearch` unwinds, and
    every make has a matching unmake inside `try/except` — including the null-move
    and quiescence paths — so board, side to move and incremental eval all return
    to their pre-move values.
12. **Why the king-pressure term, and why capped?** It is the cheapest proxy for
    attacking chances in a variant with no queens, where mate threats dominate. The
    1.5-pawn cap guarantees it can never outbid material.
13. **Why half-weight positional terms?** Measured: full weight hung far more
    material at equal depth. The score card pays for material and mate, so
    positional terms are seasoning, not the meal.
14. **Complexity?** Per node: one 48-square snapshot, a few table lookups, and move
    generation proportional to the pseudo-legal move count; the TT makes repeated
    positions free. Total time is bounded by the deadline, not by depth.
15. **What is `drive` for?** A finishable mate is 600 points, the dominant term in
    the score card, so with a lone enemy king and a rook's worth of material the
    search gets extra depth and the eval gets a mating-drive bonus.
16. **What would you do next?** Static exchange evaluation for capture ordering,
    SEE-based quiescence, late-move pruning, and stronger mate defence (king danger
    beyond the current gate) — each behind the existing test gates, because
    run-to-run spread on these opponents is about ±70 points.

## 20. Honest weaknesses (say them before the examiner finds them)

- Two helpers are unused: `_adj_points` (a read accessor) and `_find_king` (king
  squares are cached). Neither affects behaviour.
- The positional weight is deliberately low, so the agent is not a positional
  player; it wins by material, checks and mate threats.
- Mate defence is a heuristic (pressure + edge gate + danger term): a measured
  trade-off, not a solved problem.
- The search is capped at `max_depth = 12`; in the fastest positions the clock is
  the binding constraint, not the cap.
- Outcomes against these opponents are dominated by ±600 mate events, so one game
  proves little: decisions are taken over pairs of games and repeated runs.

## 21. Change log of this file

1. **Tier A (strategy prune)**: `ADJ_CHECK_PTS` check pricing, the `ROOT_CHECK_W`
   root check tax with `_threat_checks`/`_root_threat`, the `ADJ_TRADE_W` exchange
   bonus and the zero-weight `_PROX_FAR` path were removed — each had been measured
   net-negative or was dead code. `improvement.md` has the numbers.
2. **Tier B (size prune)**: comments/docstrings compacted from 73,876 B / 1,726
   lines to ~53 KB / ~1,400 lines to respect the 60 KB single-file limit; the
   removed narrative is archived in `improvement.md` and the code AST was proven
   identical.
3. **Comment repair**: the compaction left ~24 dangling comment fragments, which
   were replaced with accurate one-liners under a guard script that re-checks AST
   equality before writing. That guard caught a real off-by-one that had deleted
   `self.wtm = not self.wtm` in the null-move abort path — a useful story for a
   viva about why even "comments only" changes get tested.

