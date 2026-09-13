# `B23CS1001.py` — complete line-by-line explanation

This document explains the **final, shipped** version of the agent in
`B23CS1001.py` (1075 lines). Every line/block is covered in order, with the
*reason* it exists, not just what it does.

---

## 0. What the file is

A single-file chess agent for the **Spartans variant: 6 columns x 8 rows = 48
squares, no queens** (`BW=6`, `BH=8`). Each side starts with **6 pawns, 1 knight,
2 bishops, 2 rooks and a king** (verified against `board.py`), and the piece values
are `P=20, N=70, B=70, R=100` (`config.PIECE_VALUES`) — the rook is the strongest
piece on the board, and there is no promotion rule.

It is picked up by the tournament runner as the class named after the roll
number, `B23CS1001`, constructed as `B23CS1001(engine, max_depth=...)`, and asked
for moves through `get_best_move()` (and, for the runner's own score display,
through `evaluate_board(game_state)`).

**Scoring model this agent optimises for** (from the official rules):

| Event | Points |
|---|---|
| Each check delivered | **+2** |
| Material captured | piece value (P20 N70 B70 R100) |
| Delivering **checkmate** | **600** (overrides everything else) |

So the objective is *material + mates*; the +2 checks are a rounding error
next to a 600-point mate. That single fact drives most design decisions below
(and it is why an experiment that paid the search +2 per check was measured to
lose ~960 points over 6 paired games and was reverted — see `improvement.md`).

### Architecture at a glance

| Layer | Lines | Idea |
|---|---|---|
| Precomputed tables | 16–148 | everything computable at import time *is* computed at import time |
| Integer board + incremental eval | 37–249 | the engine's 2-char strings are converted once to small ints; eval is O(1) |
| Root driver / time management | 252–377 | iterative deepening + aspiration windows under a per-move budget |
| Alpha-beta search | 388–568 | PVS + TT + null-move + LMR + reverse-futility |
| Quiescence | 570–713 | captures + evasions, MVV-LVA, delta pruning |
| Board queries | 716–845 | attack detection and pin masks from precomputed geometry |
| Move generation | 847–938 | generate-then-validate with a per-piece pin shortcut |
| Evaluation | 940–1063 | material + PST + passed pawns + king safety + mate drive |

Speed achieved: **~31k–60k NPS** (was ~3.5k before the integer board rollout),
reaching **depth 7–8 in 0.35 s**, which is the main source of playing strength.

---

## 1. Preamble and module constants (lines 1–35)

| Line | Code | Explanation |
|---|---|---|
| 1–4 | `"""Single-file adversarial agent for Spartans Chess (6x8, no queens)."""` | Module docstring. |
| 5–6 | `import math` / `import time` | `math.log` is used **only at import time** to build the LMR table (143–148); `time.monotonic()` drives the search deadline. Neither is called in the hot loop. |
| 8–12 | `from config import (BOARD_WIDTH as BW, BOARD_HEIGHT as BH, EMPTY_SQUARE as ES, PIECE_VALUES, KING_PST_LATE_GAME, PAWN_PST, KNIGHT_PST, BISHOP_PST, ROOK_PST)` | Tournament-provided constants, aliased short because `BW`/`BH` appear in the hottest code. `BH=8`, `BW=6`. |
| 13 | `from board import Move` | Imported for type familiarity; **never actually used** (the agent hands back the engine's own `Move` objects). Harmless. |
| 16 | `MATE = 100000` | Score of being mated/checkmating. Must exceed any possible material total, so mate always outranks material. |
| 17 | `FEW_PIECE_LIMIT = 9` | At/below 9 total pieces the root search extends its depth (see line 279), because that is where mates and passed pawns are decided. |
| 18 | `KING_PST = KING_PST_LATE_GAME` | Late-game king table used when few pieces remain (king should centralise then). |
| 19 | `PST = {'P': PAWN_PST, ..., 'R': ROOK_PST}` | Dict of piece-letter → piece-square table, consumed only by the `_STATIC` build loop (56). |
| 21 | `# Move-ordering value of each piece type (capture "victim" weight).` | Comment. |
| 22 | `PT_VAL = {'P': 20, 'N': 70, 'B': 70, 'R': 100, 'K': 0}` | Victim weights used by MVV-LVA ordering. King capture is 0 because it can never happen (moves leaving the king en prise are filtered out). |
| 24–26 | `TT_EXACT = 0`, `TT_LOWER = 1`, `TT_UPPER = 2` | Transposition-table bound flags: the node's score is exact / a lower bound (beta cutoff) / an upper bound (failed low). |
| 28–35 | `_PIECE_CODES = {...}` | An early string-to-int map, **superseded by `_CODE` (45) and now unused**. Kept as documentation of the encoding. |

## 2. The integer board and the precomputed tables (lines 37–148)

### 2.1 Piece encoding (lines 37–50)

The engine stores pieces as 2-character strings (`'wP'`, `'--'`, ...). String
compare/slice in a search that visits hundreds of thousands of nodes is pure
overhead, so the agent converts once (`_load`, 180) into a flat `list[int]` of 48
entries: `index = row*BW + col`.

| Line | Code | Explanation |
|---|---|---|
| 37–41 | comment | Explains the conversion and why: "integer compares/indexing are markedly faster than string slicing, which is where the search spends most of its time." |
| 42 | `E = 0` | Empty square. **`0` is falsy**, which every inner loop exploits (`if tp:` = "occupied"). |
| 43 | `WP, WN, WB, WR, WK = 1, 2, 3, 4, 5` | White pieces. |
| 44 | `BP, BN, BB, BR, BK = 6, 7, 8, 9, 10` | Black pieces. So **`p < 6` ⇔ white**, the most frequent test in the file. |
| 45–46 | `_CODE = {'--':0, 'wP':1, ..., 'bK':10}` | The one-time string→int translation used by `_load`. |
| 47 | `_TYPE = (0,1,2,3,4,5, 1,2,3,4,5)` | Colour-independent type index: 1=P 2=N 3=B 4=R 5=K. Lets one `if` chain handle both colours. |
| 48 | `_VAL = (0,20,70,70,100,600, 20,70,70,100,600)` | Material value by code, **white positive**. King = 600 only so "king captured" arithmetic can never look cheap. |
| 49 | `_VAL_SIGN = (0,20,70,70,100,600, -20,-70,-70,-100,-600)` | Same values **signed by colour**, used to seed `_STATIC` so one addition per piece yields a white-positive score. |
| 50 | `_PTVAL = (0,20,70,70,100,0)` | Value by *type* index, king = 0, for MVV-LVA ordering. |

### 2.2 `_STATIC`: the folded static-score table (lines 52–75)

The single most important evaluation optimisation: material + piece-square table
+ centralisation are pre-added per `(piece code, square)`, so the hot loop does
**one list index** instead of three lookups and an add.

| Line | Code | Explanation |
|---|---|---|
| 52–54 | comment | "folded into a single table lookup, so the hot evaluation loop does one indexing per occupied square." |
| 55 | `_STATIC = [None] * 11` | One table per piece code 0..10 (index 0 unused). |
| 56 | `_PST_OF = {1: PAWN_PST, 2: KNIGHT_PST, 3: BISHOP_PST, 4: ROOK_PST}` | Type index → PST. King intentionally absent: king safety is dynamic (993–1020). |
| 57 | `for _code in range(1, 11):` | Build a table for every real piece. |
| 58–61 | `_t = _TYPE[_code]`, `_tab = _PST_OF.get(_t)`, `_white = _code < 6`, `_base = _VAL_SIGN[_code]` | Type, PST (or `None` for the king), colour flag, signed base value. |
| 62 | `_arr = [0] * 48` | Row-major accumulator for this piece code. |
| 63–64 | `for _r in range(BH):` / `_mr = _r if _white else BH - 1 - _r` | Black reads the vertically mirrored PST row, so tables only need to be authored for white. |
| 65–68 | `for _c in range(BW):` … `_v += _tab[_mr][_c] if _white else -_tab[_mr][_c]` | White adds the PST value, black subtracts it (score is white-positive). |
| 69–73 | `if 2 <= _t <= 4:` … `_v += (3 - _d) if _white else -(3 - _d)` | Small **centralisation** bonus for N/B/R. `_d` is the Manhattan distance from the board centre (row 3, col 2 on an 8x6 board); within a 3-square diamond it earns `3-_d`. |
| 74–75 | `_arr[_r * BW + _c] = _v` / `_STATIC[_code] = tuple(_arr)` | Stored as a **tuple**: immutable, slightly faster to index, cannot be corrupted by bugs. |

Because the same table drives the initial scan (`_load`) and every incremental
update (`_eval_add`/`_eval_sub`), making and unmaking a move costs **O(1)**
evaluation work.

### 2.3 Transposition entry (lines 78–85)

| Line | Code | Explanation |
|---|---|---|
| 78–79 | `class TTEntry:` / `__slots__ = ('depth','flag','value','best_move')` | `__slots__` avoids a per-entry `__dict__`; with tens of thousands of live entries that is real memory and allocation-time savings. |
| 81–85 | `def __init__(self, depth, flag, value, best_move)` | `depth` = depth the entry was stored at, `flag` = EXACT/LOWER/UPPER, `value` = score (mate-normalised, 517–521), `best_move` = `(from, to)` reused for ordering. |

### 2.4 Directions and precomputed geometry (lines 88–148)

| Line | Code | Explanation |
|---|---|---|
| 88–89 | `_KNIGHT_DR`, `_KNIGHT_DC` | The 8 knight row/col offsets. |
| 90–91 | `_KING_DR`, `_KING_DC` | The 8 king/adjacency offsets. |
| 92–93 | `_BISHOP_DR`, `_BISHOP_DC` | 4 diagonal directions. |
| 94–95 | `_ROOK_DR`, `_ROOK_DC` | 4 orthogonal directions. |
| 97–99 | comment | Goal: "the hot paths never do bounds checks or function calls. Built once at import time (48 squares)." |
| 100–106 | `_KNIGHT_ATT = []` … `_PAWN_CAP_B = []` | Seven per-square lists. |
| 107–108 | `for _r in range(BH):` / `for _c in range(BW):` | One import-time pass over all 48 squares. |
| 109–112 | `_KNIGHT_ATT.append(tuple(... if 0 <= _r+dr < BH and 0 <= _c+dc < BW))` | **Absolute indices** a knight can reach from this square; off-board targets are filtered out here so `_attacked` never bounds-checks. |
| 113–116 | `_KING_ATT.append(...)` | Same for the 8 king neighbours. |
| 117–121 | `_BISHOP_RAYS.append(tuple(tuple(...) for dr, dc in ...))` | **Ray-per-direction**: `_BISHOP_RAYS[sq]` is 4 tuples, each listing the squares outward along one diagonal. Slider code just walks a tuple until it hits a piece — no arithmetic, no bounds tests, cache-friendly. |
| 122–126 | `_ROOK_RAYS.append(...)` | Same for the 4 orthogonal rays. |
| 127 | `_PAWN_FWD.append((_r - 1) * BW + _c if _r - 1 >= 0 else -1)` | The single forward square of a white pawn (white moves toward row 0), or `-1` on the last rank. |
| 128–130 | `_PAWN_CAP_W.append(...)` | White pawn capture targets (row-1, col±1), boundary-filtered. |
| 131–133 | `_PAWN_CAP_B.append(...)` | Black pawn capture targets (row+1, col±1). |
| 135 | `MATE_BOUND = MATE - 1000` | Anything beyond `±MATE_BOUND` is a mate score. Used to convert mate-distance scores in/out of the TT (404–407, 518–521). |
| 136 | `QMAX = 6` | Quiescence depth cap, so a long capture chain cannot recurse without bound. |
| 138–141 | comment + `_PAWN_CAP_W_inv = _PAWN_CAP_B` / `_PAWN_CAP_B_inv = _PAWN_CAP_W` | **Reverse pawn attacks**: "which squares could a white pawn attack *from* to hit square X?" is exactly the mirrored geometry of a *black* pawn's capture targets. Reusing the two existing tables avoids building more. |
| 143–145 | comment + `_LMR = [[0] * 64 for _ in range(64)]` | **Late-move-reduction table** `R(depth, move_index)`, precomputed so the search never calls `math.log`. |
| 146–148 | `for _d in range(3, 64):` … `_LMR[_d][_i] = max(0, 1 + int(math.log(_d) * math.log(_i + 1) / 2.0))` | The standard logarithmic reduction curve. Entries stay 0 for `depth<3` and for `move_index<3`: no reduction early in the list or at shallow depth, which is what made LMR safe in this variant (an aggressive variant was tested and lost). |
| 152–154 | `class _AbortSearch(Exception)` | Control-flow signal used to unwind the whole recursion instantly when the clock runs out; a normal return would have had to be checked at every node. |
| 155–157 | `def _idx(r, c): return r * BW + c` | Row/col → flat index. Superseded by the precomputed tables in hot paths; still used where a move's coordinates arrive as raw row/col (root loop, `_root_key`). |

---

## 3. Agent state, board loading and incremental evaluation (lines 160–249)

### 3.1 `__init__` (lines 160–177)

| Line | Code | Explanation |
|---|---|---|
| 160–161 | `class B23CS1001:` + docstring | The class name **must** be the roll number: that is how the tournament runner discovers the agent. |
| 163 | `def __init__(self, engine, depth=6, max_depth=12)` | Runner passes the engine; `max_depth` caps iterative deepening. |
| 164 | `self.engine = engine` | Kept for reading the position, the move log and the side to move. |
| 165 | `self.nodes_expanded = 0` | Node counter; doubles as a cheap clock-check trigger (393) and as the NPS statistic. |
| 166–167 | `self.depth = depth` / `self.max_depth = max_depth` | `depth` is overwritten with the *completed* iteration depth (350) so callers can see how deep it actually got. |
| 168 | `self.time_used = 0.0` | Cumulative seconds spent on our moves; drives the shrinking per-move budget (261). |
| 169 | `self.test_budget = None` | Test/benchmark override: when set, the agent uses exactly this many seconds per move (272). `None` in tournament play. |
| 170–171 | `self.max_ply = 60` / `self._deadline = None` | Hard recursion cap (425) and the wall-clock deadline for the current move. |
| 172–173 | `self.b = None` / `self.wtm = True` | The integer board and the side to move; created/set by `_load`. |
| 174 | `self.tt = {}` | Transposition table: `(wtm, bytes(board))` → `TTEntry` (247, 397). A plain dict was fast enough and avoids a hash-collision-replacement policy. |
| 175 | `self.killers = {}` | depth → up to two quiet "killer" moves (`(from,to)` packed as `fr*48+to`). |
| 176 | `self.history = [0] * (48 * 48)` | History heuristic, also keyed by `fr*48+to`, so one flat list indexes it with no hashing. |
| 177 | `self.qnodes = 0` | Quiescence node counter (separate from `nodes_expanded` for profiling). |

### 3.2 `_load` — engine board → integer board + eval state (lines 180–213)

| Line | Code | Explanation |
|---|---|---|
| 180–185 | `b = []` … `self.b = b` | Flatten the engine's 8 rows of 6 strings into 48 ints using `_CODE`. Done once per `get_best_move`, not per node. |
| 186 | `self.wtm = self.engine.white_to_move` | Mirror the side to move. |
| 187–191 | `static = 0`; `wk = bk = -1`; `wnk = bnk = 0`; `wpow = bpow = 0` | Accumulators for the incremental evaluation state: total static score, both king squares, non-king piece counts and total pieces' value per side. |
| 192–195 | `for i in range(48): p = b[i]; if p == E: continue` | One scan over the 48 squares. |
| 196 | `static += _STATIC[p][i]` | The whole material+PST+centre contribution of this piece, one lookup (section 2.2). |
| 197–200 | `if p == WK: wk = i elif p == BK: bk = i` | Track king squares **incrementally**, which is why `_attacked` and quiescence never scan for kings. |
| 201–206 | `elif p < 6: wnk += 1; wpow += _VAL[p] else: bnk += 1; bpow += _VAL[p]` | Count and value the non-king pieces of each side. `wpow/bpow` power the mate-drive detection (754, 1022) and `wnk/bnk` the "few pieces" triggers. |
| 207–213 | `self._static = static` … `self.bpow = bpow` | Publish the state. From here on it is maintained by `_eval_add`/`_eval_sub`, so `_evaluate_white` is O(1). |
| 171 note | — | `_wk`/`_bk` are also re-synced defensively at the end of `_evaluate_white` (1028–1029) — see the note in section 9. |

### 3.3 `_eval_add` / `_eval_sub` — O(1) make/unmake evaluation (lines 215–245)

These two run on **every** move made and unmade anywhere in the search (root
loop, `_negamax`, `_quiescence`). They are what makes it affordable to call
`_evaluate_stm()` freely.

| Line | Code | Explanation |
|---|---|---|
| 215–216 | `def _eval_add(self, piece, fr, to, captured)` + docstring | "Incremental evaluation update when `piece` moves fr → to." |
| 217 | `self._static += _STATIC[piece][to] - _STATIC[piece][fr]` | The mover's contribution changes from its old square to its new square: two lookups, one add. |
| 218–221 | `if piece == WK: self._wk = to elif piece == BK: self._bk = to` | King squares follow the king automatically, so nothing ever has to scan for a king. |
| 222 | `if captured:` | `0` is falsy, so one test distinguishes captures from quiet moves. |
| 223 | `self._static -= _STATIC[captured][to]` | Drop the captured piece's contribution. |
| 224–229 | `if captured < 6: self.wnk -= 1; self.wpow -= _VAL[captured] else: self.bnk -= 1; self.bpow -= _VAL[captured]` | Decrement the captured side's piece count and total power. |
| 231–245 | `_eval_sub` | **Exact inverse** of `_eval_add`: subtract the mover's delta, restore the king square to `fr`, add back the captured piece and its counters. Keeping the two visibly symmetric is deliberate — any asymmetry would silently corrupt the evaluation. |

### 3.4 `_position_key` (lines 247–249)

| Line | Code | Explanation |
|---|---|---|
| 247–248 | comment | "Each square is a 0..10 code, so `bytes()` is a very fast exact key." |
| 249 | `return (self.wtm, bytes(self.b))` | The TT key. `bytes(list[int])` is a C-level conversion; since every code is < 256 there are **no hash collisions**, and including `wtm` prevents mixing the two perspectives. |

---

## 4. Root driver: time management and iterative deepening (lines 252–377)

### 4.1 `get_best_move` (lines 252–356)

| Line | Code | Explanation |
|---|---|---|
| 252–254 | `def get_best_move(self): engine = self.engine; self._load()` | Public entry point; re-syncs the integer board and eval state from the engine before every move. |
| 255–257 | `root = engine.get_legal_moves()` / `if not root: return None` | Root moves come from the **engine's own** generator. That guarantees an illegal move can never be played even if the fast internal rules had a bug — insurance paid once per move. |
| 259 | `t0 = time.monotonic()` | Start of this move's clock (`monotonic` so system-clock changes cannot break it). |
| 260 | `moves_done = (len(engine.move_log) + 1) // 2` | Full moves already played, used to shrink the budget as the game progresses. |
| 261 | `budget = (60.0 - self.time_used) * 0.45 / max(1, 75 - moves_done)` | **Budget formula**: spend 45 % of the time still left, spread over the moves assumed to remain (a 75-move game). Prevents the classic "burn all 60 s by move 20 and lose on time" failure. |
| 262 | `budget = max(0.25, min(budget, 2.5))` | Clamp: never below 0.25 s (too weak), never above 2.5 s (too greedy). |
| 264–266 | comment | Documents mate-drive: leading by ≥ a rook while the opponent has ≤1 real piece left. |
| 267 | `myk, opk, in_check, few, drive = self._snapshot()` | One board scan yields king squares, check status, "few pieces" and mate-drive. |
| 268–271 | `if drive: budget = max(budget, min(2.0, (60.0 - self.time_used) * 0.55 / max(1, 75 - moves_done)))` | When a mate is finishable, allow 55 % instead of 45 % — banking a +600 mate is worth far more than saving 0.05 s. |
| 272–273 | `if self.test_budget is not None: budget = self.test_budget` | Deterministic override for tests/benchmarks (identical position ⇒ identical move). |
| 274–275 | `deadline = t0 + budget` / `self._deadline = deadline` | Absolute wall-clock deadline, checked at node ~393 and at the top of each iteration below. |
| 277–279 | `best = None`; `prev_score = None`; `cap = self.max_depth + (10 if drive else 8 if few else 0)` | `cap` = how deep to *attempt*: normal `max_depth`, +8 when material is thin, +10 when driving for mate (those positions have few pieces, so extra depth is affordable). |
| 280–283 | `try:` / `for d in range(1, cap + 1):` / `if time.monotonic() >= deadline: break` | **Iterative deepening** 1,2,3,… Each finished iteration also seeds the next one's ordering. An aborted attempt is simply discarded, never played. |
| 284–288 | `if best is not None: bt = (best.start_row, best.start_col, best.end_row, best.end_col)` / `else: bt = None` | The previous best move as a plain 4-tuple, for fast comparison in the sort key. |
| 289–296 | `ordered = sorted(root, key=lambda m: (0 if bt is not None and (m.start_row, m.start_col, m.end_row, m.end_col) == bt else 1, self._root_key(m, drive, myk, opk)))` | Root move ordering: **the previous iteration's best move first** (the biggest single ordering win at the root), then `_root_key` (checks while driving, then captures). Python sorts tuples element-wise, so the `0/1` flag dominates. |
| 297–300 | comment + `half = None if prev_score is None else 80` | **Aspiration window** setup: full width for the first iteration, then only ±80 around the previous *search* score. The comment preserves the bug that was fixed: centring on the static eval made every root move fail low and silently kept a stale move. |
| 301–306 | `while True:` … `alpha = max(-MATE * 2, prev_score - half)` / `beta = min(MATE * 2, prev_score + half)` | Open the window, or full width when `half is None`. |
| 307–309 | `a0 = alpha` / `cur = None` / `cur_score = -MATE * 2` | Remember the window's lower edge (`a0`) to detect a fail-low, and reset the per-attempt best. |

| Line | Code | Explanation |
|---|---|---|
| 310–318 | `for mv in ordered:` … `self._eval_add(piece, fr, to, captured)` | Try each root move: decode `row/col → flat index`, make the move on the integer board, flip `wtm`, update the incremental eval. Exactly the same make/unmake protocol as inside the search. |
| 319–326 | `try: score = -self._negamax(d - 1, -beta, -alpha, 1)` / `except _AbortSearch:` … `raise` | Search the child with the **negated** window at `ply=1`. If the clock expires mid-move, unmake everything first (board integrity!) and re-raise so the last *completed* depth is kept. |
| 327–330 | `self.b[fr] = piece` … `self._eval_sub(piece, fr, to, captured)` | Unmake: restore board, side to move and eval state. |
| 331–332 | `if score > cur_score: cur_score = score` | Track the best score seen in this window attempt (used for the fail-low/high decision). |
| 333–335 | `if score > alpha: alpha = score; cur = mv` | Track the best move **within the window**. Note that a fail-low leaves `cur` as `None`, so `best` keeps its old value: an unverified move is never promoted. |
| 336–337 | `if alpha >= beta: break` | Standard beta cutoff at the root. |
| 338–339 | `if cur is not None: best = cur` | Commit this attempt's verdict. |
| 340–342 | `if half is None: prev_score = cur_score; break` | A full-width search gives an exact score ⇒ the iteration is complete. |
| 343–347 | `if cur_score <= a0 or cur_score >= beta:` / `half = None if half >= MATE else min(MATE, half * 4)` / `prev_score = cur_score` / `continue` | **Fail low or fail high ⇒ quadruple the window and re-search the same depth.** This is the fix for the aspiration bug noted at 297 (the old code kept a stale best move here). |
| 348–350 | `prev_score = cur_score` / `break` / `self.depth = d` | Score inside the window ⇒ exact; record the score and the reached depth, then deepen. |
| 351–352 | `except _AbortSearch: pass` | Out of time: fall through, keeping whatever the last **completed** iteration produced. |
| 353–355 | `finally: self._deadline = None` / `self.time_used += time.monotonic() - t0` | Always clear the deadline and bill the elapsed time, even when aborting. |
| 356 | `return best if best is not None else root[0]` | Defensive fallback: if even depth 1 could not finish, return the first legal move instead of `None`. |

### 4.2 `_root_key` (lines 358–377)

| Line | Code | Explanation |
|---|---|---|
| 358–360 | `def _root_key(self, mv, drive, myk, opk):` + docstring + `key = 0` | Root-only ordering helper; returns a key where **more negative = tried earlier**. |
| 361–371 | `if drive and opk >= 0:` … make the move … `if nk >= 0 and self._attacked(nk, my): key -= 200000` … unmake | In mate-drive mode, moves that **give check** (the king is attacked after the move, or the king square itself moved out of attack) get the biggest bonus. This is legitimate check-preference used *only* for ordering, not for scoring — the experiment that scored checks eventually lost points, but ordering toward forcing moves near a mate is free. |
| 374–376 | `cap = self.b[_idx(...)]` / `if cap != E: key -= 100000 + _VAL[cap]` | Then prefer captures, weighted by the victim's value (MVV-LVA style at the root). |
| 377 | `return key` | Smaller key ⇒ sorted earlier by `sorted` at 289. |

---

## 5. Search core (lines 380–568)

### 5.1 `_null_move_allowed` (lines 380–386)

| Line | Code | Explanation |
|---|---|---|
| 380–382 | `def _null_move_allowed(self, in_check, depth):` / `if in_check or depth < 2: return False` | Never try a null move while in check (there is a threat to answer) or below depth 2 (nothing to save). |
| 383–386 | `for p in self.b:` / `if p != E and _TYPE[p] != 5: return True` / `return False` | Require at least one **non-king** piece. In a bare-king endgame a null move can be zugzwang-riddled and prove false cutoffs, so it is disabled there. |

### 5.2 `_negamax` (lines 388–523)

The whole search is negamax with alpha-beta: a node returns the score for the
side to move, so the parent negates it.

| Line | Code | Explanation |
|---|---|---|
| 388–391 | `def _negamax(self, depth, alpha, beta, ply):` / `b = self.b` / `ng = self._negamax` / `self.nodes_expanded += 1` | Bind `self.b` and the method itself to locals — attribute lookups dominate Python search cost. `nodes_expanded` doubles as the clock-check counter. |
| 393–395 | `if (self.nodes_expanded & 255) == 0 and self._deadline is not None and time.monotonic() >= self._deadline: raise _AbortSearch()` | **Clock check every 256 nodes**: `& 255` is a much cheaper test than the time call, and 256 nodes is fine-grained enough. |
| 397–399 | `key = self._position_key()` / `entry = self.tt.get(key)` / `tt_move = None if entry is None else entry.best_move` | Probe the transposition table and remember the stored best move for ordering. |
| 401–407 | `if entry is not None and entry.depth >= depth:` / `value = entry.value` / `if value > MATE_BOUND: value -= ply elif value < -MATE_BOUND: value += ply` | Only trust an entry searched at least as deep. **Mate scores are stored relative to the node, not the root** (see 517–521), so they are converted back by adding/subtracting the current `ply`. Without this, a mate found at a deep ply would look like a nearer mate when reused higher up — a real bug in the earlier TT. |
| 408–419 | `if entry.flag == TT_EXACT: return value` / `if entry.flag == TT_LOWER: if value >= beta: return value; if value > alpha: alpha = value` / `elif entry.flag == TT_UPPER: if value <= alpha: return value; if value < beta: beta = value` | Standard bound usage: EXACT cuts immediately; LOWER/UPPER cut when they prove the window, otherwise they narrow `alpha`/`beta`. |
| 421–424 | comment + `if depth <= 0: return self._quiescence(alpha, beta)` | **Horizon**: hand over to quiescence. Deliberately placed *before* the board scan, because quiescence recomputes what it needs (the comment records this ordering as a win). |
| 425–426 | `if ply > self.max_ply: return self._evaluate_stm()` | Safety net against a non-terminating line (checks/captures could otherwise chain past the intended horizon). |
| 428 | `myk, opk, in_check, few, drive = self._snapshot()` | One scan gives everything the rest of the node needs: own/enemy king squares, check status, and whether this is a mate-drive node (which makes `_legal_moves` also compute `gives_check` for ordering). |
| 430–435 | `stand_pat = None` / `if not in_check:` / `if depth <= 3:` / `stand_pat = self._evaluate_stm()` / `if stand_pat - 120 * depth >= beta: return stand_pat` | **Reverse futility pruning (static null move)**: if even giving away 120 per remaining ply still beats beta, the opponent will not let us down to this node's real value, so return the static score. Only at depth ≤ 3, where the margin is safe. |
| 436–441 | `if self._null_move_allowed(in_check, depth):` / `self.wtm = not self.wtm` / `score = -ng(depth - 2, -beta, -beta + 1, ply + 1)` / `self.wtm = not self.wtm` / `if score >= beta: return beta` | **Null-move pruning**: pass the move (flip `wtm` *without touching the board*, which is why no eval update is needed) and search two plies shallower with a null window. If the opponent still beats beta while we do nothing, this node fails high. |
| 443–445 | `legal = self._legal_moves(myk, drive, opk, in_check)` / `if not legal:` / `return -(MATE - ply) if in_check else 0` | Generate legal moves. **No moves ⇒ mate (`-(MATE - ply)`, preferring later mates) or stalemate (0).** `-ply` encodes mate distance so the search prefers quicker mates and resists longer ones. |

| Line | Code | Explanation |
|---|---|---|
| 447–456 | comment + `killers = self.killers.get(depth, ())` / `k0 = killers[0] if killers else -1` / `k1 = killers[1] if len(killers) > 1 else -1` / `if tt_move is not None: tt_fr, tt_to = tt_move else: tt_fr = tt_to = -1` / `hist = self.history` | **Hoists every ordering input out of the per-move key function.** The comment records the win: the killer lookup used to run once *per move* inside the sort lambda. |
| 457–461 | `ordered = sorted(legal, key=lambda m: self._order_key(m, drive, b, tt_fr, tt_to, k0, k1, hist))` | Sort moves by the hot key (5.3). |
| 463–465 | `best_move = None` / `alpha_orig = alpha` / `best_score = -MATE` | `alpha_orig` is needed to classify the node for the TT (511–516); `best_score` starts below any real score. |
| 466–471 | `for idx, (fr, to, captured, gives) in enumerate(ordered):` … make the move, flip `wtm`, `_eval_add` | Try moves in order. `idx` is the *move index*, which drives LMR. |
| 472–474 | `try:` / `if idx == 0: score = -ng(depth - 1, -beta, -alpha, ply + 1)` | The first (best-ordered) move gets a **full-window** search — this is the PVS idea. |
| 475–481 | `else: search_depth = depth - 1` / `if depth >= 3 and captured == E and not gives and not in_check:` / `search_depth -= _LMR[depth if depth < 64 else 63][idx]` / `if search_depth < 0: search_depth = 0` | **Late-move reduction**: for later, quiet, non-checking moves at depth ≥ 3, search shallower using the precomputed `_LMR` table (clamped to the table size and to ≥ 0). Forcing moves (captures, checks, evasions) are never reduced — that is the safety rule that makes LMR correct here. |
| 482 | `score = -ng(search_depth, -alpha - 1, -alpha, ply + 1)` | **Null-window (zero-width) scout search** at the possibly reduced depth. |
| 483–485 | `if score > alpha and score < beta:` / `score = -ng(depth - 1, -beta, -alpha, ply + 1)` | If the scout beats alpha (and is inside the window) the reduction/null window may have been too optimistic ⇒ **re-search at full depth and full window**. |
| 486–491 | `except _AbortSearch:` / unmake … / `raise` | Clock abort: unmake the move to keep the board consistent, then propagate. |
| 492–495 | `b[fr] = piece` / `b[to] = captured` / `self.wtm = not self.wtm` / `self._eval_sub(piece, fr, to, captured)` | Unmake after a normal search. |
| 496–502 | `if score > best_score: best_score = score; best_move = (fr, to)` / `if score > alpha: alpha = score` / `if alpha >= beta: break` | Track the best move (needed for the TT and for the killer heuristic), raise alpha, and cut off on beta. |
| 504–510 | `if best_move is not None:` / `bmove = best_move[0] * 48 + best_move[1]` / `self._record_killer(depth, best_move)` / `hbonus = depth * depth` / `hval = self.history[bmove]` / `self.history[bmove] = hval + hbonus - hval * hbonus // 16384` | Update the heuristics: record the cutoff move as a killer for this depth, and bump history with the classic **bounded "gravity" update** so a very high early score cannot permanently dominate ordering. |
| 511–516 | `if best_score <= alpha_orig: flag = TT_UPPER elif best_score >= beta: flag = TT_LOWER else: flag = TT_EXACT` | Classify the node: failed low (all moves ≤ original alpha), cutoff (lower bound), or exact. |
| 517–521 | `store_val = best_score` / `if best_score > MATE_BOUND: store_val = best_score + ply elif best_score < -MATE_BOUND: store_val = best_score - ply` | **Mate-score normalisation**: store mates *relative to the current node*, so the same position reached at different plies can share one TT entry. (The values are converted back on probe, 404–407.) This was a genuine bug fix: without it, TT hits returned wrong mate distances. |
| 522 | `self.tt[key] = TTEntry(depth, flag, store_val, best_move)` | Store (simple always-replace — adequate at this size and it keeps the code small). |
| 523 | `return best_score` | Return the node value for the parent to negate. |

### 5.3 Move ordering keys (lines 525–568)

| Line | Code | Explanation |
|---|---|---|
| 525–528 | `def _ordering_key(self, move, drive, depth, tt_move=None):` / `fr, to, captured, gives = move` / `ht = fr * 48 + to` / `key = 0` | Generic ordering key (used by quiescence, where there is no per-node hoisting). More negative = tried earlier. |
| 529–530 | `if tt_move is not None and (fr, to) == tt_move: key -= 10 ** 9` | TT move first, by a huge margin. |
| 531–532 | `if drive and gives: key -= 300000` | While driving for mate, checks come before captures (they are how mates actually happen). |
| 533–535 | `if captured: key -= 100000 + _PTVAL[_TYPE[captured]] * 100 - _PTVAL[_TYPE[self.b[fr]]]` | **MVV-LVA**: prefer capturing a valuable victim (×100) with a cheap attacker. |
| 536–537 | `if ht in self.killers.get(depth, ()): key -= 50000` | Killer moves (quiet moves that caused cutoffs at this depth) jump ahead of ordinary quiets. |
| 538 | `key -= self.history[ht]` | Then history scores break remaining ties. |
| 541–557 | `_order_key(self, move, drive, b, tt_fr, tt_to, k0, k1, hist)` | **The hot version used inside `_negamax`.** Identical logic, but the TT move, both killer values and the history list arrive as arguments, so the loop body does no dict lookups, no `self.` lookups and no `in` checks. This function is called once per move per node, so it matters. |
| 559–568 | `_record_killer(self, depth, move)` | Pack the move to `fr*48+to`; ignore duplicates; `insert(0, ...)` and truncate to 2 so `killers[depth]` is a tiny two-element list (cheap to test against). |

---

## 6. Quiescence (lines 570–713)

### 6.1 `_capture_moves` — capture-only generator (lines 570–640)

| Line | Code | Explanation |
|---|---|---|
| 570–572 | `def _capture_moves(self, ksq, opp, in_check=False):` + docstring | "Legal capture moves only (used by quiescence; avoids generating quiet moves and keeps the delta-pruning friendly 4-tuple shape)." |
| 573–576 | `b = self.b` / `my_white = self.wtm` / `found = []` / `ap = found.append` | Locals only; `found` holds `(from, to, captured)` **before** legality filtering. |
| 577–583 | `for i in range(48): p = b[i]` / skip empties / skip own-piece squares / `pt = _TYPE[p]` | The single 48-square scan. `(p < 6) != my_white` is the colour test. |
| 584–589 | pawn: `caps = _PAWN_CAP_W[i] if my_white else _PAWN_CAP_B[i]` / for each target, `if tp and (tp < 6) != my_white: ap((i, t, tp))` | Pawn **captures only** (no forward pushes — quiescence is about forcing moves). |
| 590–594 | knight: `for t in _KNIGHT_ATT[i]: ... if tp and enemy: ap(...)` | Knight captures. |
| 595–599 | king: `for t in _KING_ATT[i]: ...` | King captures (the king's legality is checked later). |
| 600–607 | bishop: `for ray in _BISHOP_RAYS[i]: for t in ray: tp = b[t]; if tp: if enemy: ap(...); break` | Walk each ray, stop at the **first** piece; capture it if it is an enemy. |
| 608–615 | rook: same with `_ROOK_RAYS` | Rook captures. |
| 616–617 | `if not found: return []` | Fast exit — most quiet positions have no captures at all, and this avoids the expensive pin computation below. **This early return is a large part of why quiescence got cheap.** |
| 618–626 | `legal = []` / `if in_check or ksq < 0: pinned = 0; test_all = True` / `else: pinned = self._pinned_squares(ksq, opp); test_all = False` | Same pin shortcut as `_legal_moves`: only king moves and pinned pieces can expose the king, so in the common case most captures skip the attack test. In check, everything must be tested. |
| 627–630 | `for fr, to, captured in found:` / `if not (test_all or fr == ksq or (pinned >> fr) & 1): legal.append((fr, to, captured, False)); continue` | Accept the move without any `_attacked` call. The 4th element (`gives`) is `False` because quiescence does not need check information. |
| 631–639 | make the move on the board, `king = to if fr == ksq else ksq`, `ok = not (king >= 0 and self._attacked(king, opp))`, unmake, append if `ok` | The expensive path: actually play the move and ask "is my king attacked now?". `king >= 0` guards the (impossible) missing-king case. |

### 6.2 `_quiescence` (lines 642–713)

| Line | Code | Explanation |
|---|---|---|
| 642–650 | `def _quiescence(self, alpha, beta, qdepth=0):` + docstring | Docstring records the design: capture-only generation instead of "generate every legal move then filter" (which cost ~30 attack tests per node), MVV-LVA ordering, delta pruning, and **full evasions while in check** so horizon mates are still seen. |
| 651–655 | `self.qnodes += 1` / locals `b`, `my_white`, `my`, `opp` | Counted separately so profiling can tell search from quiescence work. |
| 656–659 | comment + `stand_pat = self._evaluate_stm()` / `myk = self._wk if my_white else self._bk` / `in_check = myk >= 0 and self._attacked(myk, opp)` | Stand-pat score, and the king square is read **incrementally** (no scan) to determine check. |
| 661–664 | `if in_check:` / `legal = self._legal_moves(myk, False, -1, True)` / `if not legal: return -(MATE - 1)` | **In check there is no stand-pat** (you must respond), so all legal evasions are generated; zero evasions ⇒ mate at the horizon. `-(MATE - 1)` keeps it just inside mate range. |
| 665–666 | `if qdepth >= QMAX: return self._evaluate_stm()` | Depth cap prevents an endless evasion chain. |
| 667–682 | `legal.sort(key=lambda m: self._ordering_key(m, False, 0))` / loop: make, flip `wtm`, `_eval_add`, `score = -self._quiescence(-beta, -alpha, qdepth + 1)`, unmake, `alpha = score`, `if alpha >= beta: return alpha` | Search every evasion, with a beta cutoff. Because evasions are searched exhaustively (not just captures), simple mates are still detected at the horizon. |
| 685–690 | `if stand_pat >= beta: return stand_pat` / `if stand_pat > alpha: alpha = stand_pat` / `if qdepth >= QMAX: return alpha` | The classic quiescence stand-pat test: the side to move can *choose* not to capture. `return alpha` (not `stand_pat`) when the depth cap is hit. |
| 692–695 | `captures = self._capture_moves(myk, opp)` / `if not captures: return alpha` / `captures.sort(key=lambda m: self._ordering_key(m, False, 0))` | Only forcing moves continue the exchange; MVV-LVA ordering maximises cutoffs. |
| 696–698 | `for fr, to, captured, gives in captures:` / `if stand_pat + _VAL[captured] + 200 < alpha: continue` | **Delta pruning**: if even winning the captured piece (plus a 200 safety margin) cannot reach alpha, skip the move entirely — this is where most of quiescence's savings come from. |
| 699–708 | make the move, flip, `_eval_add`, search, unmake, `_eval_sub` | Standard make/search/unmake around the recursive call. |
| 709–712 | `if score > alpha: alpha = score` / `if alpha >= beta: return alpha` | Fail-soft update and cutoff. |
| 713 | `return alpha` | Score for the side to move. |

---

## 7. Board queries (lines 716–845)

### 7.1 `_snapshot` (lines 716–755)

| Line | Code | Explanation |
|---|---|---|
| 716–722 | `def _snapshot(self):` + docstring | Docstring states the contract: `(my_king, opp_king, in_check, few_pieces, mate_drive)`, with mate-drive defined as "up by ≥ a rook against an opponent with at most one non-king piece". |
| 723–729 | `my_white = self.wtm` / `opp = 'b' if my_white else 'w'` / `b = self.b` / `myk = opk = -1` / `tot = 0` / `pow_my = pow_opp = 0` / `opp_nk = 0` | Initialise the accumulators once; `-1` is the "not found" sentinel for king squares. |
| 730–734 | `for i in range(48):` / `p = b[i]` / `if p == E: continue` / `tot += 1` | One scan; count occupied squares for the "few pieces" test. |
| 735–746 | `if p == WK: if my_white: myk = i else: opk = i; continue` / same for `BK` reversed | Assign the kings to *my* or *opponent* depending on the side to move — this "relativity" is what lets the whole search be colour-agnostic. |
| 747–752 | `v = _VAL[p]` / `if (p < 6) == my_white: pow_my += v else: pow_opp += v; opp_nk += 1` | Sum material for both sides and count the opponent's non-king pieces (the mate-drive condition needs just that count). |
| 753 | `in_check = myk >= 0 and self._attacked(myk, opp)` | Check status, using the incremental king square. |
| 754 | `drive = opp_nk <= 1 and pow_my - pow_opp >= 100` | Mate drive ⇒ opponent has ≤1 real piece and we lead by ≥ a rook. |
| 755 | `return myk, opk, in_check, tot <= FEW_PIECE_LIMIT, drive` | Return all five; `few` (≤ 9 pieces) is what triggers the extra depth at the root (279). |

### 7.2 `_find_king` (lines 758–764) — *unused*

| Line | Code | Explanation |
|---|---|---|
| 758–764 | `def _find_king(self, color)` | A linear 48-square king search. **Now dead code**: king squares are tracked incrementally (`_wk`/`_bk`). Left in place; harmless. |

### 7.3 `_attacked` (lines 766–809)

| Line | Code | Explanation |
|---|---|---|
| 766–771 | `def _attacked(self, sq, by):` + docstring | "Uses the precomputed geometry tables (no bounds tests, no `_idx` calls), which is the single hottest helper in the search." |
| 773–780 | `if by == 'w': pn = WN; pk = WK; pb = WB; prk = WR` / `for t in _PAWN_CAP_W_inv[sq]: if b[t] == WP: return True` | For "attacked by white", the *reverse* pawn table gives the squares a white pawn would attack *from*. One tuple walk. |
| 781–788 | `else:` the equivalent for black | Same idea for black. |
| 789–791 | `for t in _KNIGHT_ATT[sq]: if b[t] == pn: return True` | Knight attacks: ≤ 8 lookups, no bounds checks. |
| 792–794 | `for t in _KING_ATT[sq]: if b[t] == pk: return True` | King adjacency — also what makes "is this square safe for my king" one lookup. |
| 795–801 | `for ray in _BISHOP_RAYS[sq]: for t in ray: p = b[t]; if p: if p == pb: return True; break` | Bishop/queen-less diagonal attacks: walk outward, stop at the first piece, succeed only if it is an enemy bishop. |
| 802–808 | same with `_ROOK_RAYS[sq]` and `prk` | Rook attacks. |
| 809 | `return False` | No attacker found. |

### 7.4 `_pinned_squares` (lines 811–845)

The key legality optimisation: a **quiet** move can only expose its own king if
the mover was pinned, so computing the pinned set once per node lets every other
move skip the attack test entirely.

| Line | Code | Explanation |
|---|---|---|
| 811–813 | `def _pinned_squares(self, ksq, opp):` + docstring | "Bitmask of own pieces standing between the king and an enemy slider." |
| 814–818 | `b = self.b` / `pinned = 0` / `opp_white = opp == 'w'` / `bishop = WB if opp_white else BB` / `rook = WR if opp_white else BR` | Setup; the result is a Python int used as a 48-bit mask. |
| 819–831 | `for ray in _ROOK_RAYS[ksq]:` / `first = -1` / walk the ray: if a piece is found and `first < 0`, then either an **enemy** piece blocks the ray first (⇒ `break`, no pin) or it is the candidate blocker (`first = t`); once the blocker is known, an enemy **rook** further along means the blocker is pinned, so its bit is added to the `pinned` mask and the ray stops | "Exactly one friendly piece between my king and an enemy rook on this ray ⇒ it is pinned." Breaking early whenever the first piece is an enemy is what keeps this cheap. |
| 832–844 | same loop over `_BISHOP_RAYS[ksq]` with `bishop` | Diagonals. |
| 845 | `return pinned` | The bitmask consumed by `_legal_moves` (921) and `_capture_moves` (628). |

---

## 8. Move generation: `_legal_moves` (lines 847–938)

This is the most performance-critical function in the file (it runs at every
search node). It is **generate-then-validate**, exactly mirroring `board.py`, so
the agent's move set can never diverge from the engine's rules.

| Line | Code | Explanation |
|---|---|---|
| 847–854 | `def _legal_moves(self, ksq, drive=False, opk=-1, in_check=None):` + docstring | `ksq` = own king square, `opk` = enemy king square, `drive` = also compute `gives_check` (needed only for mate-drive ordering), `in_check` = precomputed by the caller when it already knows (saves an `_attacked` call). |
| 855–858 | `b = self.b` / `my_white = self.wtm` / `my = 'w' if my_white else 'b'` / `opp = 'b' if my_white else 'w'` | Local bindings; `my`/`opp` are the strings `_attacked` expects. |
| 859–860 | `pseudo = []` / `ap = pseudo.append` | Pseudo-legal `(from, to)` candidates. Validation is split into a second pass so the cheap "obviously legal" decision can be made per move. |
| 861–867 | `for i in range(48): p = b[i]` / skip empty / skip opponent pieces / `ptype = _TYPE[p]` | The single 48-square scan. |
| 868–872 | pawn: `f = _PAWN_FWD[i] if my_white else (i + BW if i + BW < 48 else -1)` / `if f >= 0 and b[f] == E: ap((i, f))` | One-square pawn push. **Pawns never move two squares in this variant** (matching the engine), and there is no promotion rule to handle. |
| 873–876 | `for t in (_PAWN_CAP_W[i] if my_white else _PAWN_CAP_B[i]): tp = b[t]; if tp and (tp < 6) != my_white: ap((i, t))` | Pawn captures, using the precomputed target tuples. |
| 877–881 | knight: `for t in _KNIGHT_ATT[i]: tp = b[t]; if tp == E or (tp < 6) != my_white: ap((i, t))` | Knight moves to empty or enemy squares. One condition handles both because `0 == E` is falsy and an enemy satisfies `(tp<6) != my_white`. |
| 882–891 | bishop: for each ray, for each square: `if tp == E: ap(...)` else `if enemy: ap(...); break` | Sliders walk until blocked; the blocking square is itself a legal capture target. |
| 892–901 | rook: same with `_ROOK_RAYS` | Rook moves. |
| 902–906 | king: `for t in _KING_ATT[i]: ...` | King moves (legality of the destination is checked in the validation pass). |
| 908–912 | `legal = []` / `if ksq < 0: for fr, to in pseudo: legal.append((fr, to, b[to], False))` / `return legal` | Degenerate case: no king tracked ⇒ nothing can be illegal, so accept everything with `captured` read straight off the board and `gives=False`. |
| 913–919 | comment + `if in_check is None: in_check = self._attacked(ksq, opp)` / `must_test = in_check or (drive and opk >= 0)` / `pinned = 0 if must_test else self._pinned_squares(ksq, opp)` | **The legality shortcut.** Only king moves and pinned pieces can expose the king, so: while in check (or while computing `gives` for mate drive, which needs every move played anyway) test everything; otherwise compute the pinned bitmask once per node. |
| 920–923 | `for fr, to in pseudo:` / `if not (must_test or fr == ksq or (pinned >> fr) & 1): legal.append((fr, to, b[to], False)); continue` | **The fast path**: moves of a non-king, non-pinned piece are provably legal, so they are emitted with **no `_attacked` call at all**. In a typical midgame node this covers ~90 % of the moves, and this is the single biggest generator-level win. |
| 924–933 | make the move on the board; `king = to if fr == ksq else ksq`; `ok = not (king >= 0 and self._attacked(king, opp))`; `gives = False`; `if drive and opk >= 0: nk = to if to == opk else opk; gives = nk >= 0 and self._attacked(nk, my)` | **The slow path**: play the move, test whether our king is attacked (works whether the moving piece was the king or a pinned piece), and — only when driving — whether the *enemy* king is attacked (again handling "the enemy king itself was captured"). |
| 934–937 | `b[fr] = piece` / `b[to] = captured` / `if ok: legal.append((fr, to, captured, gives))` | Unmake and emit. The returned 4-tuple `(from, to, captured, gives)` is exactly the shape the ordering key and the search loop destructure. |
| 938 | `return legal` | Fully legal move list. |

> **Note (measured, not kept):** a single-pass rewrite of this function (classify
> during generation, no intermediate list) was implemented and measured at
> **15.752 µs/call vs 15.720 µs/call** — identical. The second pass was never the
> bottleneck, so the simpler original was kept (see `improvement.md`, Round 9).

---

## 9. Evaluation (lines 940–1063)

### 9.1 `_evaluate_stm` (lines 940–943)

| Line | Code | Explanation |
|---|---|---|
| 940–941 | `# --- evaluation` + `def _evaluate_stm(self):` | "Side to move" wrapper. |
| 942–943 | `v = self._evaluate_white()` / `return v if self.wtm else -v` | The core evaluation is written **white-positive** (matching `_STATIC`); negate it when black is to move so the search can always assume "bigger = better for me". |

### 9.2 `_evaluate_white` (lines 945–1030)

| Line | Code | Explanation |
|---|---|---|
| 945–953 | `def _evaluate_white(self):` + docstring | Docstring states the composition and the key claim: "The material/PST/centre part and the king squares are maintained incrementally … only the endgame passed-pawn scan touches the board." |
| 954 | `b = self.b` | Local binding (used only by the passed-pawn scan and king shield). |
| 955–957 | comment + `score = self._static` | The evaluation **starts** as the precomputed material+PST+centre total — zero work, it is already maintained by `_eval_add`/`_eval_sub`. This is what makes the midgame eval O(1). |
| 958–963 | `w_king = self._wk` / `b_king = self._bk` / `w_nk = self.wnk` / `b_nk = self.bnk` / `w_pow = self.wpow` / `b_pow = self.bpow` | Pull all incremental state into locals: king squares, non-king piece counts and material power per side. |
| 964 | `nk = w_nk + b_nk` | Total non-king pieces, used twice: endgame detection here and the king-PST switch at 998. |
| 965 | `if nk <= 12:` | Enter the endgame-only **passed-pawn scan**. The guard is deliberate: in the midgame the scan is skipped entirely (the comment at 966–967 says exactly this), which keeps node cost low for the 90 % of nodes that are not endgames. |
| 968–972 | `for i in range(48): p = b[i]` / `if p == WP:` / `r = i // BW; c = i - r * BW` | Scan for white pawns; derive rank and file. |
| 973–978 | `for rr in range(r - 1, -1, -1): base = rr * BW` / `if b[base + c] == BP or (c > 0 and b[base + c - 1] == BP) or (c + 1 < BW and b[base + c + 1] == BP): break` | Walk forward toward the promotion rank (row 0 for white) checking the three squares a black pawn could occupy to stop it. `for … else` (979) is the Python idiom for "no blocker found" ⇒ the pawn is passed. |
| 979–980 | `else: score += 10 + (BH - 1 - r) * 4` | **Passed pawn bonus**: a flat 10 plus 4 per rank advanced (relative to the pawn's own home rank, so a pawn about to promote is worth much more). |
| 981–991 | `elif p == BP:` … mirrored scan toward row 7 … `else: score -= 10 + r * 4` | The identical logic for black pawns, subtracted (white-positive score). |
| 993–997 | `if w_king >= 0 and b_king >= 0:` / `wr = w_king // BW; wc = w_king - wr * BW` / `br = …; bc = …` | King safety only makes sense with both kings on the board (`w_king`/`b_king` are `-1` if not found, e.g. inside a partial search state). Decode both kings' coordinates. |
| 998–1000 | `if nk <= 6:` / `score += KING_PST[wr][wc]` / `score -= KING_PST[BH - 1 - br][bc]` | **Late-game king table**: with ≤ 6 non-king pieces the king should be active/centralised, so the late-game PST applies (black mirrored via `BH - 1 - br`). |

| Line | Code | Explanation |
|---|---|---|
| 1001–1002 | `else:` / `# King shield (inlined): friendly pawns in front of each king.` | Midgame king safety, written out inline rather than as a helper call for speed. |
| 1003–1011 | `rr = wr - 1` / `if rr >= 0:` / `base = rr * BW` / `if wc > 0 and b[base + wc - 1] == WP: score += 6` / `if b[base + wc] == WP: score += 6` / `if wc + 1 < BW and b[base + wc + 1] == WP: score += 6` | White king shield: +6 for each friendly pawn on the three squares directly in front of the king (the rank the king would step to). Castling-like structure is favoured without any castling rules in this variant. |
| 1012–1020 | `rr = br + 1` / `if rr < BH:` / `base = rr * BW` / the three black-pawn checks, each `score -= 6` | The mirror image for black (black's forward direction is row+1). |
| 1022–1026 | `if w_pow - b_pow >= 100 and b_nk <= 1 and w_king >= 0 and b_king >= 0: score += self._drive_bonus('w', w_king, b_king)` / `elif b_pow - w_pow >= 100 and w_nk <= 1 and …: score -= self._drive_bonus('b', b_king, w_king)` | **Mate-drive evaluation**: exactly when one side leads by ≥ 100 (a rook) against an opponent with ≤ 1 non-king piece, add the drive bonus (9.3). This is the term that converts "materially winning" into "actually mate within the horizon", and a mate is 600 points. |
| 1027–1030 | comment + `self._wk = w_king` / `self._bk = b_king` / `return score` | Re-sync the incremental king squares (they are locals here, and writing them back makes the function safe to call even if `_load`/`_eval_*` were bypassed) and return the white-positive score. |

### 9.3 `_drive_bonus` — the mate-conversion term (lines 1032–1063)

| Line | Code | Explanation |
|---|---|---|
| 1032–1034 | `def _drive_bonus(self, leader, lk, ek):` + docstring | "Positive score (for `leader`) pushing the enemy king to the edge while the leader's king and rook(s) move in for the mate." |
| 1035–1038 | `b = self.b` / `er, ec = divmod(ek, BW)` / `lr, lc = divmod(lk, BW)` / `score = 0` | Decode both kings. `divmod` is used here because this runs rarely (only in drive positions). |
| 1039–1046 | `rowd = min(er, BH - 1 - er)` / `cold = min(ec, BW - 1 - ec)` / `if rowd == 0 and cold == 0: score += 140` / `elif rowd == 0 or cold == 0: score += 80` / `elif rowd <= 1 or cold <= 1: score += 30` | **Drive the enemy king to the edge/corner**: `rowd`/`cold` are distances to the nearest edge, so 0 means it is already on that edge. Corner is best (140), then any edge (80), then one step from an edge (30). This is the standard "KR vs K" mating heuristic. |
| 1047–1053 | `dist = max(abs(lr - er), abs(lc - ec))` / `if dist <= 2: score += 50` / `elif dist <= 3: score += 25` / `elif dist <= 4: score += 10` | Bring the leader's king close (Chebyshev distance), which is a prerequisite for the mate. |
| 1054–1062 | comment + `cuts = 0` / `for i in range(48): if b[i] == (WR if leader == 'w' else BR): rr = i // BW; rc = i - rr * BW; if rr == er or rc == ec: cuts += 1` / `score += 40 * cuts` | **Cut-off rooks**: a rook on the enemy king's rank or file traps it (the classic "box" method). +40 per cutting rook. There are no queens in this variant, so rooks are the only pieces that can do this. |
| 1063 | `return score` | All positive for the leader; the caller adds or subtracts it depending on which side leads (1022–1026). |

---

## 10. Public evaluation hook (lines 1065–1072)

| Line | Code | Explanation |
|---|---|---|
| 1065–1066 | `def evaluate_board(self, game_state="ongoing"):` + docstring | "Public evaluation hook used by the tournament runner" — the runner may ask the agent to score a position; it must not disturb search state. |
| 1067–1068 | `if game_state == "checkmate": return MATE if self.engine.white_to_move else -MATE` | A finished mate returns the largest possible magnitude, signed for the side to move (the mated side is to move in a checkmate position). |
| 1069–1070 | `if game_state == "stalemate": return 0` | Stalemate is a draw in this variant ⇒ 0. |
| 1071–1072 | `self._load()` / `return self._evaluate_white()` | Otherwise re-sync from the engine and return the static white-positive score. `_load` is what makes this safe to call between searches. |
| 1073–1075 | blank lines | End of file. |

---

## 11. Notes, invariants and gotchas

### 11.1 Invariants the code relies on

1. **`0` means empty and is falsy** — every "is this square occupied / did this move
   capture anything" test is a truthiness test. Never assign a non-zero value for
   "empty".
2. **`p < 6` ⇔ white.** All colour logic is one comparison; `_VAL`/`_STATIC` are
   white-positive and `_evaluate_stm` negates for black.
3. **`_eval_add` and `_eval_sub` must stay exact inverses**, and every make **must**
   be paired with an unmake — including on the abort path (321–326, 486–491, and
   the root loop). An unbalanced make/unmake silently corrupts `_static`,
   `_wk`/`_bk`, `wpow`/`bpow` and therefore all later scores.
4. **`wtm` flips exactly once per make** (and once back per unmake). The null-move
   search (437–439) is the one deliberate exception: it flips `wtm` *without*
   touching the board, so no eval update is needed either.
5. **Mate scores are relative to the node inside the TT** and converted to
   root-relative on use (404–407 / 518–521). Anything out of `±MATE_BOUND` is a
   mate.
6. **`_wk`/`_bk` are maintained incrementally**; `-1` means "not tracked".

### 11.2 Dead but harmless code

| Lines | What | Note |
|---|---|---|
| 9, 10 | `EMPTY_SQUARE as ES`, `PIECE_VALUES` | Imported, never used (the file has its own `E` and `_VAL`). |
| 13 | `from board import Move` | Imported, never used. |
| 28–35 | `_PIECE_CODES` | Superseded by `_CODE` (45). |
| 758–764 | `_find_king` | Superseded by the incremental `_wk`/`_bk`. |

Removing them changes nothing functionally — they are left in place to keep the
diff vs the validated, tournament-tested file byte-identical.

### 11.3 Things that were tried and **not** kept (all measured)

| Idea | Measured result | Where documented |
|---|---|---|
| Opening book | statistically neutral (paired seeded test) | `improvement.md` |
| Late-move pruning (LMP), counter-move heuristic, unbounded check extension | each failed the paired/gate test | `improvement.md` |
| **+2 per check in the search** (root and `ply ≤ 2`, with a precomputed candidate-square pre-filter) | **−960 points** over 6 paired games; it lost material and stopped finding mates | `improvement.md` Round 8 |
| Single-pass `_legal_moves` | **identical** speed (15.75 vs 15.72 µs); reverted | `improvement.md` Round 9 |
| Aggressive LMR / other pruning tweaks | reverted (regressions) | `improvement.md` |

The pattern is consistent: **material and mate are the dominant terms** (a check is
2 points, a pawn 20, a rook 100, a mate 600), so only changes aimed at deeper,
correct search of material/mate survived.

### 11.4 How to run/validate

```bash
python3 -m py_compile B23CS1001.py   # syntax
python3 smoke_test.py                # end-to-end sanity
python3 run_parity.py                # 3195 positions: agent move set vs engine, 0 mismatches
python3 test_endgame.py              # mate-conversion test (timing-dependent, 4-5/5)
python3 run_games_fast.py            # head-to-head vs the black-box agent (official score)
```

`run_parity.py` is the key correctness gate: it asserts that the fast internal
rules agree with `board.py` on every position it visits. **Any change to
`_legal_moves`, `_attacked` or `_pinned_squares` must keep it at 0 mismatches.**

Because the benchmark opponent is **stochastic** and a single mate swings the
score by ±600, only *paired, seeded, multi-game* comparisons are meaningful —
single game results are noise.

