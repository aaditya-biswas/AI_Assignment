""""
Single-file adversarial agent for Spartans Chess (6x8, no queens).

"""
import math
import time

from config import (
    BOARD_WIDTH as BW, BOARD_HEIGHT as BH, EMPTY_SQUARE as ES,
    PIECE_VALUES, KING_PST_LATE_GAME,
    PAWN_PST, KNIGHT_PST, BISHOP_PST, ROOK_PST,
)
from board import Move


MATE = 100000            # scores for mates exceed any material total
FEW_PIECE_LIMIT = 9      # at/below this many pieces: cheap terminal check
KING_PST = KING_PST_LATE_GAME
PST = {'P': PAWN_PST, 'N': KNIGHT_PST, 'B': BISHOP_PST, 'R': ROOK_PST}

# Move-ordering value of each piece type (capture "victim" weight).
PT_VAL = {'P': 20, 'N': 70, 'B': 70, 'R': 100, 'K': 0}

TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2

_PIECE_CODES = {
    '--': 0,
    'wP': 1, 'bP': 2,
    'wN': 3, 'bN': 4,
    'wB': 5, 'bB': 6,
    'wR': 7, 'bR': 8,
    'wK': 9, 'bK': 10,
}

# ------------------------------------------------------------------ int board
# Internally the agent works on a flat list of small ints instead of the
# engine's 2-char strings (0 empty, 1..5 = white P/N/B/R/K, 6..10 = black).
# Integer compares/indexing are markedly faster than string slicing, which is
# where the search spends most of its time.
E = 0
WP, WN, WB, WR, WK = 1, 2, 3, 4, 5
BP, BN, BB, BR, BK = 6, 7, 8, 9, 10
_CODE = {'--': 0, 'wP': 1, 'wN': 2, 'wB': 3, 'wR': 4, 'wK': 5,
         'bP': 6, 'bN': 7, 'bB': 8, 'bR': 9, 'bK': 10}
_TYPE = (0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5)   # code -> 1P 2N 3B 4R 5K
_VAL = (0, 20, 70, 70, 100, 600, 20, 70, 70, 100, 600)
_VAL_SIGN = (0, 20, 70, 70, 100, 600, -20, -70, -70, -100, -600)
_PTVAL = (0, 20, 70, 70, 100, 0)            # by type index (king -> 0)

# ------------------------------------------------------- king pressure (checks)
# "Check avoidance" evaluation: enemy pieces hanging around our king are what
# produce the endless stream of +2 checks (and eventually mates), so each king
# carries a running measure of how much enemy material is near it.  This is the
# classic "king tropism" idea (Chess Programming Wiki), made free here by the
# incremental make/unmake: the value is maintained in _eval_add/_eval_sub and so
# costs nothing per evaluation.
_WT = (0, 1, 2, 2, 3, 0, 1, 2, 2, 3, 0)     # attacker weight by code (K -> 0)
_PROX_RANGE = 3                              # only pieces this close matter
_PROX = []
for _k in range(48):
    _kr, _kc = divmod(_k, BW)
    _row = []
    for _s in range(48):
        _sr, _sc = divmod(_s, BW)
        _d = max(abs(_sr - _kr), abs(_sc - _kc))
        _row.append(_PROX_RANGE + 1 - _d if 0 < _d <= _PROX_RANGE else 0)
    _PROX.append(tuple(_row))

# Fully precomputed static score per (piece code, square): signed material +
# piece-square table + centralisation folded into a single table lookup, so
# the hot evaluation loop does one indexing per occupied square.
_STATIC = [None] * 11
_PST_OF = {1: PAWN_PST, 2: KNIGHT_PST, 3: BISHOP_PST, 4: ROOK_PST}
for _code in range(1, 11):
    _t = _TYPE[_code]
    _tab = _PST_OF.get(_t)
    _white = _code < 6
    _base = _VAL_SIGN[_code]
    _arr = [0] * 48
    for _r in range(BH):
        _mr = _r if _white else BH - 1 - _r
        for _c in range(BW):
            _v = _base
            if _tab is not None:
                _v += _tab[_mr][_c] if _white else -_tab[_mr][_c]
            if 2 <= _t <= 4:
                _d = (_r - 3 if _r > 3 else 3 - _r) + \
                     (_c - 2 if _c > 2 else 2 - _c)
                if _d < 3:
                    _v += (3 - _d) if _white else -(3 - _d)
            _arr[_r * BW + _c] = _v
    _STATIC[_code] = tuple(_arr)


class TTEntry:
    __slots__ = ('depth', 'flag', 'value', 'best_move')

    def __init__(self, depth, flag, value, best_move):
        self.depth = depth
        self.flag = flag
        self.value = value
        self.best_move = best_move


_KNIGHT_DR = (2, 2, -2, -2, 1, 1, -1, -1)
_KNIGHT_DC = (1, -1, 1, -1, 2, -2, 2, -2)
_KING_DR = (1, -1, 0, 0, 1, 1, -1, -1)
_KING_DC = (0, 0, 1, -1, 1, -1, 1, -1)
_BISHOP_DR = (1, 1, -1, -1)
_BISHOP_DC = (1, -1, 1, -1)
_ROOK_DR = (1, -1, 0, 0)
_ROOK_DC = (0, 0, 1, -1)

# ---------------------------------------------------------------- fast tables
# Precomputed per-square attack geometry so the hot paths never do bounds
# checks or function calls.  Built once at import time (48 squares).
_KNIGHT_ATT = []
_KING_ATT = []
_BISHOP_RAYS = []
_ROOK_RAYS = []
_PAWN_FWD = []
_PAWN_CAP_W = []
_PAWN_CAP_B = []
for _r in range(BH):
    for _c in range(BW):
        _KNIGHT_ATT.append(tuple(
            (_r + dr) * BW + (_c + dc)
            for dr, dc in zip(_KNIGHT_DR, _KNIGHT_DC)
            if 0 <= _r + dr < BH and 0 <= _c + dc < BW))
        _KING_ATT.append(tuple(
            (_r + dr) * BW + (_c + dc)
            for dr, dc in zip(_KING_DR, _KING_DC)
            if 0 <= _r + dr < BH and 0 <= _c + dc < BW))
        _BISHOP_RAYS.append(tuple(
            tuple((_r + dr * k) * BW + (_c + dc * k)
                  for k in range(1, 8)
                  if 0 <= _r + dr * k < BH and 0 <= _c + dc * k < BW)
            for dr, dc in zip(_BISHOP_DR, _BISHOP_DC)))
        _ROOK_RAYS.append(tuple(
            tuple((_r + dr * k) * BW + (_c + dc * k)
                  for k in range(1, 8)
                  if 0 <= _r + dr * k < BH and 0 <= _c + dc * k < BW)
            for dr, dc in zip(_ROOK_DR, _ROOK_DC)))
        _PAWN_FWD.append((_r - 1) * BW + _c if _r - 1 >= 0 else -1)
        _PAWN_CAP_W.append(tuple(
            (_r - 1) * BW + (_c + dc)
            for dc in (-1, 1) if _r - 1 >= 0 and 0 <= _c + dc < BW))
        _PAWN_CAP_B.append(tuple(
            (_r + 1) * BW + (_c + dc)
            for dc in (-1, 1) if _r + 1 < BH and 0 <= _c + dc < BW))

MATE_BOUND = MATE - 1000   # scores beyond this encode mate distances
QMAX = 6                   # quiescence depth cap

# Inverse pawn tables: the squares a white/black pawn would have to stand on in
# order to attack a given square (same geometry as the opposite pawn's captures).
_PAWN_CAP_W_inv = _PAWN_CAP_B
_PAWN_CAP_B_inv = _PAWN_CAP_W

# Late-Move-Reduction lookup: R(d, i) = floor(1 + ln(d)*ln(i+1)/2) for d>2, i>2,
# precomputed so the search never calls math.log.
_LMR = [[0] * 64 for _ in range(64)]
for _d in range(3, 64):
    for _i in range(3, 64):
        _LMR[_d][_i] = max(0, 1 + int(math.log(_d) * math.log(_i + 1) / 2.0))



class _AbortSearch(Exception):
    """Raised when the per-move time budget is exhausted mid-search."""


def _idx(r, c):
    return r * BW + c


class B23CS1001:
    """Alpha-beta agent (fast internal rules copy). Class name = roll number."""

    # ------------------------------------------- time control (self-contained)
    # Everything the agent needs to budget its own clock lives in THIS file, so
    # the submission is a genuine single file: the official runner only has to
    # construct it and call get_best_move().  The tournament gives each player
    # 1 minute for the whole game (sudden death, no increment) and running out
    # of time LOSES, so the allocator always keeps a reserve back.
    #
    # Scheme (after the usual engine practice: a soft bound the search aims to
    # finish within, and a hard bound it may extend to when the position turns
    # out to be unstable - "panic time"):
    #   soft = remaining/moves_left * TIME_FRACTION * criticality
    #   hard = min(soft * HARD_FACTOR, spendable, remaining*MAX_SHARE_CRIT)
    GAME_SECONDS = 60.0            # 1-minute clock per player
    ASSUMED_MOVES = 75             # assumed total game length for the spread
    MIN_MOVES_LEFT = 8             # divisor floor for the spread
    TIME_FRACTION = 0.75           # share of the clock the game may consume
    MIN_MOVE_SECONDS = 0.15        # never move instantly
    HARD_FACTOR = 2.5              # hard bound = soft * this
    MAX_SHARE = 0.25               # one move never takes >25% of the clock
    MAX_SHARE_CRIT = 0.35          # ... nor >35% even in a critical spot
    RESERVE_SECONDS = 2.0          # always keep this much back (timeout = loss)
    ITER_PREDICT = 2.5             # projected cost of the next iteration
    # --- dynamic factors: positional advantage / criticality ---------------
    DRIVE_MULT = 3.0               # we can force mate: +600, the dominant term
    CHECK_MULT = 1.5               # in check: find the escape or lose the king
    FEW_MULT = 1.2                 # endgame: precision matters
    WIDE_ROOT = 25                 # this many legal moves = harder choice
    WIDE_MULT = 1.1
    PANIC_DROP = 100               # score fell this much -> think harder
    PANIC_MULT = 1.5
    EASY_STABILITY = 3             # same best move for N moves with a steady
    EASY_DROP = 30                 # score -> the move is obvious, spend less
    EASY_MULT = 0.5
    # --- check-avoidance (king pressure) evaluation ------------------------
    PRESS_SCALE = 1                # weight of the king-pressure difference
    PRESS_CAP = 30                 # hard cap (1.5 pawns): never beats material

    def __init__(self, engine, depth=6, max_depth=12):
        self.engine = engine
        self.nodes_expanded = 0
        self.depth = depth
        self.max_depth = max_depth
        self.time_used = 0.0
        self.test_budget = None
        self.max_ply = 60
        self._deadline = None
        self._soft_deadline = None
        self.b = None
        self.wtm = True
        self.tt = {}
        self.killers = {}
        self.history = [0] * (48 * 48)
        self.qnodes = 0
        # Cross-move search history driving the dynamic time control.
        self._last_stability = 0      # consecutive moves with an unchanged best
        self._last_drop = 0           # score drop over the previous move
        # King-pressure state (check avoidance), maintained incrementally.
        self._press_w = 0             # enemy pressure around the white king
        self._press_b = 0             # enemy pressure around the black king

    # ------------------------------------------------------------- setup
    def _load(self):
        b = []
        for row in self.engine.board:
            for p in row:
                b.append(_CODE[p])
        self.b = b
        self.wtm = self.engine.white_to_move
        # Incremental evaluation state (kept in sync on every make/unmake).
        static = 0
        wk = bk = -1
        wnk = bnk = 0
        wpow = bpow = 0
        for i in range(48):
            p = b[i]
            if p == E:
                continue
            static += _STATIC[p][i]
            if p == WK:
                wk = i
            elif p == BK:
                bk = i
            elif p < 6:
                wnk += 1
                wpow += _VAL[p]
            else:
                bnk += 1
                bpow += _VAL[p]
        self._static = static
        self._wk = wk
        self._bk = bk
        self.wnk = wnk
        self.bnk = bnk
        self.wpow = wpow
        self.bpow = bpow
        # King pressure (check avoidance): computed once here, then maintained
        # incrementally by _eval_add/_eval_sub.
        self._press_w = self._pressure_on(wk, False)
        self._press_b = self._pressure_on(bk, True)

    def _pressure_on(self, ksq, by_white):
        """From-scratch king pressure: enemy material near the king (tropism)."""
        if ksq < 0:
            return 0
        b = self.b
        prox = _PROX[ksq]
        tot = 0
        for i in range(48):
            p = b[i]
            if p != E and (p < 6) == by_white:
                w = _WT[p]
                if w:
                    tot += w * prox[i]
        return tot

    def _press_add(self, code, sq, sign):
        """Add/remove `code`'s contribution to the pressure on the enemy king."""
        w = _WT[code]
        if w:
            if code < 6:
                ek = self._bk
                if ek >= 0:
                    self._press_b += sign * w * _PROX[ek][sq]
            else:
                ek = self._wk
                if ek >= 0:
                    self._press_w += sign * w * _PROX[ek][sq]

    def _press_captured(self, code, sq, sign):
        """Add/remove a captured piece's pressure on the king of its own colour."""
        w = _WT[code]
        if w:
            if code < 6:
                ck = self._wk
                if ck >= 0:
                    self._press_w += sign * w * _PROX[ck][sq]
            else:
                ck = self._bk
                if ck >= 0:
                    self._press_b += sign * w * _PROX[ck][sq]

    def _eval_add(self, piece, fr, to, captured):
        """Incremental evaluation update when `piece` moves fr -> to."""
        self._static += _STATIC[piece][to] - _STATIC[piece][fr]
        king_moved = piece == WK or piece == BK
        if piece == WK:
            self._wk = to
        elif piece == BK:
            self._bk = to
        else:
            # The mover carries its king pressure along with it.
            self._press_add(piece, to, 1)
            self._press_add(piece, fr, -1)
        if captured:
            self._static -= _STATIC[captured][to]
            if captured < 6:
                self.wnk -= 1
                self.wpow -= _VAL[captured]
            else:
                self.bnk -= 1
                self.bpow -= _VAL[captured]
            self._press_captured(captured, to, -1)
        if king_moved:
            # Every enemy piece's proximity changes when a king moves.
            if piece == WK:
                self._press_w = self._pressure_on(to, False)
            else:
                self._press_b = self._pressure_on(to, True)

    def _eval_sub(self, piece, fr, to, captured):
        """Reverse of _eval_add (undo a move)."""
        self._static -= _STATIC[piece][to] - _STATIC[piece][fr]
        king_moved = piece == WK or piece == BK
        if piece == WK:
            self._wk = fr
        elif piece == BK:
            self._bk = fr
        else:
            self._press_add(piece, to, -1)
            self._press_add(piece, fr, 1)
        if captured:
            self._static += _STATIC[captured][to]
            if captured < 6:
                self.wnk += 1
                self.wpow += _VAL[captured]
            else:
                self.bnk += 1
                self.bpow += _VAL[captured]
            self._press_captured(captured, to, 1)
        if king_moved:
            if piece == WK:
                self._press_w = self._pressure_on(fr, False)
            else:
                self._press_b = self._pressure_on(fr, True)

    def _position_key(self):
        # Each square is a 0..10 code, so bytes() is a very fast exact key.
        return (self.wtm, bytes(self.b))

    # ------------------------------------------------------------ public API
    def get_best_move(self):
        engine = self.engine
        self._load()
        root = engine.get_legal_moves()
        if not root:
            return None

        t0 = time.monotonic()
        # A forced move needs no search at all.
        if len(root) == 1:
            self.time_used += time.monotonic() - t0
            return root[0]

        moves_done = (len(engine.move_log) + 1) // 2
        myk, opk, in_check, few, drive = self._snapshot()
        # Dynamic time control: (soft, hard) seconds for this move, derived from
        # the clock, the game phase and how critical the position is.
        soft, hard = self._plan_time(moves_done, in_check, drive, few, len(root))
        if self.test_budget is not None:
            soft = hard = self.test_budget     # test-harness override only
        soft_limit = t0 + soft
        hard_limit = t0 + hard
        self._deadline = hard_limit
        self._soft_deadline = soft_limit

        best = None
        prev_score = None
        changes = 0            # root best-move changes during this move
        drop = 0               # score fall over the last iteration
        iter_secs = 0.0        # duration of the last completed iteration
        limit = soft_limit     # raised to hard_limit while the move is unstable
        cap = self.max_depth + (10 if drive else 8 if few else 0)
        try:
            for d in range(1, cap + 1):
                now = time.monotonic()
                if now >= hard_limit:
                    break
                # Do not start an iteration that cannot finish in the budget.
                if best is not None and \
                        now + iter_secs * self.ITER_PREDICT > limit:
                    break
                it0 = now
                score_before = prev_score
                if best is not None:
                    bt = (best.start_row, best.start_col,
                          best.end_row, best.end_col)
                else:
                    bt = None
                ordered = sorted(
                    root,
                    key=lambda m: (
                        0 if bt is not None and
                        (m.start_row, m.start_col, m.end_row, m.end_col) == bt
                        else 1,
                        self._root_key(m, drive, myk, opk),
                    ))
                # Aspiration window centred on the last *search* score (the
                # previous version centred it on the static eval, which made
                # every root move fail low and silently kept a stale move).
                half = None if prev_score is None else 80
                while True:
                    if half is None:
                        alpha, beta = -MATE * 2, MATE * 2
                    else:
                        alpha = max(-MATE * 2, prev_score - half)
                        beta = min(MATE * 2, prev_score + half)
                    a0 = alpha
                    cur = None
                    cur_score = -MATE * 2
                    for mv in ordered:
                        fr = _idx(mv.start_row, mv.start_col)
                        to = _idx(mv.end_row, mv.end_col)
                        piece = self.b[fr]
                        captured = self.b[to]
                        self.b[to] = piece
                        self.b[fr] = E
                        self.wtm = not self.wtm
                        self._eval_add(piece, fr, to, captured)
                        try:
                            score = -self._negamax(d - 1, -beta, -alpha, 1)
                        except _AbortSearch:
                            self.b[fr] = piece
                            self.b[to] = captured
                            self.wtm = not self.wtm
                            self._eval_sub(piece, fr, to, captured)
                            raise
                        self.b[fr] = piece
                        self.b[to] = captured
                        self.wtm = not self.wtm
                        self._eval_sub(piece, fr, to, captured)
                        if score > cur_score:
                            cur_score = score
                        if score > alpha:
                            alpha = score
                            cur = mv
                        if alpha >= beta:
                            break
                    if cur is not None:
                        best = cur
                    if half is None:
                        prev_score = cur_score
                        break
                    if cur_score <= a0 or cur_score >= beta:
                        # Fail low / fail high: widen and re-search this depth.
                        half = None if half >= MATE else min(MATE, half * 4)
                        prev_score = cur_score
                        continue
                    prev_score = cur_score
                    break
                self.depth = d
                iter_secs = time.monotonic() - it0
                new_key = ((best.start_row, best.start_col, best.end_row,
                            best.end_col) if best is not None else None)
                if bt is not None and new_key != bt:
                    changes += 1
                    limit = hard_limit          # unstable move -> panic time
                if score_before is not None:
                    drop = max(0, score_before - cur_score)
                    if drop >= self.PANIC_DROP:
                        limit = hard_limit
                # A forced mate for us is already the best possible outcome.
                if cur_score > MATE_BOUND:
                    break
        except _AbortSearch:
            pass
        finally:
            self._deadline = None
            self._soft_deadline = None
            self.time_used += time.monotonic() - t0
            self._last_drop = drop
            # An unchanged root move over consecutive moves = "easy move".
            if changes == 0 and best is not None:
                self._last_stability += 1
            else:
                self._last_stability = 0
        return best if best is not None else root[0]

    def _plan_time(self, moves_done, in_check, drive, few, n_root):
        """Return (soft, hard) seconds for this move - the dynamic time control.

        Sudden death on a 60 s clock: the remaining time is spread over the moves
        still expected, then scaled by how critical the position looks.  The
        multipliers encode what this variant rewards - a mate is worth 600, so
        mate-critical positions (we are driving for mate, or we are in check) get
        by far the most time, while a position whose best move has not changed
        for several plies and whose score is steady is treated as obvious and
        gets less.  `hard` is the ceiling the search may extend to when the root
        move keeps changing or the score drops (panic time); it is capped so no
        single move can eat the clock, because running out of time LOSES.
        """
        remaining = max(0.05, self.GAME_SECONDS - self.time_used)
        moves_left = max(self.MIN_MOVES_LEFT, self.ASSUMED_MOVES - moves_done)
        base = remaining * self.TIME_FRACTION / moves_left

        mult = 1.0
        if drive:
            mult *= self.DRIVE_MULT        # a finishable mate is +600
        elif in_check:
            mult *= self.CHECK_MULT        # must find the escape
        if few:
            mult *= self.FEW_MULT          # endgame precision
        if n_root >= self.WIDE_ROOT:
            mult *= self.WIDE_MULT         # many options = harder decision
        if self._last_drop >= self.PANIC_DROP:
            mult *= self.PANIC_MULT        # the score just fell: trouble
        elif (self._last_stability >= self.EASY_STABILITY and
                self._last_drop <= self.EASY_DROP):
            mult *= self.EASY_MULT         # obvious and stable: move along

        reserve = min(self.RESERVE_SECONDS, remaining * 0.5)
        # The floor must scale with the clock, otherwise a fixed minimum would
        # spend more than half of what is left in a near-empty clock and could
        # lose on time.  Together with MAX_SHARE_CRIT this guarantees that no
        # move ever takes more than half of the remaining time.
        floor = min(self.MIN_MOVE_SECONDS, remaining * 0.1)
        spendable = max(floor, remaining - reserve)
        soft = max(floor, min(base * mult, spendable,
                              remaining * self.MAX_SHARE))
        hard = max(soft, min(soft * self.HARD_FACTOR, spendable,
                             remaining * self.MAX_SHARE_CRIT))
        return soft, hard

    def _root_key(self, mv, drive, myk, opk):
        """Root move ordering: in mate-drive prefer checks, then captures."""
        key = 0
        if drive and opk >= 0:
            my = 'w' if self.wtm else 'b'
            fr = _idx(mv.start_row, mv.start_col)
            to = _idx(mv.end_row, mv.end_col)
            piece = self.b[fr]
            captured = self.b[to]
            self.b[to] = piece
            self.b[fr] = E
            nk = to if to == opk else opk
            if nk >= 0 and self._attacked(nk, my):
                key -= 200000
            self.b[fr] = piece
            self.b[to] = captured
        cap = self.b[_idx(mv.end_row, mv.end_col)]
        if cap != E:
            key -= 100000 + _VAL[cap]
        return key

    # ---------------------------------------------------------------- search
    def _null_move_allowed(self, in_check, depth):
        if in_check or depth < 2:
            return False
        for p in self.b:
            if p != E and _TYPE[p] != 5:
                return True
        return False

    def _negamax(self, depth, alpha, beta, ply):
        b = self.b
        ng = self._negamax          # local binding speeds up recursion
        self.nodes_expanded += 1

        if (self.nodes_expanded & 255) == 0 and self._deadline is not None \
            and time.monotonic() >= self._deadline:
            raise _AbortSearch()

        key = self._position_key()
        entry = self.tt.get(key)
        tt_move = None if entry is None else entry.best_move

        if entry is not None and entry.depth >= depth:
            value = entry.value
            # Convert the stored node-relative mate score back to root-relative.
            if value > MATE_BOUND:
                value -= ply
            elif value < -MATE_BOUND:
                value += ply
            if entry.flag == TT_EXACT:
                return value
            if entry.flag == TT_LOWER:
                if value >= beta:
                    return value
                if value > alpha:
                    alpha = value
            elif entry.flag == TT_UPPER:
                if value <= alpha:
                    return value
                if value < beta:
                    beta = value

        # Horizon handling runs before the board scan: quiescence and the eval
        # recompute what they need, so _snapshot is skipped at leaf nodes.
        if depth <= 0:
            return self._quiescence(alpha, beta)
        if ply > self.max_ply:
            return self._evaluate_stm()

        myk, opk, in_check, few, drive = self._snapshot()

        stand_pat = None
        if not in_check:
            if depth <= 3:
                stand_pat = self._evaluate_stm()
                if stand_pat - 120 * depth >= beta:
                    return stand_pat
            if self._null_move_allowed(in_check, depth):
                self.wtm = not self.wtm
                score = -ng(depth - 2, -beta, -beta + 1, ply + 1)
                self.wtm = not self.wtm
                if score >= beta:
                    return beta

        legal = self._legal_moves(myk, drive, opk, in_check)
        if not legal:
            return -(MATE - ply) if in_check else 0

        # Ordering inputs are hoisted out of the per-move key function (the
        # killer lookup used to run once per move inside the key lambda).
        killers = self.killers.get(depth, ())
        k0 = killers[0] if killers else -1
        k1 = killers[1] if len(killers) > 1 else -1
        if tt_move is not None:
            tt_fr, tt_to = tt_move
        else:
            tt_fr = tt_to = -1
        hist = self.history
        ordered = sorted(
            legal,
            key=lambda m: self._order_key(m, drive, b, tt_fr, tt_to,
                                          k0, k1, hist),
        )

        best_move = None
        alpha_orig = alpha
        best_score = -MATE
        for idx, (fr, to, captured, gives) in enumerate(ordered):
            piece = b[fr]
            b[to] = piece
            b[fr] = E
            self.wtm = not self.wtm
            self._eval_add(piece, fr, to, captured)
            try:
                if idx == 0:
                    score = -ng(depth - 1, -beta, -alpha, ply + 1)
                else:
                    search_depth = depth - 1
                    if depth >= 3 and captured == E and not gives \
                            and not in_check:
                        search_depth -= _LMR[depth if depth < 64 else 63][idx]
                    if search_depth < 0:
                        search_depth = 0
                    score = -ng(search_depth, -alpha - 1, -alpha, ply + 1)
                    if score > alpha and score < beta:
                        # PVS re-search with the full window.
                        score = -ng(depth - 1, -beta, -alpha, ply + 1)
            except _AbortSearch:
                b[fr] = piece
                b[to] = captured
                self.wtm = not self.wtm
                self._eval_sub(piece, fr, to, captured)
                raise
            b[fr] = piece
            b[to] = captured
            self.wtm = not self.wtm
            self._eval_sub(piece, fr, to, captured)
            if score > best_score:
                best_score = score
                best_move = (fr, to)
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break

        if best_move is not None:
            bmove = best_move[0] * 48 + best_move[1]
            self._record_killer(depth, best_move)
            hbonus = depth * depth
            hval = self.history[bmove]
            # Bounded ("gravity") update keeps early scores from dominating.
            self.history[bmove] = hval + hbonus - hval * hbonus // 16384
        if best_score <= alpha_orig:
            flag = TT_UPPER
        elif best_score >= beta:
            flag = TT_LOWER
        else:
            flag = TT_EXACT
        store_val = best_score
        if best_score > MATE_BOUND:
            store_val = best_score + ply
        elif best_score < -MATE_BOUND:
            store_val = best_score - ply
        self.tt[key] = TTEntry(depth, flag, store_val, best_move)
        return best_score

    def _ordering_key(self, move, drive, depth, tt_move=None):
        fr, to, captured, gives = move
        ht = fr * 48 + to
        key = 0
        if tt_move is not None and (fr, to) == tt_move:
            key -= 10**9
        if drive and gives:
            key -= 300000
        if captured:
            key -= 100000 + _PTVAL[_TYPE[captured]] * 100 \
                - _PTVAL[_TYPE[self.b[fr]]]
        if ht in self.killers.get(depth, ()):
            key -= 50000
        key -= self.history[ht]
        return key

    def _order_key(self, move, drive, b, tt_fr, tt_to, k0, k1, hist):
        """Hot ordering key with all lookups pre-resolved by the caller
        (TT move, killers and history are passed in)."""
        fr, to, captured, gives = move
        key = 0
        if fr == tt_fr and to == tt_to:
            key -= 10 ** 9
        if drive and gives:
            key -= 300000
        if captured:
            key -= 100000 + _PTVAL[_TYPE[captured]] * 100 \
                - _PTVAL[_TYPE[b[fr]]]
        ht = fr * 48 + to
        if ht == k0 or ht == k1:
            key -= 50000
        key -= hist[ht]
        return key

    def _record_killer(self, depth, move):
        if move is None:
            return
        fm = move[0] * 48 + move[1]
        lst = self.killers.setdefault(depth, [])
        if fm in lst:
            return
        lst.insert(0, fm)
        if len(lst) > 2:
            lst.pop()

    def _capture_moves(self, ksq, opp, in_check=False):
        """Legal capture moves only (used by quiescence; avoids generating
        quiet moves and keeps the delta-pruning friendly 4-tuple shape)."""
        b = self.b
        my_white = self.wtm
        found = []
        ap = found.append
        for i in range(48):
            p = b[i]
            if p == E:
                continue
            if (p < 6) != my_white:
                continue
            pt = _TYPE[p]
            if pt == 1:
                caps = _PAWN_CAP_W[i] if my_white else _PAWN_CAP_B[i]
                for t in caps:
                    tp = b[t]
                    if tp and (tp < 6) != my_white:
                        ap((i, t, tp))
            elif pt == 2:
                for t in _KNIGHT_ATT[i]:
                    tp = b[t]
                    if tp and (tp < 6) != my_white:
                        ap((i, t, tp))
            elif pt == 5:
                for t in _KING_ATT[i]:
                    tp = b[t]
                    if tp and (tp < 6) != my_white:
                        ap((i, t, tp))
            elif pt == 3:
                for ray in _BISHOP_RAYS[i]:
                    for t in ray:
                        tp = b[t]
                        if tp:
                            if (tp < 6) != my_white:
                                ap((i, t, tp))
                            break
            elif pt == 4:
                for ray in _ROOK_RAYS[i]:
                    for t in ray:
                        tp = b[t]
                        if tp:
                            if (tp < 6) != my_white:
                                ap((i, t, tp))
                            break
        if not found:
            return []
        legal = []
        if in_check or ksq < 0:
            pinned = 0
            test_all = True
        else:
            # Same pin shortcut as _legal_moves: only king moves and pinned
            # pieces can expose the king, so most captures skip the test.
            pinned = self._pinned_squares(ksq, opp)
            test_all = False
        for fr, to, captured in found:
            if not (test_all or fr == ksq or (pinned >> fr) & 1):
                legal.append((fr, to, captured, False))
                continue
            piece = b[fr]
            b[to] = piece
            b[fr] = E
            king = to if fr == ksq else ksq
            ok = not (king >= 0 and self._attacked(king, opp))
            b[fr] = piece
            b[to] = captured
            if ok:
                legal.append((fr, to, captured, False))
        return legal

    def _quiescence(self, alpha, beta, qdepth=0):
        """Forcing-move search: captures, plus full evasions while in check.

        This replaces the old "generate every legal move then filter" version
        (which paid ~30 attack tests per node for a handful of captures) with a
        capture-only generator, MVV-LVA ordering and delta pruning.  In check,
        all evasions are searched so simple mates are still detected at the
        horizon.
        """
        self.qnodes += 1
        b = self.b
        my_white = self.wtm
        my = 'w' if my_white else 'b'
        opp = 'b' if my_white else 'w'
        # King squares are tracked incrementally (self._wk / self._bk).
        stand_pat = self._evaluate_stm()
        myk = self._wk if my_white else self._bk
        in_check = myk >= 0 and self._attacked(myk, opp)

        if in_check:
            legal = self._legal_moves(myk, False, -1, True)
            if not legal:
                return -(MATE - 1)
            if qdepth >= QMAX:
                return self._evaluate_stm()
            legal.sort(key=lambda m: self._ordering_key(m, False, 0))
            for fr, to, captured, gives in legal:
                piece = b[fr]
                b[to] = piece
                b[fr] = E
                self.wtm = not self.wtm
                self._eval_add(piece, fr, to, captured)
                score = -self._quiescence(-beta, -alpha, qdepth + 1)
                b[fr] = piece
                b[to] = captured
                self.wtm = not self.wtm
                self._eval_sub(piece, fr, to, captured)
                if score > alpha:
                    alpha = score
                if alpha >= beta:
                    return alpha
            return alpha

        if stand_pat >= beta:
            return stand_pat
        if stand_pat > alpha:
            alpha = stand_pat
        if qdepth >= QMAX:
            return alpha

        captures = self._capture_moves(myk, opp)
        if not captures:
            return alpha
        captures.sort(key=lambda m: self._ordering_key(m, False, 0))
        for fr, to, captured, gives in captures:
            if stand_pat + _VAL[captured] + 200 < alpha:
                continue                       # delta pruning
            piece = b[fr]
            b[to] = piece
            b[fr] = E
            self.wtm = not self.wtm
            self._eval_add(piece, fr, to, captured)
            score = -self._quiescence(-beta, -alpha, qdepth + 1)
            b[fr] = piece
            b[to] = captured
            self.wtm = not self.wtm
            self._eval_sub(piece, fr, to, captured)
            if score > alpha:
                alpha = score
            if alpha >= beta:
                return alpha
        return alpha

    # ------------------------------------------------------------- snapshot
    def _snapshot(self):
        """One-scan board summary used at every search node.

        Returns (my_king, opp_king, in_check, few_pieces, mate_drive).
        mate_drive is true when the side to move is up by >= a rook against an
        opponent that has at most one non-king piece left.
        """
        my_white = self.wtm
        opp = 'b' if my_white else 'w'
        b = self.b
        myk = opk = -1
        tot = 0
        pow_my = pow_opp = 0
        opp_nk = 0
        for i in range(48):
            p = b[i]
            if p == E:
                continue
            tot += 1
            if p == WK:
                if my_white:
                    myk = i
                else:
                    opk = i
                continue
            if p == BK:
                if my_white:
                    opk = i
                else:
                    myk = i
                continue
            v = _VAL[p]
            if (p < 6) == my_white:
                pow_my += v
            else:
                pow_opp += v
                opp_nk += 1
        in_check = myk >= 0 and self._attacked(myk, opp)
        drive = opp_nk <= 1 and pow_my - pow_opp >= 100
        return myk, opk, in_check, tot <= FEW_PIECE_LIMIT, drive

    # ------------------------------------------------------ attack/legality
    def _find_king(self, color):
        b = self.b
        target = WK if color == 'w' else BK
        for i in range(48):
            if b[i] == target:
                return i
        return -1

    def _attacked(self, sq, by):
        """Is square `sq` attacked by any piece of colour `by`?

        Uses the precomputed geometry tables (no bounds tests, no _idx calls),
        which is the single hottest helper in the search.
        """
        b = self.b
        if by == 'w':
            pn = WN
            pk = WK
            pb = WB
            prk = WR
            for t in _PAWN_CAP_W_inv[sq]:
                if b[t] == WP:
                    return True
        else:
            pn = BN
            pk = BK
            pb = BB
            prk = BR
            for t in _PAWN_CAP_B_inv[sq]:
                if b[t] == BP:
                    return True
        for t in _KNIGHT_ATT[sq]:
            if b[t] == pn:
                return True
        for t in _KING_ATT[sq]:
            if b[t] == pk:
                return True
        for ray in _BISHOP_RAYS[sq]:
            for t in ray:
                p = b[t]
                if p:
                    if p == pb:
                        return True
                    break
        for ray in _ROOK_RAYS[sq]:
            for t in ray:
                p = b[t]
                if p:
                    if p == prk:
                        return True
                    break
        return False

    def _pinned_squares(self, ksq, opp):
        """Bitmask of own pieces standing between the king and an enemy slider
        (only these quiet moves can ever expose the king)."""
        b = self.b
        pinned = 0
        opp_white = opp == 'w'
        bishop = WB if opp_white else BB
        rook = WR if opp_white else BR
        for ray in _ROOK_RAYS[ksq]:
            first = -1
            for t in ray:
                p = b[t]
                if p:
                    if first < 0:
                        if (p < 6) == opp_white:   # enemy blocks the ray first
                            break
                        first = t
                    else:
                        if p == rook:
                            pinned |= 1 << first
                        break
        for ray in _BISHOP_RAYS[ksq]:
            first = -1
            for t in ray:
                p = b[t]
                if p:
                    if first < 0:
                        if (p < 6) == opp_white:
                            break
                        first = t
                    else:
                        if p == bishop:
                            pinned |= 1 << first
                        break
        return pinned

    def _legal_moves(self, ksq, drive=False, opk=-1, in_check=None):
        """Legal moves as (from, to, captured, gives_check) tuples.

        Rules mirror board.py exactly (generate-then-validate); the king of the
        side that just moved must not be left in check.  When `drive` is set we
        additionally record whether each move gives check (mate-drive needs to
        order forcing moves first).
        """
        b = self.b
        my_white = self.wtm
        my = 'w' if my_white else 'b'
        opp = 'b' if my_white else 'w'
        pseudo = []
        ap = pseudo.append
        for i in range(48):
            p = b[i]
            if p == E:
                continue
            if (p < 6) != my_white:
                continue
            ptype = _TYPE[p]
            if ptype == 1:
                f = _PAWN_FWD[i] if my_white else (
                    i + BW if i + BW < 48 else -1)
                if f >= 0 and b[f] == E:
                    ap((i, f))
                for t in (_PAWN_CAP_W[i] if my_white else _PAWN_CAP_B[i]):
                    tp = b[t]
                    if tp and (tp < 6) != my_white:
                        ap((i, t))
            elif ptype == 2:
                for t in _KNIGHT_ATT[i]:
                    tp = b[t]
                    if tp == E or (tp < 6) != my_white:
                        ap((i, t))
            elif ptype == 3:
                for ray in _BISHOP_RAYS[i]:
                    for t in ray:
                        tp = b[t]
                        if tp == E:
                            ap((i, t))
                        else:
                            if (tp < 6) != my_white:
                                ap((i, t))
                            break
            elif ptype == 4:
                for ray in _ROOK_RAYS[i]:
                    for t in ray:
                        tp = b[t]
                        if tp == E:
                            ap((i, t))
                        else:
                            if (tp < 6) != my_white:
                                ap((i, t))
                            break
            elif ptype == 5:
                for t in _KING_ATT[i]:
                    tp = b[t]
                    if tp == E or (tp < 6) != my_white:
                        ap((i, t))

        legal = []
        if ksq < 0:
            for fr, to in pseudo:
                legal.append((fr, to, b[to], False))
            return legal
        # Only king moves and pinned-piece moves can expose the king, so when
        # not in check (and not driving for mate, which needs `gives`) the
        # expensive attack test is skipped for every other move.
        if in_check is None:
            in_check = self._attacked(ksq, opp)
        must_test = in_check or (drive and opk >= 0)
        pinned = 0 if must_test else self._pinned_squares(ksq, opp)
        for fr, to in pseudo:
            if not (must_test or fr == ksq or (pinned >> fr) & 1):
                legal.append((fr, to, b[to], False))
                continue
            captured = b[to]
            piece = b[fr]
            b[to] = piece
            b[fr] = E
            king = to if fr == ksq else ksq
            ok = not (king >= 0 and self._attacked(king, opp))
            gives = False
            if drive and opk >= 0:
                nk = to if to == opk else opk
                gives = nk >= 0 and self._attacked(nk, my)
            b[fr] = piece
            b[to] = captured
            if ok:
                legal.append((fr, to, captured, gives))
        return legal

    # --------------------------------------------------------- evaluation
    def _evaluate_stm(self):
        v = self._evaluate_white()
        return v if self.wtm else -v

    def _evaluate_white(self):
        """Static evaluation (white-positive), O(1) in the midgame.

        Material + piece-square tables dominate (they mirror the tournament
        scoring), with small centralisation, passed-pawn and king-safety terms.
        The material/PST/centre part and the king squares are maintained
        incrementally on every make/unmake (see `_load`); only the endgame
        passed-pawn scan touches the board.
        """
        b = self.b
        # Everything below is maintained incrementally (see _load / make-unmake),
        # so the common midgame evaluation is O(1) - no 48-square scan.
        score = self._static
        w_king = self._wk
        b_king = self._bk
        w_nk = self.wnk
        b_nk = self.bnk
        w_pow = self.wpow
        b_pow = self.bpow
        nk = w_nk + b_nk
        if nk <= 12:
            # Passed-pawn scan (endgame only), so the common midgame eval never
            # has to collect pawn squares.
            for i in range(48):
                p = b[i]
                if p == WP:
                    r = i // BW
                    c = i - r * BW
                    for rr in range(r - 1, -1, -1):
                        base = rr * BW
                        if b[base + c] == BP or \
                                (c > 0 and b[base + c - 1] == BP) or \
                                (c + 1 < BW and b[base + c + 1] == BP):
                            break
                    else:
                        score += 10 + (BH - 1 - r) * 4
                elif p == BP:
                    r = i // BW
                    c = i - r * BW
                    for rr in range(r + 1, BH):
                        base = rr * BW
                        if b[base + c] == WP or \
                                (c > 0 and b[base + c - 1] == WP) or \
                                (c + 1 < BW and b[base + c + 1] == WP):
                            break
                    else:
                        score -= 10 + r * 4

        if w_king >= 0 and b_king >= 0:
            wr = w_king // BW
            wc = w_king - wr * BW
            br = b_king // BW
            bc = b_king - br * BW
            if nk <= 6:
                score += KING_PST[wr][wc]
                score -= KING_PST[BH - 1 - br][bc]
            else:
                # King shield (inlined): friendly pawns in front of each king.
                rr = wr - 1
                if rr >= 0:
                    base = rr * BW
                    if wc > 0 and b[base + wc - 1] == WP:
                        score += 6
                    if b[base + wc] == WP:
                        score += 6
                    if wc + 1 < BW and b[base + wc + 1] == WP:
                        score += 6
                rr = br + 1
                if rr < BH:
                    base = rr * BW
                    if bc > 0 and b[base + bc - 1] == BP:
                        score -= 6
                    if b[base + bc] == BP:
                        score -= 6
                    if bc + 1 < BW and b[base + bc + 1] == BP:
                        score -= 6

        # Check avoidance: enemy material near a king is what generates the
        # endless stream of +2 checks (and then the mates).  Bounded by
        # PRESS_CAP and only counted while the attacker still has real material,
        # so it can never outweigh material - and never a mate, because a mate
        # score is ~100000 and always overrules any evaluation term.
        press = 0
        if w_nk >= 2:                     # white attacks the black king
            press += self._press_b
        if b_nk >= 2:                     # black attacks the white king
            press -= self._press_w
        if press:
            if press > self.PRESS_CAP:
                press = self.PRESS_CAP
            elif press < -self.PRESS_CAP:
                press = -self.PRESS_CAP
            score += press * self.PRESS_SCALE

        if w_pow - b_pow >= 100 and b_nk <= 1 and w_king >= 0 and b_king >= 0:
            score += self._drive_bonus('w', w_king, b_king)
        elif b_pow - w_pow >= 100 and w_nk <= 1 and w_king >= 0 \
                and b_king >= 0:
            score -= self._drive_bonus('b', b_king, w_king)
        # Defensive drive: the case the bonus above does *not* cover, i.e. the
        # trailing side still has two or more non-king pieces.  Losing a king is
        # worth 600, so a boxed-in king is penalised well before the endgame has
        # been stripped down to a bare king.
        if w_king >= 0 and b_king >= 0:
            if b_pow - w_pow >= 100 and w_nk >= 2:
                score -= self._king_danger(w_king, 'b')
            elif w_pow - b_pow >= 100 and b_nk >= 2:
                score += self._king_danger(b_king, 'w')
        # Expose the king squares so quiescence does not need a second scan.
        self._wk = w_king
        self._bk = b_king
        return score

    def _drive_bonus(self, leader, lk, ek):
        """Positive score (for `leader`) pushing the enemy king to the edge
        while the leader's king and rook(s) move in for the mate."""
        b = self.b
        er, ec = divmod(ek, BW)
        lr, lc = divmod(lk, BW)
        score = 0
        rowd = min(er, BH - 1 - er)
        cold = min(ec, BW - 1 - ec)
        if rowd == 0 and cold == 0:          # enemy king in a corner
            score += 140
        elif rowd == 0 or cold == 0:         # on an edge
            score += 80
        elif rowd <= 1 or cold <= 1:
            score += 30
        dist = max(abs(lr - er), abs(lc - ec))
        if dist <= 2:
            score += 50
        elif dist <= 3:
            score += 25
        elif dist <= 4:
            score += 10
        # Rooks that cut off a rank or file gain a bonus.
        cuts = 0
        for i in range(48):
            if b[i] == (WR if leader == 'w' else BR):
                rr = i // BW
                rc = i - rr * BW
                if rr == er or rc == ec:
                    cuts += 1
        score += 40 * cuts
        return score

    def _king_danger(self, ksq, opp):
        """How endangered a king is while `opp` is the side with mating material.

        Defensive counterpart of `_drive_bonus`: only *defensive* facts are
        scored (edge proximity, escape squares, cut-off enemy rooks), because the
        attacker's incentives (bring my king near, cut with rooks) are already
        supplied by `_drive_bonus`.  The escape-square count is the part that
        catches a mate that is one tempo away; it is only paid for near an edge,
        where being boxed in can actually matter.
        """
        b = self.b
        kr, kc = divmod(ksq, BW)
        rowd = min(kr, BH - 1 - kr)
        cold = min(kc, BW - 1 - kc)
        score = 0
        if rowd == 0 and cold == 0:          # trapped in a corner
            score += 60
        elif rowd == 0 or cold == 0:         # on an edge
            score += 30
        elif rowd <= 1 and cold <= 1:
            score += 10
        if rowd <= 1 or cold <= 1:
            escapes = 0
            for t in _KING_ATT[ksq]:
                if b[t] == E and not self._attacked(t, opp):
                    escapes += 1
            if escapes == 0:                 # every flight square is covered
                score += 200
            elif escapes == 1:
                score += 60
        rook = BR if opp == 'b' else WR
        for ray in _ROOK_RAYS[ksq]:
            for t in ray:
                p = b[t]
                if p:
                    if p == rook:            # rook cuts the king's rank/file
                        score += 45
                    break
        return score

    def evaluate_board(self, game_state="ongoing"):
        """Public evaluation hook used by the tournament runner."""
        if game_state == "checkmate":
            return MATE if self.engine.white_to_move else -MATE
        if game_state == "stalemate":
            return 0
        self._load()
        return self._evaluate_white()



