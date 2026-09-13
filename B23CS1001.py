"""Single-file adversarial agent for Spartans Chess (6x8, no queens)."""
import math
import time

from config import (
    BOARD_WIDTH as BW, BOARD_HEIGHT as BH,
    KING_PST_LATE_GAME,
    PAWN_PST, KNIGHT_PST, BISHOP_PST, ROOK_PST,
)


MATE = 100000            # scores for mates exceed any material total
FEW_PIECE_LIMIT = 9
KING_PST = KING_PST_LATE_GAME

TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2

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

_WT = (0, 1, 2, 2, 3, 0, 1, 2, 2, 3, 0)     # attacker weight by code (K -> 0)
_PROX_RANGE = 3                              # only pieces this close matter


def _build_prox(prox_range=_PROX_RANGE):
    """King tropism table: weight per (king square, enemy square) pair."""
    table = []
    for _k in range(48):
        _kr, _kc = divmod(_k, BW)
        _row = []
        for _s in range(48):
            _sr, _sc = divmod(_s, BW)
            _d = max(abs(_sr - _kr), abs(_sc - _kc))
            _row.append(0 if _d == 0 or _d > prox_range else 4 - _d)
        table.append(tuple(_row))
    return table


_PROX = _build_prox()

_STATIC = [None] * 11
_PST_OF = {1: PAWN_PST, 2: KNIGHT_PST, 3: BISHOP_PST, 4: ROOK_PST}
_POS_SCALE = 0.5


def _build_static(pos_scale=_POS_SCALE):
    """Precompute material + pos_scale * positional, per (piece code, square)."""
    table = [None] * 11
    for _code in range(1, 11):
        _t = _TYPE[_code]
        _tab = _PST_OF.get(_t)
        _white = _code < 6
        _base = _VAL_SIGN[_code]
        _arr = [0] * 48
        for _r in range(BH):
            _mr = _r if _white else BH - 1 - _r
            for _c in range(BW):
                _pos = 0
                if _tab is not None:
                    _pos += _tab[_mr][_c] if _white else -_tab[_mr][_c]
                if 2 <= _t <= 4:
                    _d = (_r - 3 if _r > 3 else 3 - _r) + \
                         (_c - 2 if _c > 2 else 2 - _c)
                    if _d < 3:
                        _pos += (3 - _d) if _white else -(3 - _d)
                _arr[_r * BW + _c] = _base + int(_pos * pos_scale)
        table[_code] = tuple(_arr)
    return table


_STATIC = _build_static()


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
_ABORT_MASK = 15
                           # move overshoots by is what eats the clock reserve
                           # over a long game, and it scales with NPS: at 255
                           # nodes it was ~2 s per game even under CPU load,
                           # against ~0.6 s here.  One time.monotonic() per 16
                           # nodes is well under 0.1% of the run time.

_PAWN_CAP_W_inv = _PAWN_CAP_B
_PAWN_CAP_B_inv = _PAWN_CAP_W

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

    # out to be unstable - "panic time"):
    GAME_SECONDS = 60.0            # 1-minute clock per player
    ASSUMED_MOVES = 75             # assumed total game length for the spread
    MIN_MOVES_LEFT = 8             # divisor floor for the spread
    TIME_FRACTION = 0.90           # share of the clock the game may consume
    MIN_MOVE_SECONDS = 0.30
    HARD_FACTOR = 2.5              # hard bound = soft * this
    MAX_SHARE = 0.25               # one move never takes >25% of the clock
    MAX_SHARE_CRIT = 0.35          # ... nor >35% even in a critical spot
    RESERVE_SECONDS = 5.0          # hard pool: never spent (timeout = loss)
    TINY_SLICE = 0.02              # seconds: the last-resort move allowance
    MIN_RESERVE = 4.5              # absolute floor: usage <= 60 - MIN_RESERVE
    ITER_PREDICT = 2.5             # projected cost of the next iteration
    DRIVE_MULT = 3.0               # we can force mate: +600, the dominant term
    CHECK_MULT = 1.5               # in check: find the escape or lose the king
    FEW_MULT = 1.6                 # endgame: precision matters (the phase that
                                   # actually decides material and checks)
    WIDE_ROOT = 25                 # this many legal moves = harder choice
    WIDE_MULT = 1.1
    PANIC_DROP = 100               # score fell this much -> think harder
    PANIC_MULT = 1.5
    EASY_STABILITY = 3             # same best move for N moves with a steady
    EASY_DROP = 30                 # score -> the move is obvious, spend less
    EASY_MULT = 0.5
    PRESS_SCALE = 1                # weight of the king-pressure difference
    PRESS_CAP = 30                 # hard cap (1.5 pawns): never beats material
    DANGER_EVEN = 0.5              # fraction of the danger term at level material
    DANGER_PRESS_MIN = 8           # tropism threshold for the gate above
    CHECK_EXT = 1
    CHECK_EXT_PLY = 5              # ... only this close to the root
    CHECK_EXT_BUDGET = 6           # ... and at most this many per move
    ADJ_CAP_PLY = 150              # ply at which the harness adjudicates
    ADJ_LATE_PLY = 40              # "late" = this close to the cap
    ADJ_LEAD_PTS = 20              # this far ahead/behind late: play to the score
    ADJ_SAFE_PTS = 20              # protect mode: what counts as a real risk

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
        self._last_stability = 0      # consecutive moves with an unchanged best
        self._last_drop = 0           # score drop over the previous move
        self._press_w = 0             # enemy pressure around the white king
        self._press_b = 0             # enemy pressure around the black king
        self._adj_ply = 0             # plies already banked
        self._adj_caps = [0, 0]       # [white, black] captured-piece points
        self._adj_chk = [0, 0]        # [white, black] check points (2 each)
        self._adj_mode = 0            # 0 neutral, +1 protect a lead, -1 chase
        self._root_ck = {}            # (from, to) -> does this root move check?
        self._ext_left = 0            # check-extension budget for the current move

    def _load(self):
        b = []
        for row in self.engine.board:
            for p in row:
                b.append(_CODE[p])
        self.b = b
        self.wtm = self.engine.white_to_move
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
        # incrementally by _eval_add/_eval_sub.
        self._press_w = self._pressure_on(wk, False)
        self._press_b = self._pressure_on(bk, True)
        # follows from it.
        self._load_adj()

    def _load_adj(self):
        """Keep the official points table exact (captures + 2 per check given)."""
        log = self.engine.move_log
        if len(log) < self._adj_ply:           # undo, or a brand new game
            self._adj_ply = 0
            self._adj_caps = [0, 0]
            self._adj_chk = [0, 0]
        n = len(log)
        for i in range(self._adj_ply, n):
            v = _VAL[_CODE[log[i].piece_captured]]
            if v:
                self._adj_caps[i & 1] += v     # ply 0 is White's move
            # are in check right now, it was that move that gave it.
            if i == n - 1 and self.engine.is_in_check():
                self._adj_chk[i & 1] += 2
        self._adj_ply = n

        self._adj_mode = 0
        if self.ADJ_LEAD_PTS > 0 and self.ADJ_CAP_PLY - n <= self.ADJ_LATE_PLY:
            me = 0 if self.engine.white_to_move else 1
            lead = (self._adj_caps[me] + self._adj_chk[me]) - \
                   (self._adj_caps[1 - me] + self._adj_chk[1 - me])
            if lead >= self.ADJ_LEAD_PTS:
                self._adj_mode = 1
            elif lead <= -self.ADJ_LEAD_PTS:
                self._adj_mode = -1

    def _adj_points(self):
        """Official points banked so far, as (ours, theirs)."""
        me = 0 if self.engine.white_to_move else 1
        return (self._adj_caps[me] + self._adj_chk[me],
                self._adj_caps[1 - me] + self._adj_chk[1 - me])

    def _gives_check(self, mv, opk):
        """Does this move give check?  One attack test, root use only."""
        if opk < 0:
            return False
        fr = _idx(mv.start_row, mv.start_col)
        to = _idx(mv.end_row, mv.end_col)
        piece = self.b[fr]
        cap = self.b[to]
        self.b[to] = piece
        self.b[fr] = E
        nk = to if to == opk else opk
        ok = self._attacked(nk, 'w' if self.engine.white_to_move else 'b')
        self.b[fr] = piece
        self.b[to] = cap
        return ok

    def _root_hang(self, fr, to):
        """Value of the piece this move leaves attacked AND undefended."""
        b = self.b
        p = b[fr]
        if p == E:
            return 0
        cap = b[to]
        b[to] = p
        b[fr] = E
        my_white = self.wtm
        my = 'w' if my_white else 'b'
        opp = 'b' if my_white else 'w'
        q = b[to]
        hang = 0
        if q != E and (q < 6) == my_white and _TYPE[q] != 5 \
                and self._attacked(to, opp) and not self._attacked(to, my):
            hang = _VAL[q]
        b[fr] = p
        b[to] = cap
        return hang

    def _adj_bank(self, mv, opk=-1):
        """Bank OUR move into the tally (its captures, and a check if any)."""
        me = 0 if self.engine.white_to_move else 1
        v = _VAL[_CODE[mv.piece_captured]]
        if v:
            self._adj_caps[me] += v
        key = (_idx(mv.start_row, mv.start_col), _idx(mv.end_row, mv.end_col))
        gives = self._root_ck.get(key)
        if gives is None:
            gives = self._gives_check(mv, opk)
        if gives:
            self._adj_chk[me] += 2
        self._adj_ply = len(self.engine.move_log) + 1

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

    def _eval_add(self, piece, fr, to, captured):
        """Incremental evaluation update when `piece` moves fr -> to."""
        self._static += _STATIC[piece][to] - _STATIC[piece][fr]
        if captured:
            self._static -= _STATIC[captured][to]
            if captured < 6:
                self.wnk -= 1
                self.wpow -= _VAL[captured]
            else:
                self.bnk -= 1
                self.bpow -= _VAL[captured]
            cek = self._bk if captured < 6 else self._wk
            if cek >= 0 and _PROX[cek][to]:
                self._press_add(captured, to, -1)
        if piece == WK:
            self._wk = to
            self._press_w = self._pressure_on(to, False)
        elif piece == BK:
            self._bk = to
            self._press_b = self._pressure_on(to, True)
        else:
            # update is a no-op, so it is skipped.
            ek = self._bk if piece < 6 else self._wk
            if ek >= 0 and (_PROX[ek][to] or _PROX[ek][fr]):
                self._press_add(piece, to, 1)
                self._press_add(piece, fr, -1)

    def _eval_sub(self, piece, fr, to, captured):
        """Reverse of _eval_add (undo a move)."""
        self._static -= _STATIC[piece][to] - _STATIC[piece][fr]
        if captured:
            self._static += _STATIC[captured][to]
            if captured < 6:
                self.wnk += 1
                self.wpow += _VAL[captured]
            else:
                self.bnk += 1
                self.bpow += _VAL[captured]
            cek = self._bk if captured < 6 else self._wk
            if cek >= 0 and _PROX[cek][to]:
                self._press_add(captured, to, 1)
        if piece == WK:
            self._wk = fr
            self._press_w = self._pressure_on(fr, False)
        elif piece == BK:
            self._bk = fr
            self._press_b = self._pressure_on(fr, True)
        else:
            ek = self._bk if piece < 6 else self._wk
            if ek >= 0 and (_PROX[ek][to] or _PROX[ek][fr]):
                self._press_add(piece, to, -1)
                self._press_add(piece, fr, 1)

    def _position_key(self):
        return (self.wtm, bytes(self.b))

    def get_best_move(self):
        engine = self.engine
        t0 = time.monotonic()
        self._load()
        root = engine.get_legal_moves()
        if not root:
            self.time_used += time.monotonic() - t0
            return None

        # A forced move needs no search at all.
        if len(root) == 1:
            self._adj_bank(root[0], self._bk if self.wtm else self._wk)
            self.time_used += time.monotonic() - t0
            return root[0]

        moves_done = (len(engine.move_log) + 1) // 2
        myk, opk, in_check, few, drive = self._snapshot()
        soft, hard = self._plan_time(moves_done, in_check, drive, few, len(root))
        if self.test_budget is not None:
            soft = hard = self.test_budget     # test-harness override only
        soft_limit = t0 + soft
        hard_limit = t0 + hard
        self._deadline = hard_limit
        self._soft_deadline = soft_limit

        # transposition table stays consistent.
        self._root_ck = {}
        if opk >= 0:
            for mv in root:
                self._root_ck[(_idx(mv.start_row, mv.start_col),
                               _idx(mv.end_row, mv.end_col))] = \
                    self._gives_check(mv, opk)
        self._ext_left = self.CHECK_EXT_BUDGET

        best = None
        prev_score = None
        changes = 0            # root best-move changes during this move
        drop = 0               # score fall over the last iteration
        iter_secs = 0.0        # duration of the last completed iteration
        limit = soft_limit
        cap = self.max_depth + (10 if drive else 8 if few else 0)
        try:
            for d in range(1, cap + 1):
                now = time.monotonic()
                if now >= hard_limit:
                    break
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
                if cur_score > MATE_BOUND:
                    break
        except _AbortSearch:
            pass
        finally:
            self._deadline = None
            self._soft_deadline = None
            self.time_used += time.monotonic() - t0
            self._last_drop = drop
            if changes == 0 and best is not None:
                self._last_stability += 1
            else:
                self._last_stability = 0
        if best is None:
            best = root[0]                 # aborted before any move was scored
        # opponent's reply left to read.
        self._adj_bank(best, opk)
        return best

    def _plan_time(self, moves_done, in_check, drive, few, n_root):
        """Return (soft, hard) seconds for this move - the dynamic time control."""
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
        room = max(0.0, remaining - self.RESERVE_SECONDS)
        # capped by the pool, so it can never touch the reserve.
        floor = min(self.MIN_MOVE_SECONDS, remaining * 0.1, room)
        tiny = min(self.TINY_SLICE, max(0.0, remaining - self.MIN_RESERVE))
        if tiny > floor:
            floor = tiny
        spendable = max(floor, min(remaining - reserve, room))
        soft = max(floor, min(base * mult, spendable,
                              remaining * self.MAX_SHARE))
        hard = max(soft, min(soft * self.HARD_FACTOR, spendable,
                             remaining * self.MAX_SHARE_CRIT))
        guard = max(0.0, remaining - self.MIN_RESERVE)
        if hard > guard:                # absolute floor: never spent
            hard = guard
        if soft > hard:
            soft = hard
        return soft, hard

    def _root_key(self, mv, drive, myk, opk):
        """Root move ordering: mate-drive first, then the adjudication policy."""
        key = 0
        adj = self._adj_mode
        if (drive or adj) and opk >= 0:
            fr = _idx(mv.start_row, mv.start_col)
            to = _idx(mv.end_row, mv.end_col)
            piece = self.b[fr]
            captured = self.b[to]
            gives = self._root_ck.get((fr, to))
            if gives is None:                  # map not built: test it directly
                gives = self._gives_check(mv, opk)
            if drive and gives:
                key -= 200000
            elif adj < 0 and gives:
                key -= 20000                   # chasing points: checks are +2
            elif adj > 0 and captured == E:
                if self._root_hang(fr, to) >= self.ADJ_SAFE_PTS:
                    key += 30000 + _VAL[piece]
        cap = self.b[_idx(mv.end_row, mv.end_col)]
        if cap != E:
            key -= 100000 + _VAL[cap]
        return key

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

        if (self.nodes_expanded & _ABORT_MASK) == 0 and self._deadline is not None \
            and time.monotonic() >= self._deadline:
            raise _AbortSearch()

        key = self._position_key()
        entry = self.tt.get(key)
        tt_move = None if entry is None else entry.best_move

        if entry is not None and entry.depth >= depth:
            value = entry.value
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

        if depth <= 0:
            return self._quiescence(alpha, beta)
        if ply > self.max_ply:
            return self._evaluate_stm()

        myk, opk, in_check, few, drive = self._snapshot()

        # up and steal depth from the quiet positions.
        if in_check and self.CHECK_EXT and ply <= self.CHECK_EXT_PLY \
                and self._ext_left > 0:
            depth += 1
            self._ext_left -= 1

        stand_pat = None
        if not in_check:
            if depth <= 3:
                stand_pat = self._evaluate_stm()
                if stand_pat - 120 * depth >= beta:
                    return stand_pat
            if self._null_move_allowed(in_check, depth):
                self.wtm = not self.wtm
                try:
                    score = -ng(depth - 2, -beta, -beta + 1, ply + 1)
                except _AbortSearch:
                    # Keep make/unmake symmetric on every abort path: an
                    # stop the search in here).
                    self.wtm = not self.wtm
                    raise
                self.wtm = not self.wtm
                if score >= beta:
                    return beta

        legal = self._legal_moves(myk, drive, opk, in_check)
        if not legal:
            return -(MATE - ply) if in_check else 0

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
        """Hot ordering key with all lookups pre-resolved by the caller"""
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
        """Legal capture moves only (used by quiescence; avoids generating"""
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
        """Forcing-move search: captures, plus full evasions while in check."""
        self.qnodes += 1
        # blocks below, exactly like _negamax/root do.
        if self._deadline is not None and (self.qnodes & _ABORT_MASK) == 0 \
                and time.monotonic() >= self._deadline:
            raise _AbortSearch
        b = self.b
        my_white = self.wtm
        my = 'w' if my_white else 'b'
        opp = 'b' if my_white else 'w'
        myk = self._wk if my_white else self._bk
        in_check = myk >= 0 and self._attacked(myk, opp)
        stand_pat = self._evaluate_stm()

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
                try:
                    score = -self._quiescence(-beta, -alpha, qdepth + 1)
                except _AbortSearch:
                    b[fr] = piece      # restore before unwinding (_negamax-style)
                    b[to] = captured
                    self.wtm = not self.wtm
                    self._eval_sub(piece, fr, to, captured)
                    raise
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
            try:
                score = -self._quiescence(-beta, -alpha, qdepth + 1)
            except _AbortSearch:
                b[fr] = piece          # restore before unwinding (_negamax-style)
                b[to] = captured
                self.wtm = not self.wtm
                self._eval_sub(piece, fr, to, captured)
                raise
            b[fr] = piece
            b[to] = captured
            self.wtm = not self.wtm
            self._eval_sub(piece, fr, to, captured)
            if score > alpha:
                alpha = score
            if alpha >= beta:
                return alpha
        return alpha

    def _snapshot(self):
        """One-scan board summary used at every search node."""
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

    def _find_king(self, color):
        b = self.b
        target = WK if color == 'w' else BK
        for i in range(48):
            if b[i] == target:
                return i
        return -1

    def _attacked(self, sq, by):
        """Is square `sq` attacked by any piece of colour `by`?"""
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
        """Bitmask of own pieces standing between the king and an enemy slider"""
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
        """Legal moves as (from, to, captured, gives_check) tuples."""
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

    def _evaluate_stm(self):
        v = self._evaluate_white()
        return v if self.wtm else -v

    def _evaluate_white(self):
        """Static evaluation (white-positive), O(1) in the midgame."""
        b = self.b
        score = self._static
        w_king = self._wk
        b_king = self._bk
        w_nk = self.wnk
        b_nk = self.bnk
        w_pow = self.wpow
        b_pow = self.bpow
        nk = w_nk + b_nk
        if nk <= 12:
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
        # a mate just as easily as a material-down one.
        if w_king >= 0 and b_king >= 0:
            if b_nk >= 2 and (b_pow - w_pow >= 100
                              or self._edge_danger(w_king, self._press_w)):
                d = self._king_danger(w_king, 'b')
                score -= d if b_pow - w_pow >= 100 else int(d * self.DANGER_EVEN)
            if w_nk >= 2 and (w_pow - b_pow >= 100
                              or self._edge_danger(b_king, self._press_b)):
                d = self._king_danger(b_king, 'w')
                score += d if w_pow - b_pow >= 100 else int(d * self.DANGER_EVEN)
        self._wk = w_king
        self._bk = b_king
        return score

    def _drive_bonus(self, leader, lk, ek):
        """Positive score (for `leader`) pushing the enemy king to the edge"""
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

    def _edge_danger(self, ksq, press):
        """Cheap O(1) gate for `_king_danger`, which scans rays and flight squares."""
        if ksq < 0:
            return False
        kr, kc = divmod(ksq, BW)
        if 1 < kr < BH - 2 and 1 < kc < BW - 2:
            return False                     # safely in the middle
        return press >= self.DANGER_PRESS_MIN

    def _king_danger(self, ksq, opp):
      
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



