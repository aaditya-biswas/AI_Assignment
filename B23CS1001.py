"""
Single-file adversarial agent for Spartans Chess (6x8, no queens).

"""
import math
import time

from config import (
    BOARD_WIDTH as BW, BOARD_HEIGHT as BH,
    KING_PST_LATE_GAME,
    PAWN_PST, KNIGHT_PST, BISHOP_PST, ROOK_PST,
)


MATE = 100000            # scores for mates exceed any material total
FEW_PIECE_LIMIT = 9      # at/below this many pieces: cheap terminal check
KING_PST = KING_PST_LATE_GAME

TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2

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
# Distances 1-3 are weighted by closeness (3/2/1).  Chebyshev distance is blind
# to long-range attackers though - a rook checking down an open file or a bishop
# on a diagonal is a real +2-check threat from any distance - so _PROX_FAR gives
# those squares a flat weight out to _PROX_RANGE.  Measured against real games
# the opponent scored 60-116 points from checks, so this reach matters.
_PROX_RANGE = 3                              # only pieces this close matter
_PROX_FAR = 0                                # flat weight out to _PROX_RANGE


def _build_prox(prox_range=_PROX_RANGE, far=_PROX_FAR):
    """King tropism table: weight per (king square, enemy square) pair."""
    table = []
    for _k in range(48):
        _kr, _kc = divmod(_k, BW)
        _row = []
        for _s in range(48):
            _sr, _sc = divmod(_s, BW)
            _d = max(abs(_sr - _kr), abs(_sc - _kc))
            if _d == 0 or _d > prox_range:
                _row.append(0)
            elif _d <= 3:
                _row.append(4 - _d)
            else:
                _row.append(far)
        table.append(tuple(_row))
    return table


_PROX = _build_prox()

# Fully precomputed static score per (piece code, square): signed material +
# piece-square table + centralisation folded into a single table lookup, so
# the hot evaluation loop does one indexing per occupied square.
_STATIC = [None] * 11
_PST_OF = {1: PAWN_PST, 2: KNIGHT_PST, 3: BISHOP_PST, 4: ROOK_PST}
# The official scorecard pays ONLY material (and checks), so the search must not
# trade a 20-point pawn for a positional nicety: _POS_SCALE shrinks the
# positional part (piece-square table + centralisation) while the material part
# stays exactly at the tournament values.  Measured over a fixed 48-position
# suite, 1.0 left 6.7 points of material hanging per position, 0.5 leaves 2.5,
# and 0.25 is worse again (4.6) - so 0.5 is what this variant wants.
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
_ABORT_MASK = 15           # test the hard deadline every 16 nodes.  The drift a
                           # move overshoots by is what eats the clock reserve
                           # over a long game, and it scales with NPS: at 255
                           # nodes it was ~2 s per game even under CPU load,
                           # against ~0.6 s here.  One time.monotonic() per 16
                           # nodes is well under 0.1% of the run time.

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
    TIME_FRACTION = 0.90           # share of the clock the game may consume
    MIN_MOVE_SECONDS = 0.15        # never move instantly
    HARD_FACTOR = 2.5              # hard bound = soft * this
    MAX_SHARE = 0.25               # one move never takes >25% of the clock
    MAX_SHARE_CRIT = 0.35          # ... nor >35% even in a critical spot
    # The reserve is deliberately generous and it is a *hard pool*: `_plan_time`
    # never plans to spend past GAME_SECONDS - RESERVE_SECONDS, and it shares
    # what is left of that pool over the moves still expected, so a long game
    # glides into small searches instead of instant ones.  The runner times the
    # WHOLE get_best_move() call and a time-out loses the game outright (600
    # points), so the reserve has to cover the abort granularity, per-move
    # bookkeeping and a loaded machine.  `test_budget.py --stress` plays whole
    # games on the real clock and fails if any move leaves less than 3.5 s.
    RESERVE_SECONDS = 5.0          # hard pool: never spent (timeout = loss)
    # Even if that pool is ever exhausted, the search still gets TINY_SLICE per
    # move - enough for depth 1-2 instead of a move in generation order - and it
    # may only eat into the reserve down to MIN_RESERVE.  The gap between the two
    # constants is what covers the abort granularity and per-move bookkeeping.
    TINY_SLICE = 0.02              # seconds: the last-resort move allowance
    MIN_RESERVE = 4.5              # absolute floor: usage <= 60 - MIN_RESERVE
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
    # Check defence.  Pricing checks inside the search was implemented and
    # REJECTED by measurement: negamax makes the term symmetric, so it also pays
    # US for giving check, and that trades material for 2-point checks -
    # interleaved games at 2-4 points lost with material -160 to -400 (mated in
    # 62-77 plies), against -120/+100 with it disabled.  A bounded asymmetric
    # root tax on the checking replies we leave (ROOT_CHECK_W = 4) was also
    # measured and rejected: it does kill the check column (mean check diff
    # -81 -> -1.5 over 8 interleaved games) but it pays for that by ducking -
    # material +97.5 -> -27.5, official +16.5 -> -7.0, i.e. a net loss.
    # What is left is the one trade that costs nothing: exchanging off the
    # opponent's long-range pieces when we are ahead on points, since an
    # exchange is material neutral and each rook/bishop removed is a check
    # source (and a mate threat) removed with it.
    ROOT_CHECK_W = 0               # score tax per checking reply left available
    ROOT_CHECK_CAP = 4             # ... capped at this many replies
    ADJ_TRADE_W = 0                # bonus for an exchange while ahead on points
    # --- adjudication scorecard (the tournament's points table) ------------
    # The official scoring is captures at PIECE_VALUES (P20 N70 B70 R100) plus
    # 2 points for every check GIVEN, a mate is 600 and overrides that table,
    # and a time-out is a loss.  run_games_fast.py stops a game that is still
    # running after ADJ_CAP_PLY plies and adjudicates it on the table, which is
    # how most games against a material-dominant opponent end, so the agent
    # keeps the exact tally and plays to the score late in the game.
    ADJ_CAP_PLY = 150              # ply at which the harness adjudicates
    # A check is worth only +2 points while a pawn is 20 and a piece 70-100, so
    # biasing the search towards checks trades material for small change: paired
    # games with this at 2 averaged ~80 points of material per game worse than
    # with it at 0.  The tally and the lead-protection below stay (they are about
    # not throwing away a lead), but the check bonus is off by default.
    ADJ_CHECK_PTS = 0              # root bonus per check given (official: 2)
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
        # Cross-move search history driving the dynamic time control.
        self._last_stability = 0      # consecutive moves with an unchanged best
        self._last_drop = 0           # score drop over the previous move
        # King-pressure state (check avoidance), maintained incrementally.
        self._press_w = 0             # enemy pressure around the white king
        self._press_b = 0             # enemy pressure around the black king
        # Adjudication tally (official points table: captures + 2 per check).
        self._adj_ply = 0             # plies already banked
        self._adj_caps = [0, 0]       # [white, black] captured-piece points
        self._adj_chk = [0, 0]        # [white, black] check points (2 each)
        self._adj_mode = 0            # 0 neutral, +1 protect a lead, -1 chase
        self._root_ck = {}            # (from, to) -> does this root move check?
        self._root_threat = {}        # (from, to) -> checking replies it allows

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
        # Official points table (captures + checks) and the late-game mode that
        # follows from it.
        self._load_adj()

    # -------------------------------------------------- adjudication table
    def _load_adj(self):
        """Keep the official points table exact (captures + 2 per check given).

        A game the harness still sees running after ADJ_CAP_PLY plies is decided
        on this table, so the agent has to know the score it is playing for.
        Only the plies not banked yet are read - the opponent's move, plus our
        own move banked when we picked it - so the cost is a couple of
        dictionary lookups per move: exact, and nothing that touches the clock.
        """
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
            # Only the last ply can still be tested for a delivered check: if we
            # are in check right now, it was that move that gave it.
            if i == n - 1 and self.engine.is_in_check():
                self._adj_chk[i & 1] += 2
        self._adj_ply = n

        # Play-to-the-score mode: late in a game that will be adjudicated, our
        # own points are what count, so a lead is worth protecting and a
        # deficit is worth chasing (a check is +2, a mate is 600).
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
        """Does this move give check?  One attack test, root use only.

        The mover's colour is read from the engine, not from `self.wtm`: the
        search flips `wtm` while it runs and only `_load` restores it.
        """
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

    def _threat_checks(self, fr, to):
        """How many of the opponent's replies would give check after this move?

        Root-only probe, run once per move (never per node) and used as a bounded
        score tax on moves that leave our king facing the endless stream of +2
        checks.  It enumerates the opponent's legal replies in the resulting
        position and tests each for attacking our king.
        """
        b = self.b
        piece = b[fr]
        cap = b[to]
        my_white = self.wtm
        opp = 'b' if my_white else 'w'
        myk = self._wk if my_white else self._bk
        b[to] = piece
        b[fr] = E
        if piece == WK or piece == BK:       # our king may have moved itself
            myk = to
        self.wtm = not my_white
        their_king = self._bk if self.wtm else self._wk
        replies = self._legal_moves(their_king, False, -1, None)
        count = 0
        for f2, t2, cap2, _gives in replies:
            if cap2 == WK or cap2 == BK:
                continue                     # a king is never captured
            p2 = b[f2]
            b[t2] = p2
            b[f2] = E
            if myk >= 0 and self._attacked(myk, opp):
                count += 1
            b[f2] = p2
            b[t2] = cap2
        self.wtm = not self.wtm
        b[fr] = piece
        b[to] = cap
        return count

    def _root_hang(self, fr, to):
        """Value of the piece this move leaves attacked AND undefended.

        Root-only probe on the internal board (a few attack tests per root
        move, once per move rather than per node), used as a bounded score tax:
        the eval may still prefer a positional nicety, but not at the price of
        handing over material.  Kings are excluded (they cannot be captured) and
        the tax is capped, so a deliberate sacrifice or a mate stays available.
        """
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
        """Bank OUR move into the tally (its captures, and a check if any).

        Called on every return path of get_best_move, so the next move has only
        the opponent's reply left to read.  Everything read here comes from the
        engine and from the check map built *before* the search started, never
        from search-mutated state.
        """
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
            # A captured piece stops exerting its own pressure on the ENEMY
            # king (the king squares have not moved yet, so this is exact).
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
            # The mover carries its king pressure along with it.  Most moves are
            # far from the enemy king, where both lookups are 0 and the whole
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
        # Each square is a 0..10 code, so bytes() is a very fast exact key.
        return (self.wtm, bytes(self.b))

    # ------------------------------------------------------------ public API
    def get_best_move(self):
        engine = self.engine
        # The clock starts on the FIRST line: the runner times this whole call,
        # so _load(), root-move generation and per-move overheads have to be
        # counted here too, or the reserve silently leaks a little each move.
        t0 = time.monotonic()
        self._load()
        root = engine.get_legal_moves()
        if not root:
            self.time_used += time.monotonic() - t0
            return None

        # A forced move needs no search at all.
        if len(root) == 1:
            # It still has to be banked: a forced move can give check (+2).
            self._adj_bank(root[0], self._bk if self.wtm else self._wk)
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

        # Which root moves give check?  The adjudication table pays 2 points per
        # check GIVEN, so ADJ_CHECK_PTS is added to that move's root score below.
        # Computed once per move (never per iteration) and never fed into the
        # shared eval, so the transposition table stays consistent.
        self._root_ck = {}
        self._root_threat = {}
        if opk >= 0:
            for mv in root:
                key = (_idx(mv.start_row, mv.start_col),
                       _idx(mv.end_row, mv.end_col))
                self._root_ck[key] = self._gives_check(mv, opk)
                if self.ROOT_CHECK_W:
                    self._root_threat[key] = self._threat_checks(*key)

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
                        if self.ADJ_CHECK_PTS and self._root_ck.get((fr, to)):
                            score += self.ADJ_CHECK_PTS    # this move banks a check
                        if self.ROOT_CHECK_W:
                            # Bounded tax for leaving checking replies available:
                            # the opponent is paid +2 for every check it lands.
                            th = self._root_threat.get((fr, to), 0)
                            if th > self.ROOT_CHECK_CAP:
                                th = self.ROOT_CHECK_CAP
                            score -= th * self.ROOT_CHECK_W
                        if self.ADJ_TRADE_W and self._adj_mode > 0 \
                                and captured != E and _TYPE[captured] in (3, 4):
                            # Ahead on points, late: an exchange costs no
                            # material and removes a rook or bishop, i.e. one of
                            # the pieces that farm +2 checks and mate threats.
                            score += self.ADJ_TRADE_W
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
        if best is None:
            best = root[0]                 # aborted before any move was scored
        # Bank the move we are about to play into the adjudication tally (its
        # captures, and its +2 if it gives check), so the next move only has the
        # opponent's reply left to read.
        self._adj_bank(best, opk)
        return best

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
        # Hard pool: the reserve below is never spent, whatever the multipliers
        # say.  Without this the MAX_SHARE_CRIT crawl spends 35% of whatever is
        # left on every single move, and no per-move guard can bound that sum -
        # real games were measured at 59.6 s and 59.1 s of the 60 s clock.
        room = max(0.0, remaining - self.RESERVE_SECONDS)
        # The floor must scale with the clock, otherwise a fixed minimum would
        # spend more than half of what is left in a near-empty clock.  It is
        # capped by the pool, so it can never touch the reserve.
        floor = min(self.MIN_MOVE_SECONDS, remaining * 0.1, room)
        # ... but a move is never free: once the pool is gone the search still
        # gets a tiny slice, which is enough for depth 1-2 and far better than
        # returning a move in generation order.  It may only eat into the
        # reserve down to MIN_RESERVE, which is what covers the abort drift.
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
        """Root move ordering: mate-drive first, then the adjudication policy.

        Reordering only matters for moves the search scores equally (a root move
        is kept only on a strict improvement), so this can never give away a
        tactic or a mate - it just decides which of two equal moves to prefer.
        """
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
                # Protecting a lead: demote a piece left hanging, because that is
                # exactly how the opponent takes the points back before the cap.
                if self._root_hang(fr, to) >= self.ADJ_SAFE_PTS:
                    key += 30000 + _VAL[piece]
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

        if (self.nodes_expanded & _ABORT_MASK) == 0 and self._deadline is not None \
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
                try:
                    score = -ng(depth - 2, -beta, -beta + 1, ply + 1)
                except _AbortSearch:
                    # Keep make/unmake symmetric on every abort path: an
                    # unfinished null move must still hand `wtm` back, or the
                    # caller unwinds with the board and the side to move out of
                    # step (the qsearch guard and the node-count abort can both
                    # stop the search in here).
                    self.wtm = not self.wtm
                    raise
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
        # Hard-deadline guard for the forcing search: _negamax only tests the
        # deadline every 255 nodes, so a capture-rich tree at QMAX can overshoot
        # inside a single node.  The abort is re-raised through the unmake
        # blocks below, exactly like _negamax/root do.
        if self._deadline is not None and (self.qnodes & _ABORT_MASK) == 0 \
                and time.monotonic() >= self._deadline:
            raise _AbortSearch
        b = self.b
        my_white = self.wtm
        my = 'w' if my_white else 'b'
        opp = 'b' if my_white else 'w'
        # King squares are tracked incrementally (self._wk / self._bk).
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



