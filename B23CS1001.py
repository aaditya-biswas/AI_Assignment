"""
Single-file adversarial agent for Spartans Chess (6x8, no queens).

This file is the ONLY agent file we modify (the allocated template
ai_player.py).  For submission it is renamed to B23CS1001.py with the class
already named B23CS1001.  The tournament engine (board.py / config.py /
game_runner.py) is NEVER modified.

Speed layer
-----------
Searching on top of the engine's get_legal_moves() is slow (~15k nodes/s), so
this file contains a private, read-only COPY of the game rules operating on a
flat 48-square board.  It is validated against engine.get_legal_moves() by a
differential test (see run_parity.py).  Every rule matches the engine exactly:
pawns move one square, no castling/en-passant/promotion, generate-then-check.

Search design (alpha-beta negamax + iterative deepening):
  - capture-first (MVV-LVA) move ordering
  - horizon handling: light check extension + terminal detection whenever few
    pieces remain (never "win" into a stalemate, never miss a mate)
  - per-node deadline check -> never blows the 60 second game clock
  - endgame evaluation drives the king forward and the enemy king to the edge
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



class _AbortSearch(Exception):
    """Raised when the per-move time budget is exhausted mid-search."""


def _idx(r, c):
    return r * BW + c


class B23CS1001:
    """Alpha-beta agent (fast internal rules copy). Class name = roll number."""

    def __init__(self, engine, depth=6, max_depth=12):
        self.engine = engine
        self.nodes_expanded = 0
        self.depth = depth
        self.max_depth = max_depth
        self.time_used = 0.0
        self.test_budget = None
        self.max_ply = 60
        self._deadline = None
        self._search_nodes = 0
        self.b = None
        self.wtm = True
        self.tt = {}
        self.killers = {}
        self.history = {}
        self.qnodes = 0

    # ------------------------------------------------------------- setup
    def _load(self):
        b = []
        for row in self.engine.board:
            for p in row:
                b.append(_CODE[p])
        self.b = b
        self.wtm = self.engine.white_to_move

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
        moves_done = (len(engine.move_log) + 1) // 2
        budget = (60.0 - self.time_used) * 0.45 / max(1, 75 - moves_done)
        budget = max(0.25, min(budget, 2.5))

        # Mate-drive detection: we lead by >= a rook and the opponent has no
        # real pieces left.  When driving for mate we may use a little more of
        # the clock and must search deeper.
        myk, opk, in_check, few, drive = self._snapshot()
        if drive:
            budget = max(budget, min(2.0,
                         (60.0 - self.time_used) * 0.55 /
                         max(1, 75 - moves_done)))
        if self.test_budget is not None:
            budget = self.test_budget
        deadline = t0 + budget
        self._deadline = deadline
        self._search_nodes = 0

        best = None
        prev_score = None
        cap = self.max_depth + (10 if drive else 8 if few else 0)
        try:
            for d in range(1, cap + 1):
                if time.monotonic() >= deadline:
                    break
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
                        try:
                            score = -self._negamax(d - 1, -beta, -alpha, 1)
                        except _AbortSearch:
                            self.b[fr] = piece
                            self.b[to] = captured
                            self.wtm = not self.wtm
                            raise
                        self.b[fr] = piece
                        self.b[to] = captured
                        self.wtm = not self.wtm
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
        except _AbortSearch:
            pass
        finally:
            self._deadline = None
            self.time_used += time.monotonic() - t0
        return best if best is not None else root[0]

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
    def _lmr_reduction(self, depth, idx):
        if depth <= 2 or idx <= 2:
            return 0
        return max(0, 1 + int(math.log(depth) * math.log(idx + 1) / 2.0))

    def _null_move_allowed(self, in_check, depth):
        if in_check or depth < 2:
            return False
        for p in self.b:
            if p != E and _TYPE[p] != 5:
                return True
        return False

    def _negamax(self, depth, alpha, beta, ply):
        b = self.b
        self.nodes_expanded += 1
        self._search_nodes += 1

        if (self._search_nodes & 255) == 0 and self._deadline is not None \
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

        myk, opk, in_check, few, drive = self._snapshot()

        if depth <= 0:
            return self._quiescence(alpha, beta)
        if ply > self.max_ply:
            return self._evaluate_stm()

        stand_pat = None
        if not in_check:
            if depth <= 3:
                stand_pat = self._evaluate_stm()
                if stand_pat - 120 * depth >= beta:
                    return stand_pat
            if self._null_move_allowed(in_check, depth):
                self.wtm = not self.wtm
                score = -self._negamax(depth - 2, -beta, -beta + 1, ply + 1)
                self.wtm = not self.wtm
                if score >= beta:
                    return beta

        legal = self._legal_moves(myk, drive, opk)
        if not legal:
            return -(MATE - ply) if in_check else 0

        ordered = sorted(
            legal,
            key=lambda m: self._ordering_key(m, drive, depth, tt_move),
        )

        best_move = None
        alpha_orig = alpha
        best_score = -MATE
        for idx, (fr, to, captured, gives) in enumerate(ordered):
            piece = b[fr]
            b[to] = piece
            b[fr] = E
            self.wtm = not self.wtm
            try:
                if idx == 0:
                    score = -self._negamax(depth - 1, -beta, -alpha, ply + 1)
                else:
                    search_depth = depth - 1
                    if depth >= 3 and captured == E and not gives \
                            and not in_check:
                        search_depth -= self._lmr_reduction(depth, idx)
                    if search_depth < 0:
                        search_depth = 0
                    score = -self._negamax(search_depth, -alpha - 1, -alpha,
                                           ply + 1)
                    if score > alpha and score < beta:
                        # PVS re-search with the full window.
                        score = -self._negamax(depth - 1, -beta, -alpha,
                                               ply + 1)
            except _AbortSearch:
                b[fr] = piece
                b[to] = captured
                self.wtm = not self.wtm
                raise
            b[fr] = piece
            b[to] = captured
            self.wtm = not self.wtm
            if score > best_score:
                best_score = score
                best_move = (fr, to)
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break

        if best_move is not None:
            self._record_killer(depth, best_move)
            self.history[best_move] = self.history.get(best_move, 0) \
                + max(1, depth * depth)
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
        key = 0
        if tt_move is not None and (fr, to) == tt_move:
            key -= 10**9
        if drive and gives:
            key -= 300000
        if captured:
            key -= 100000 + _PTVAL[_TYPE[captured]] * 100 \
                - _PTVAL[_TYPE[self.b[fr]]]
        killers = self.killers.get(depth, ())
        if (fr, to) in killers:
            key -= 50000
        key -= self.history.get((fr, to), 0)
        return key

    def _record_killer(self, depth, move):
        if move is None:
            return
        fr, to = move
        lst = self.killers.setdefault(depth, [])
        if (fr, to) in lst:
            return
        lst.insert(0, (fr, to))
        if len(lst) > 2:
            lst.pop()

    def _capture_moves(self, ksq, opp):
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
        legal = []
        for fr, to, captured in found:
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
        # One evaluation scan also yields both king squares (via self._wk/_bk).
        stand_pat = self._evaluate_stm()
        myk = self._wk if my_white else self._bk
        in_check = myk >= 0 and self._attacked(myk, opp)

        if in_check:
            legal = self._legal_moves(myk)
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
                score = -self._quiescence(-beta, -alpha, qdepth + 1)
                b[fr] = piece
                b[to] = captured
                self.wtm = not self.wtm
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
            score = -self._quiescence(-beta, -alpha, qdepth + 1)
            b[fr] = piece
            b[to] = captured
            self.wtm = not self.wtm
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
        """Own pieces standing between the king and an enemy slider on the
        same line (only these quiet moves can ever expose the king)."""
        b = self.b
        pinned = set()
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
                            pinned.add(first)
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
                            pinned.add(first)
                        break
        return pinned

    def _legal_moves(self, ksq, drive=False, opk=-1):
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
        must_test = self._attacked(ksq, opp) or (drive and opk >= 0)
        pinned = set() if must_test else self._pinned_squares(ksq, opp)
        for fr, to in pseudo:
            if not (must_test or fr == ksq or fr in pinned):
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
        """Fast single-pass static evaluation (white-positive).

        Material + piece-square tables dominate (they mirror the tournament
        scoring), with small centralisation, passed-pawn and king-safety terms.
        Everything is plain ints computed in one 48-square scan (no dict of
        dicts), which makes it cheap enough to call at every leaf.
        """
        b = self.b
        score = 0
        w_king = b_king = -1
        w_pow = b_pow = 0
        w_nk = b_nk = 0
        w_pawns = []
        b_pawns = []
        for i in range(48):
            p = b[i]
            if p == E:
                continue
            if p == WK:
                w_king = i
                continue
            if p == BK:
                b_king = i
                continue
            score += _STATIC[p][i]
            if p < 6:
                w_pow += _VAL[p]
                w_nk += 1
                if p == WP:
                    w_pawns.append(i)
            else:
                b_pow += _VAL[p]
                b_nk += 1
                if p == BP:
                    b_pawns.append(i)

        if w_nk + b_nk <= 12:
            for i in w_pawns:
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
            for i in b_pawns:
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
            if w_nk + b_nk <= 6:
                wr = w_king // BW
                wc = w_king - wr * BW
                br = b_king // BW
                bc = b_king - br * BW
                score += KING_PST[wr][wc]
                score -= KING_PST[BH - 1 - br][bc]
            else:
                score += self._king_shield(w_king, 'w')
                score -= self._king_shield(b_king, 'b')

        if w_pow - b_pow >= 100 and b_nk <= 1 and w_king >= 0 and b_king >= 0:
            score += self._drive_bonus('w', w_king, b_king)
        elif b_pow - w_pow >= 100 and w_nk <= 1 and w_king >= 0 \
                and b_king >= 0:
            score -= self._drive_bonus('b', b_king, w_king)
        # Expose the king squares so quiescence does not need a second scan.
        self._wk = w_king
        self._bk = b_king
        return score

    def _king_shield(self, ksq, col):
        """Small bonus for friendly pawns directly in front of the king."""
        b = self.b
        r = ksq // BW
        c = ksq - r * BW
        rr = r - 1 if col == 'w' else r + 1
        if rr < 0 or rr >= BH:
            return 0
        base = rr * BW
        pawn = WP if col == 'w' else BP
        s = 0
        if c > 0 and b[base + c - 1] == pawn:
            s += 6
        if b[base + c] == pawn:
            s += 6
        if c + 1 < BW and b[base + c + 1] == pawn:
            s += 6
        return s

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

    def evaluate_board(self, game_state="ongoing"):
        """Public evaluation hook used by the tournament runner."""
        if game_state == "checkmate":
            return MATE if self.engine.white_to_move else -MATE
        if game_state == "stalemate":
            return 0
        return self._evaluate_white()



