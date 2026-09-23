
import time
import random
from config import (
    EMPTY_SQUARE, WHITE_KING, BLACK_KING,
    BOARD_WIDTH, BOARD_HEIGHT,
)

MATE = 100000
INF = 10 ** 9
MAX_PLY = 64

VAL = {"P": 200, "N": 700, "B": 700, "R": 1000, "K": 10000}

# White-perspective piece-square tables. Black uses the vertical flip.
# Pawns cannot promote: last-rank pawns are dead and heavily penalized.
PST_P = (
    (-180, -180, -180, -180, -180, -180),
    (  -16,   -4,    4,    4,   -4,  -16),
    (   12,   24,   44,   44,   24,   12),
    (   20,   36,   56,   56,   36,   20),
    (   16,   28,   44,   44,   28,   16),
    (    6,   16,   24,   24,   16,    6),
    (    0,    0,    0,    0,    0,    0),
    (    0,    0,    0,    0,    0,    0),
)
PST_P_EG = (
    (-180, -180, -180, -180, -180, -180),
    (    0,    8,   16,   16,    8,    0),
    (   24,   36,   48,   48,   36,   24),
    (   32,   44,   56,   56,   44,   32),
    (   20,   32,   40,   40,   32,   20),
    (    8,   16,   24,   24,   16,    8),
    (    0,    0,    0,    0,    0,    0),
    (    0,    0,    0,    0,    0,    0),
)
PST_N = (
    (-64, -28, -20, -20, -28, -64),
    (-28,   0,  16,  16,   0, -28),
    (-20,  20,  36,  36,  20, -20),
    (-16,  24,  44,  44,  24, -16),
    (-16,  24,  44,  44,  24, -16),
    (-20,  20,  36,  36,  20, -20),
    (-28,   0,  16,  16,   0, -28),
    (-64, -28, -20, -20, -28, -64),
)
PST_B = (
    (-28, -16, -12, -12, -16, -28),
    (-16,   8,   4,   4,   8, -16),
    (-12,   4,  20,  20,   4, -12),
    (-12,  12,  24,  24,  12, -12),
    (-12,  12,  24,  24,  12, -12),
    (-12,   4,  20,  20,   4, -12),
    (-16,  12,   8,   8,  12, -16),
    (-28, -16, -12, -12, -16, -28),
)
PST_R = (
    ( 20,  28,  36,  36,  28,  20),
    ( 28,  36,  44,  44,  36,  28),
    (  4,  12,  20,  20,  12,   4),
    (  0,   8,  16,  16,   8,   0),
    (  0,   8,  16,  16,   8,   0),
    (  0,   8,  16,  16,   8,   0),
    (  4,  12,  20,  20,  12,   4),
    (  8,  16,  28,  20,  28,  16),
)
PST_K_MG = (
    (-100,  -80,  -80,  -80,  -80, -100),
    ( -80,  -60,  -60,  -60,  -60,  -80),
    ( -80,  -60,  -40,  -40,  -60,  -80),
    ( -80,  -60,  -40,  -40,  -60,  -80),
    ( -60,  -40,  -20,  -20,  -40,  -60),
    ( -40,  -20,    0,    0,  -20,  -40),
    (  24,   36,   16,   -8,   16,   36),
    (  44,   72,   56,   16,   56,   72),
)
PST_K_EG = (
    (-44, -20,   0,   0, -20, -44),
    (-20,  12,  32,  32,  12, -20),
    (  0,  32,  52,  52,  32,   0),
    (  8,  40,  64,  64,  40,   8),
    (  8,  40,  64,  64,  40,   8),
    (  0,  32,  52,  52,  32,   0),
    (-20,  12,  32,  32,  12, -20),
    (-44, -20,   0,   0, -20, -44),
)

N_DELTA = ((2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2))
B_DIR = ((1, 1), (1, -1), (-1, 1), (-1, -1))
R_DIR = ((1, 0), (-1, 0), (0, 1), (0, -1))
K_DELTA = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))

_PI = {"wP": 0, "wN": 1, "wB": 2, "wR": 3, "wK": 4,
       "bP": 5, "bN": 6, "bB": 7, "bR": 8, "bK": 9}
_rng = random.Random(20260913)
ZT = [[[_rng.getrandbits(64) for _ in range(10)] for _ in range(6)] for _ in range(8)]
ZS = _rng.getrandbits(64)

# Small opening preferences from the start (and the mirrored black replies).
# Used only when the suggested move is still legal.
_OPEN_W = ((7, 0, 5, 1), (6, 3, 5, 3), (6, 2, 5, 2), (6, 4, 5, 4), (7, 5, 5, 4))
_OPEN_B = ((0, 0, 2, 1), (1, 3, 2, 3), (1, 2, 2, 2), (1, 4, 2, 4), (0, 5, 2, 4))


def _hidx(m):
    return ((m.start_row * 6 + m.start_col) * 8 + m.end_row) * 6 + m.end_col


def _is_cap(m):
    return m.piece_captured != EMPTY_SQUARE


def _is_king_cap(m):
    return m.piece_captured == WHITE_KING or m.piece_captured == BLACK_KING


class B23CS1082:
    def __init__(self, engine):
        self.engine = engine
        self.nodes_expanded = 0
        self.depth = 6
        self.time_left = 59.4
        self.tt = {}
        self.hist = [0] * (8 * 6 * 8 * 6)
        self.killers = [[None, None] for _ in range(MAX_PLY + 4)]
        self.time_up = False
        self.search_start = 0.0
        self.allocated = 0.8
        self._moves_played = 0
        self.last_score = 0
        self.path_count = {}

    def get_best_move(self):
        try:
            return self._choose_move()
        except Exception:
            try:
                ms = self.engine.get_legal_moves()
                return ms[0] if ms else None
            except Exception:
                return None

    def evaluate_board(self, game_state=None):
        if game_state == "checkmate":
            return -MATE if self.engine.white_to_move else MATE
        if game_state == "stalemate":
            return 0
        return self._evaluate()

    def _choose_move(self):
        self.nodes_expanded = 0
        self.time_up = False
        self.search_start = time.perf_counter()

        legal = self.engine.get_legal_moves()
        if not legal:
            return None
        if len(legal) == 1:
            self._account_time()
            self._moves_played += 1
            return legal[0]

        for m in legal:
            if _is_king_cap(m):
                self._account_time()
                self._moves_played += 1
                return m

        self._set_budget(len(legal))
        self.path_count = {self._hash(): 1}
        best = self._opening(legal) or self._greedy(legal)
        prev_score = self.last_score

        # Iterative deepening: only keep a depth if the iteration finished.
        for d in range(1, 48):
            if d > 1:
                elapsed = time.perf_counter() - self.search_start
                if elapsed >= self.allocated * 0.45:
                    break
            score, move = self._root_search(legal, d, best, prev_score)
            if self.time_up:
                break
            if move is not None:
                best = move
                prev_score = score
                self.depth = d
            if abs(prev_score) > MATE - 400:
                break

        self.last_score = prev_score
        self._account_time()
        self._moves_played += 1
        if len(self.tt) > 250000:
            self.tt.clear()
        return best

    def _set_budget(self, nmoves):
        t = self.time_left
        remain = max(14, 52 - self._moves_played)
        alloc = t / remain * 0.90
        if t > 32:
            alloc = min(alloc, 1.30)
        elif t > 18:
            alloc = min(alloc, 1.00)
        elif t > 9:
            alloc = min(alloc, 0.55)
        elif t > 4:
            alloc = min(alloc, 0.28)
        else:
            alloc = min(alloc, max(0.035, t * 0.10))
        if abs(self.last_score) > 1800:
            alloc = min(alloc, 0.40)
        elif abs(self.last_score) > 900:
            alloc = min(alloc, 0.70)
        if self.engine.is_in_check():
            alloc = min(alloc * 1.25, t - 0.45 if t > 1.0 else alloc)
        if nmoves <= 3:
            alloc = min(alloc, 0.40)
        if t > 1.0:
            alloc = min(alloc, t - 0.50)
        self.allocated = max(0.025, alloc)

    def _account_time(self):
        self.time_left -= time.perf_counter() - self.search_start
        if self.time_left < 0.05:
            self.time_left = 0.05

    def _opening(self, legal):
        n = len(self.engine.move_log)
        if n > 5:
            return None
        prefs = _OPEN_W if self.engine.white_to_move else _OPEN_B
        for pr in prefs:
            for m in legal:
                if (m.start_row, m.start_col, m.end_row, m.end_col) == pr:
                    return m
        return None

    def _greedy(self, legal):
        best, bs = legal[0], -INF
        white = self.engine.white_to_move
        for m in legal:
            s = 0
            if _is_cap(m):
                if _is_king_cap(m):
                    return m
                s += VAL[m.piece_captured[1]] * 8 - VAL[m.piece_moved[1]]
            s += (m.start_row - m.end_row) if white else (m.end_row - m.start_row)
            if s > bs:
                bs, best = s, m
        return best

    def _root_search(self, moves, depth, prev_best, prev_score):
        ordered = list(moves)
        self._order(ordered, None, 0)
        if prev_best is not None:
            for i, m in enumerate(ordered):
                if m == prev_best:
                    ordered.insert(0, ordered.pop(i))
                    break

        window = 70 if depth >= 3 else INF

        def run(a, b):
            best_sc, best_m = -INF, None
            aa = a
            for i, m in enumerate(ordered):
                self.engine.make_move(m)
                try:
                    if _is_king_cap(m):
                        sc = MATE - 1
                    elif self.engine.get_repetition_count() >= 2:
                        sc = 0
                    elif i == 0:
                        sc = -self._search(depth - 1, -b, -aa, 1, True)
                    else:
                        sc = -self._search(depth - 1, -aa - 1, -aa, 1, True)
                        if not self.time_up and sc > aa and sc < b:
                            sc = -self._search(depth - 1, -b, -aa, 1, True)
                finally:
                    self.engine.undo_move()
                if self.time_up:
                    return best_sc, best_m
                if sc > best_sc:
                    best_sc, best_m = sc, m
                    if sc > aa:
                        aa = sc
                        if aa >= b:
                            break
            return best_sc, best_m

        lo, hi = prev_score - window, prev_score + window
        sc, mv = run(lo, hi)
        if self.time_up:
            return sc, mv
        if mv is not None and window != INF and (sc <= lo or sc >= hi):
            sc2, mv2 = run(-INF, INF)
            if not self.time_up and mv2 is not None:
                return sc2, mv2
        return sc, mv

    def _poll(self):
        if (self.nodes_expanded & 31) == 0:
            if time.perf_counter() - self.search_start >= self.allocated:
                self.time_up = True
                return True
        return False

    def _search(self, depth, alpha, beta, ply, can_null):
        self.nodes_expanded += 1
        if self._poll():
            return 0
        if ply >= MAX_PLY:
            return self._eval_stm()

        key = self._hash()
        pc = self.path_count.get(key, 0)
        if ply > 0 and pc:
            return 0
        self.path_count[key] = pc + 1
        try:
            return self._search_body(depth, alpha, beta, ply, can_null, key)
        finally:
            if pc:
                self.path_count[key] = pc
            else:
                self.path_count.pop(key, None)

    def _search_body(self, depth, alpha, beta, ply, can_null, key):
        in_check = self.engine.is_in_check()
        if in_check:
            depth += 1

        orig_alpha = alpha
        tt_move = None
        hit = self.tt.get(key)
        if hit is not None:
            td, ts, tf, tm = hit
            tt_move = tm
            if td >= depth and ply > 0:
                if tf == 0:
                    return ts
                if tf == 1 and ts >= beta:
                    return ts
                if tf == 2 and ts <= alpha:
                    return ts

        if depth <= 0:
            return self._quiesce(alpha, beta, ply, 6)

        if (can_null and depth >= 3 and not in_check
                and beta < MATE - 1000 and self._nm_ok()):
            self.engine.white_to_move = not self.engine.white_to_move
            R = 2 + (depth >= 6)
            try:
                sc = -self._search(depth - 1 - R, -beta, -beta + 1, ply + 1, False)
            finally:
                self.engine.white_to_move = not self.engine.white_to_move
            if self.time_up:
                return 0
            if sc >= beta:
                return sc

        moves = self.engine.get_legal_moves()
        if not moves:
            return (-MATE + ply) if in_check else 0

        self._order(moves, tt_move, ply)
        best = -INF
        best_tup = None

        for i, m in enumerate(moves):
            cap = _is_cap(m)
            red = 0
            if depth >= 3 and i >= 3 and not cap and not in_check:
                red = 1
                if i >= 8 and depth >= 5:
                    red = 2

            self.engine.make_move(m)
            try:
                if _is_king_cap(m):
                    sc = MATE - ply - 1
                else:
                    sc = -self._search(depth - 1 - red, -beta, -alpha, ply + 1, True)
                    if red and not self.time_up and sc > alpha:
                        sc = -self._search(depth - 1, -beta, -alpha, ply + 1, True)
            finally:
                self.engine.undo_move()

            if self.time_up:
                return 0
            if sc > best:
                best = sc
                best_tup = (m.start_row, m.start_col, m.end_row, m.end_col)
                if sc > alpha:
                    alpha = sc
                    if alpha >= beta:
                        if not cap:
                            k = self.killers[ply]
                            if k[0] is None or m != k[0]:
                                k[1] = k[0]
                                k[0] = m
                            hi = _hidx(m)
                            self.hist[hi] += depth * depth
                            if self.hist[hi] > 80000:
                                for j in range(len(self.hist)):
                                    self.hist[j] >>= 1
                        break

        if best <= orig_alpha:
            flag = 2
        elif best >= beta:
            flag = 1
        else:
            flag = 0
        self.tt[key] = (depth, best, flag, best_tup)
        return best

    def _quiesce(self, alpha, beta, ply, qdepth):
        self.nodes_expanded += 1
        if self._poll():
            return 0
        if ply >= MAX_PLY or qdepth <= 0:
            return self._eval_stm()

        in_check = self.engine.is_in_check()
        if in_check:
            moves = self.engine.get_legal_moves()
            if not moves:
                return -MATE + ply
            self._order(moves, None, ply)
            best = -INF
            for m in moves:
                self.engine.make_move(m)
                try:
                    sc = (MATE - ply - 1) if _is_king_cap(m) else -self._quiesce(
                        -beta, -alpha, ply + 1, qdepth - 1)
                finally:
                    self.engine.undo_move()
                if self.time_up:
                    return 0
                if sc > best:
                    best = sc
                if sc > alpha:
                    alpha = sc
                    if alpha >= beta:
                        return alpha
            return best

        stand = self._eval_stm()
        if stand >= beta:
            return stand
        if stand > alpha:
            alpha = stand

        caps = self._legal_captures()
        if not caps:
            return stand
        self._order(caps, None, ply)
        for m in caps:
            cv = VAL[m.piece_captured[1]]
            if m.piece_captured[1] != "K" and stand + cv + 90 < alpha:
                continue
            self.engine.make_move(m)
            try:
                sc = (MATE - ply - 1) if _is_king_cap(m) else -self._quiesce(
                    -beta, -alpha, ply + 1, qdepth - 1)
            finally:
                self.engine.undo_move()
            if self.time_up:
                return 0
            if sc > alpha:
                alpha = sc
                if alpha >= beta:
                    return alpha
        return alpha

    def _legal_captures(self):
        out = []
        for m in self.engine._get_all_possible_moves():
            if not _is_cap(m):
                continue
            self.engine.make_move(m)
            ok = not self.engine._is_king_in_check()
            self.engine.undo_move()
            if ok:
                out.append(m)
        return out

    def _nm_ok(self):
        t = "w" if self.engine.white_to_move else "b"
        b = self.engine.board
        for r in range(BOARD_HEIGHT):
            row = b[r]
            for c in range(BOARD_WIDTH):
                p = row[c]
                if p[0] == t and p[1] in "NBR":
                    return True
        return False

    def _order(self, moves, tt_tup, ply):
        moves.sort(key=lambda m: self._ms(m, tt_tup, ply), reverse=True)

    def _ms(self, m, tt_tup, ply):
        if tt_tup is not None and (m.start_row, m.start_col, m.end_row, m.end_col) == tt_tup:
            return 4000000
        if _is_cap(m):
            if _is_king_cap(m):
                return 3000000
            return 1000000 + 10 * VAL[m.piece_captured[1]] - VAL[m.piece_moved[1]]
        k = self.killers[ply]
        if k[0] is not None and m == k[0]:
            return 900000
        if k[1] is not None and m == k[1]:
            return 800000
        return self.hist[_hidx(m)]

    def _hash(self):
        h = 0
        b = self.engine.board
        for r in range(8):
            row = b[r]
            zr = ZT[r]
            for c in range(6):
                p = row[c]
                if p != EMPTY_SQUARE:
                    h ^= zr[c][_PI[p]]
        if self.engine.white_to_move:
            h ^= ZS
        return h

    def _eval_stm(self):
        e = self._evaluate()
        return e if self.engine.white_to_move else -e

    def _slide_mob(self, b, r, c, dirs, friendly):
        n = 0
        for dr, dc in dirs:
            rr, cc = r + dr, c + dc
            while 0 <= rr < 8 and 0 <= cc < 6:
                t = b[rr][cc]
                if t.startswith(friendly):
                    break
                n += 1
                if t != EMPTY_SQUARE:
                    break
                rr += dr
                cc += dc
        return n

    def _evaluate(self):
        """White-positive heuristic: material, PST, structure, king, mobility."""
        b = self.engine.board
        mg = 0
        eg = 0
        wpf = [0] * 6
        bpf = [0] * 6
        wn = bn = wb = bb = wr = br = 0
        wk = bk = None

        for r in range(8):
            row = b[r]
            for c in range(6):
                p = row[c]
                if p == EMPTY_SQUARE:
                    continue
                t = p[1]
                if p[0] == "w":
                    if t == "P":
                        wpf[c] += 1
                        mg += 200 + PST_P[r][c]   #  P=20 → internal ×10
                        eg += 200 + PST_P_EG[r][c]
                    elif t == "N":
                        wn += 1
                        bonus = PST_N[r][c]
                        mg += 700 + bonus           #  N=70 → internal ×10
                        eg += 700 + bonus
                        mob = 0
                        for dr, dc in N_DELTA:
                            rr, cc = r + dr, c + dc
                            if 0 <= rr < 8 and 0 <= cc < 6 and not b[rr][cc].startswith("w"):
                                mob += 1
                        mg += mob * 8
                        eg += mob * 6
                    elif t == "B":
                        wb += 1
                        bonus = PST_B[r][c]
                        mg += 700 + bonus           #  B=70 → internal ×10 (= N)
                        eg += 700 + bonus
                        mob = self._slide_mob(b, r, c, B_DIR, "w")
                        mg += mob * 6
                        eg += mob * 4
                    elif t == "R":
                        wr += 1
                        bonus = PST_R[r][c]
                        mg += 1000 + bonus          #  R=100 → internal ×10
                        eg += 1000 + bonus
                        mob = self._slide_mob(b, r, c, R_DIR, "w")
                        mg += mob * 4
                        eg += mob * 4
                    else:
                        wk = (r, c)
                        mg += PST_K_MG[r][c]
                        eg += PST_K_EG[r][c]
                else:
                    rr = 7 - r
                    if t == "P":
                        bpf[c] += 1
                        mg -= 200 + PST_P[rr][c]
                        eg -= 200 + PST_P_EG[rr][c]
                    elif t == "N":
                        bn += 1
                        bonus = PST_N[rr][c]
                        mg -= 700 + bonus
                        eg -= 700 + bonus
                        mob = 0
                        for dr, dc in N_DELTA:
                            r2, c2 = r + dr, c + dc
                            if 0 <= r2 < 8 and 0 <= c2 < 6 and not b[r2][c2].startswith("b"):
                                mob += 1
                        mg -= mob * 8
                        eg -= mob * 6
                    elif t == "B":
                        bb += 1
                        bonus = PST_B[rr][c]
                        mg -= 700 + bonus
                        eg -= 700 + bonus
                        mob = self._slide_mob(b, r, c, B_DIR, "b")
                        mg -= mob * 6
                        eg -= mob * 4
                    elif t == "R":
                        br += 1
                        bonus = PST_R[rr][c]
                        mg -= 1000 + bonus
                        eg -= 1000 + bonus
                        mob = self._slide_mob(b, r, c, R_DIR, "b")
                        mg -= mob * 4
                        eg -= mob * 4
                    else:
                        bk = (r, c)
                        mg -= PST_K_MG[rr][c]
                        eg -= PST_K_EG[rr][c]

        if wb >= 2:
            mg += 44
            eg += 56
        if bb >= 2:
            mg -= 44
            eg -= 56

        for c in range(6):
            if wpf[c] > 1:
                d = wpf[c] - 1
                mg -= 32 * d
                eg -= 44 * d
            if bpf[c] > 1:
                d = bpf[c] - 1
                mg += 32 * d
                eg += 44 * d
            w_iso = wpf[c] and (c == 0 or wpf[c - 1] == 0) and (c == 5 or wpf[c + 1] == 0)
            b_iso = bpf[c] and (c == 0 or bpf[c - 1] == 0) and (c == 5 or bpf[c + 1] == 0)
            if w_iso:
                mg -= 20
                eg -= 28
            if b_iso:
                mg += 20
                eg += 28

        for r in range(8):
            row = b[r]
            for c in range(6):
                p = row[c]
                if p == "wR":
                    if wpf[c] == 0:
                        mg += 36 if bpf[c] == 0 else 16
                        eg += 24 if bpf[c] == 0 else 8
                elif p == "bR":
                    if bpf[c] == 0:
                        mg -= 36 if wpf[c] == 0 else 16
                        eg -= 24 if wpf[c] == 0 else 8

        if wk:
            kr, kc = wk
            shield = 0
            for dc in (-1, 0, 1):
                cc = kc + dc
                if 0 <= cc < 6 and kr > 0 and b[kr - 1][cc] == "wP":
                    shield += 1
            mg += shield * 28
            for r in range(max(0, kr - 2), min(8, kr + 3)):
                for c in range(max(0, kc - 2), min(6, kc + 3)):
                    p = b[r][c]
                    if p[0] == "b" and p[1] != "K":
                        mg -= {"P": 6, "N": 22, "B": 18, "R": 30}[p[1]]
        if bk:
            kr, kc = bk
            shield = 0
            for dc in (-1, 0, 1):
                cc = kc + dc
                if 0 <= cc < 6 and kr < 7 and b[kr + 1][cc] == "bP":
                    shield += 1
            mg -= shield * 28
            for r in range(max(0, kr - 2), min(8, kr + 3)):
                for c in range(max(0, kc - 2), min(6, kc + 3)):
                    p = b[r][c]
                    if p[0] == "w" and p[1] != "K":
                        mg += {"P": 6, "N": 22, "B": 18, "R": 30}[p[1]]

        if wk and bk:
            eg -= (abs(wk[0] - 3) + abs(wk[1] - 2) + abs(wk[1] - 3) // 2) * 4
            eg += (abs(bk[0] - 4) + abs(bk[1] - 2) + abs(bk[1] - 3) // 2) * 4
            wmat = 200 * sum(wpf) + 700 * wn + 700 * wb + 1000 * wr
            bmat = 200 * sum(bpf) + 700 * bn + 700 * bb + 1000 * br
            md = wmat - bmat
            if md > 560:   # ~2.8 pawns advantage (pawn=200)
                er, ec = min(bk[0], 7 - bk[0]), min(bk[1], 5 - bk[1])
                mop = (3 - er) * 44 + (2 - ec) * 56
                mop += (14 - abs(wk[0] - bk[0]) - abs(wk[1] - bk[1])) * 16
                eg += mop
                mg += mop // 3
            elif md < -560:
                er, ec = min(wk[0], 7 - wk[0]), min(wk[1], 5 - wk[1])
                mop = (3 - er) * 44 + (2 - ec) * 56
                mop += (14 - abs(wk[0] - bk[0]) - abs(wk[1] - bk[1])) * 16
                eg -= mop
                mg -= mop // 3

        phase = wn + bn + wb + bb + 2 * (wr + br)
        if phase > 16:
            phase = 16
        score = (mg * phase + eg * (16 - phase)) // 16
        score += 16 if self.engine.white_to_move else -16  
        if self.engine.is_in_check():
            if self.engine.white_to_move:
                score -= 20   # current player (white) is in check = bad for white
            else:
                score += 20   # current player (black) is in check = good for white
        return score
