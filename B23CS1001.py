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

    # ------------------------------------------------------------- setup
    def _load(self):
        b = []
        for row in self.engine.board:
            b.extend(row)
        self.b = b
        self.wtm = self.engine.white_to_move

    def _position_key(self):
        return (tuple(self.b), self.wtm)

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
        prev_score = 0
        cap = self.max_depth + (10 if drive else 8 if few else 0)
        try:
            for d in range(1, cap + 1):
                if time.monotonic() >= deadline:
                    break
                alpha, beta = -MATE * 2, MATE * 2
                if d > 1:
                    alpha = max(-MATE, prev_score - 80)
                    beta = min(MATE, prev_score + 80)
                cur = None
                for mv in sorted(root, key=lambda m: self._root_key(m, drive,
                                                                    myk, opk)):
                    fr = _idx(mv.start_row, mv.start_col)
                    to = _idx(mv.end_row, mv.end_col)
                    piece = self.b[fr]
                    captured = self.b[to]
                    self.b[to] = piece
                    self.b[fr] = ES
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
                    if score > alpha:
                        alpha = score
                        cur = mv
                    if alpha >= beta:
                        break
                if cur is not None:
                    best = cur
                if best is not None:
                    prev_score = self._evaluate_stm()
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
            self.b[fr] = ES
            nk = to if to == opk else opk
            if nk >= 0 and self._attacked(nk, my):
                key -= 200000
            self.b[fr] = piece
            self.b[to] = captured
        cap = mv.piece_captured
        if cap != ES:
            key -= 100000 + abs(PIECE_VALUES.get(cap, 0))
        return key

    # ---------------------------------------------------------------- search
    def _negamax(self, depth, alpha, beta, ply):
        b = self.b
        self.nodes_expanded += 1
        self._search_nodes += 1

        if (self._search_nodes & 255) == 0 and self._deadline is not None \
            and time.monotonic() >= self._deadline:
            raise _AbortSearch()

        key = self._position_key()
        entry = self.tt.get(key)

        if entry is not None and entry.depth >= depth:
            if entry.flag == TT_EXACT:
                return entry.value
            if entry.flag == TT_LOWER:
                alpha = max(alpha, entry.value)
            elif entry.flag == TT_UPPER:
                beta = min(beta, entry.value)
            if alpha >= beta:
                return entry.value

        myk, opk, in_check, few, drive = self._snapshot()

        if depth <= 0 and not in_check and not few:
            return self._evaluate_stm()
        if ply > self.max_ply:
            return self._evaluate_stm()

        legal = self._legal_moves(myk, drive, opk)
        if not legal:
            return -(MATE - ply) if in_check else 0
        if depth <= 0 and few and not in_check:
            return self._evaluate_stm()

        tt_move = None if entry is None else entry.best_move
        ordered = sorted(
            legal,
            key=lambda m: self._move_key(m, drive)
            if tt_move is None or (m[0], m[1]) != tt_move
            else -10**9,
        )

        best_move = None
        alpha_orig = alpha
        best_score = -MATE
        for fr, to, captured, gives in ordered:
            piece = b[fr]
            b[to] = piece
            b[fr] = ES
            self.wtm = not self.wtm
            try:
                score = -self._negamax(max(depth - 1, 0), -beta, -alpha, ply + 1)
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

        if best_score <= alpha_orig:
            flag = TT_UPPER
        elif best_score >= beta:
            flag = TT_LOWER
        else:
            flag = TT_EXACT
        self.tt[key] = TTEntry(depth, flag, best_score, best_move)
        return best_score

    def _ordering_key(self, move, drive, depth):
        fr, to, captured, gives = move
        key = 0
        if drive and gives:
            key -= 300000
        if captured != ES:
            victim = PT_VAL.get(captured[1], 0)
            key -= 100000 + victim * 100
        killers = self.killers.get(depth, ())
        if (fr, to) in killers:
            key -= 50000
        key -= self.history.get((fr, to), 0)
        return key

    def _record_killer(self, depth, move):
        lst = self.killers.setdefault(depth, [])
        if move in lst:
            return
        lst.insert(0, move)
        if len(lst) > 2:
            lst.pop()

    def _move_key(self, move, drive):
        fr, to, captured, gives = move
        if drive and gives:
            return -300000
        if captured != ES:
            victim = PT_VAL.get(captured[1], 0)
            return -100000 - victim * 100
        return 0

    def _quiescence(self, alpha, beta):
        """Search only forcing moves: captures and checks."""
        stand_pat = self._evaluate_stm()
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat

        myk, opk, in_check, few, drive = self._snapshot()
        legal = self._legal_moves(myk, drive, opk)
        if not legal:
            return -(MATE - 1) if in_check else 0

        captures = []
        for mv in legal:
            fr, to, captured, gives = mv
            if captured != ES or gives:
                captures.append(mv)
        if not captures:
            return stand_pat

        captures.sort(key=lambda m: self._move_key(m, drive), reverse=True)
        for fr, to, captured, gives in captures:
            piece = self.b[fr]
            self.b[to] = piece
            self.b[fr] = ES
            self.wtm = not self.wtm
            score = -self._quiescence(-beta, -alpha)
            self.b[fr] = piece
            self.b[to] = captured
            self.wtm = not self.wtm
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break
        return alpha

    # ------------------------------------------------------------- snapshot
    def _snapshot(self):
        """One-scan board summary used at every search node.

        Returns (my_king, opp_king, in_check, few_pieces, mate_drive).
        mate_drive is true when the side to move is up by >= a rook against an
        opponent that has at most one non-king piece left.
        """
        my = 'w' if self.wtm else 'b'
        opp = 'b' if self.wtm else 'w'
        b = self.b
        myk = opk = -1
        tot = 0
        pow_my = pow_opp = 0
        opp_nk = 0
        for i in range(48):
            p = b[i]
            if p == ES:
                continue
            tot += 1
            color, ptype = p[0], p[1]
            if ptype == 'K':
                if color == my:
                    myk = i
                else:
                    opk = i
                continue
            v = abs(PIECE_VALUES.get(p, 0))
            if color == my:
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
        target = color + 'K'
        for i in range(48):
            if b[i] == target:
                return i
        return -1

    def _attacked(self, sq, by):
        """Is square `sq` attacked by any piece of colour `by`?"""
        b = self.b
        r, c = divmod(sq, BW)
        if by == 'w':
            pr = r + 1
            for dc in (-1, 1):
                if pr < BH:
                    pc = c + dc
                    if 0 <= pc < BW and b[_idx(pr, pc)] == 'wP':
                        return True
        else:
            pr = r - 1
            for dc in (-1, 1):
                if pr >= 0:
                    pc = c + dc
                    if 0 <= pc < BW and b[_idx(pr, pc)] == 'bP':
                        return True
        for dr, dc in zip(_KNIGHT_DR, _KNIGHT_DC):
            nr, nc = r + dr, c + dc
            if 0 <= nr < BH and 0 <= nc < BW and b[_idx(nr, nc)] == by + 'N':
                return True
        for dr, dc in zip(_KING_DR, _KING_DC):
            nr, nc = r + dr, c + dc
            if 0 <= nr < BH and 0 <= nc < BW and b[_idx(nr, nc)] == by + 'K':
                return True
        for dr, dc in zip(_BISHOP_DR, _BISHOP_DC):
            nr, nc = r + dr, c + dc
            while 0 <= nr < BH and 0 <= nc < BW:
                p = b[_idx(nr, nc)]
                if p != ES:
                    if p == by + 'B':
                        return True
                    break
                nr += dr
                nc += dc
        for dr, dc in zip(_ROOK_DR, _ROOK_DC):
            nr, nc = r + dr, c + dc
            while 0 <= nr < BH and 0 <= nc < BW:
                p = b[_idx(nr, nc)]
                if p != ES:
                    if p == by + 'R':
                        return True
                    break
                nr += dr
                nc += dc
        return False

    def _legal_moves(self, ksq, drive=False, opk=-1):
        """Legal moves as (from, to, captured, gives_check) tuples.

        Rules mirror board.py exactly (generate-then-validate); the king of the
        side that just moved must not be left in check.  When `drive` is set we
        additionally record whether each move gives check (mate-drive needs to
        order forcing moves first).
        """
        b = self.b
        my = 'w' if self.wtm else 'b'
        opp = 'b' if self.wtm else 'w'
        pseudo = []
        for i in range(48):
            p = b[i]
            if p == ES or p[0] != my:
                continue
            ptype = p[1]
            r, c = divmod(i, BW)
            if ptype == 'P':
                dr = -1 if my == 'w' else 1
                nr = r + dr
                if 0 <= nr < BH and b[_idx(nr, c)] == ES:
                    pseudo.append((i, _idx(nr, c)))
                for dc in (-1, 1):
                    nc = c + dc
                    if 0 <= nr < BH and 0 <= nc < BW and \
                            b[_idx(nr, nc)] != ES and b[_idx(nr, nc)][0] == opp:
                        pseudo.append((i, _idx(nr, nc)))
            elif ptype == 'N':
                for dr, dc in zip(_KNIGHT_DR, _KNIGHT_DC):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < BH and 0 <= nc < BW:
                        t = b[_idx(nr, nc)]
                        if t == ES or t[0] == opp:
                            pseudo.append((i, _idx(nr, nc)))
            elif ptype == 'B':
                for dr, dc in zip(_BISHOP_DR, _BISHOP_DC):
                    nr, nc = r + dr, c + dc
                    while 0 <= nr < BH and 0 <= nc < BW:
                        t = b[_idx(nr, nc)]
                        if t == ES:
                            pseudo.append((i, _idx(nr, nc)))
                        else:
                            if t[0] == opp:
                                pseudo.append((i, _idx(nr, nc)))
                            break
                        nr += dr
                        nc += dc
            elif ptype == 'R':
                for dr, dc in zip(_ROOK_DR, _ROOK_DC):
                    nr, nc = r + dr, c + dc
                    while 0 <= nr < BH and 0 <= nc < BW:
                        t = b[_idx(nr, nc)]
                        if t == ES:
                            pseudo.append((i, _idx(nr, nc)))
                        else:
                            if t[0] == opp:
                                pseudo.append((i, _idx(nr, nc)))
                            break
                        nr += dr
                        nc += dc
            elif ptype == 'K':
                for dr, dc in zip(_KING_DR, _KING_DC):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < BH and 0 <= nc < BW:
                        t = b[_idx(nr, nc)]
                        if t == ES or t[0] == opp:
                            pseudo.append((i, _idx(nr, nc)))

        legal = []
        for fr, to in pseudo:
            captured = b[to]
            piece = b[fr]
            b[to] = piece
            b[fr] = ES
            king = to if fr == ksq else ksq
            gives = False
            if drive and opk >= 0:
                nk = to if to == opk else opk
                gives = nk >= 0 and self._attacked(nk, my)
            if king >= 0 and not self._attacked(king, opp):
                legal.append((fr, to, captured, gives))
            b[fr] = piece
            b[to] = captured
        return legal

    # ----------------------------------------------------------- counters
    def _piece_count(self):
        n = 0
        for p in self.b:
            if p != ES:
                n += 1
        return n

    def _non_king_count(self):
        n = 0
        for p in self.b:
            if p != ES and p[1] != 'K':
                n += 1
        return n

    def _endgame(self):
        return self._non_king_count() <= 6

    # --------------------------------------------------------- evaluation
    def _evaluate_stm(self):
        v = self._evaluate_white()
        return v if self.wtm else -v

    def _mobility_bonus(self, color):
        my = color
        opp = 'b' if color == 'w' else 'w'
        my_legal = 0
        opp_legal = 0
        for i in range(48):
            p = self.b[i]
            if p == ES or p[0] != my:
                continue
            r, c = divmod(i, BW)
            if p[1] == 'P':
                dr = -1 if my == 'w' else 1
                nr = r + dr
                if 0 <= nr < BH and 0 <= c < BW and self.b[_idx(nr, c)] == ES:
                    my_legal += 1
                for dc in (-1, 1):
                    nc = c + dc
                    if 0 <= nr < BH and 0 <= nc < BW and self.b[_idx(nr, nc)][0] == opp:
                        my_legal += 1
            elif p[1] == 'N':
                my_legal += 8
            elif p[1] == 'B':
                my_legal += 7
            elif p[1] == 'R':
                my_legal += 6
            elif p[1] == 'K':
                my_legal += 5

        for i in range(48):
            p = self.b[i]
            if p == ES or p[0] != opp:
                continue
            r, c = divmod(i, BW)
            if p[1] == 'P':
                dr = -1 if opp == 'w' else 1
                nr = r + dr
                if 0 <= nr < BH and 0 <= c < BW and self.b[_idx(nr, c)] == ES:
                    opp_legal += 1
                for dc in (-1, 1):
                    nc = c + dc
                    if 0 <= nr < BH and 0 <= nc < BW and self.b[_idx(nr, nc)][0] == my:
                        opp_legal += 1
            elif p[1] == 'N':
                opp_legal += 8
            elif p[1] == 'B':
                opp_legal += 7
            elif p[1] == 'R':
                opp_legal += 6
            elif p[1] == 'K':
                opp_legal += 5
        return my_legal - opp_legal

    def _pawn_structure_bonus(self, color):
        score = 0
        for i in range(48):
            p = self.b[i]
            if p == ES or p[0] != color:
                continue
            if p[1] != 'P':
                continue
            r, c = divmod(i, BW)
            if color == 'w':
                score += r
            else:
                score += (BH - 1 - r)
        return score

    def _passed_pawn_bonus(self, color):
        score = 0
        foe = 'b' if color == 'w' else 'w'
        for i in range(48):
            p = self.b[i]
            if p == ES or p[0] != color or p[1] != 'P':
                continue
            r, c = divmod(i, BW)
            blocked = False
            for dc in (-1, 0, 1):
                nc = c + dc
                if 0 <= nc < BW:
                    rr = r + (1 if color == 'w' else -1)
                    if 0 <= rr < BH and self.b[_idx(rr, nc)] == foe + 'P':
                        blocked = True
            if blocked:
                continue
            score += 20 + (BH - 1 - r if color == 'w' else r)
        return score

    def _king_safety_bonus(self, color):
        king = self._find_king(color)
        if king < 0:
            return 0
        kr, kc = divmod(king, BW)
        score = 0
        for i in range(48):
            p = self.b[i]
            if p == ES or p[0] != color:
                continue
            if p[1] == 'K':
                continue
            r, c = divmod(i, BW)
            if abs(r - kr) <= 2 and abs(c - kc) <= 2:
                score += 2
        foe = 'b' if color == 'w' else 'w'
        for i in range(48):
            p = self.b[i]
            if p == ES or p[0] != foe:
                continue
            if p[1] == 'K':
                continue
            r, c = divmod(i, BW)
            if abs(r - kr) <= 2 and abs(c - kc) <= 2:
                score -= 2
        return score

    def _evaluate_white(self):
        """Static evaluation with light mobility, passed-pawn, and king-safety terms."""
        b = self.b
        total = 0
        w_king = b_king = -1
        w_pow = b_pow = 0
        w_nk = b_nk = 0
        for i in range(48):
            p = b[i]
            if p == ES:
                continue
            r = i // BW
            c = i - r * BW
            color, ptype = p[0], p[1]
            if ptype == 'K':
                if color == 'w':
                    w_king = i
                else:
                    b_king = i
                continue
            if color == 'w':
                w_pow += PIECE_VALUES[p]
                w_nk += 1
                total += PIECE_VALUES[p]
                if ptype in PST:
                    total += PST[ptype][r][c]
                if ptype in ('N', 'B', 'R'):
                    cr, cc = divmod(i, BW)
                    center_dist = abs(cr - 3) + abs(cc - 2)
                    total += max(0, 3 - center_dist)
            else:
                b_pow += -PIECE_VALUES[p]
                b_nk += 1
                total -= PIECE_VALUES[p]
                if ptype in PST:
                    mr = BH - 1 - r
                    total -= PST[ptype][mr][c]
                if ptype in ('N', 'B', 'R'):
                    cr, cc = divmod(i, BW)
                    center_dist = abs(cr - 3) + abs(cc - 2)
                    total -= max(0, 3 - center_dist)

        if w_king >= 0:
            wr, wc = divmod(w_king, BW)
            total += 4 - min(abs(wr - 3), 3) - min(abs(wc - 2), 3)
        if b_king >= 0:
            br, bc = divmod(b_king, BW)
            total -= 4 - min(abs(br - 3), 3) - min(abs(bc - 2), 3)

        total += self._mobility_bonus('w') * 2
        total -= self._mobility_bonus('b') * 2
        total += self._pawn_structure_bonus('w') // 3
        total -= self._pawn_structure_bonus('b') // 3
        total += self._passed_pawn_bonus('w')
        total -= self._passed_pawn_bonus('b')
        total += self._king_safety_bonus('w')
        total -= self._king_safety_bonus('b')

        if w_nk + b_nk <= 6 and w_king >= 0 and b_king >= 0:
            wr, wc = divmod(w_king, BW)
            br, bc = divmod(b_king, BW)
            total += KING_PST[wr][wc]
            total -= KING_PST[BH - 1 - br][bc]

        if w_pow - b_pow >= 100 and b_nk <= 1 and b_king >= 0 and w_king >= 0:
            total += self._drive_bonus('w', w_king, b_king)
        elif b_pow - w_pow >= 100 and w_nk <= 1 and w_king >= 0 \
                and b_king >= 0:
            total -= self._drive_bonus('b', b_king, w_king)
        return total

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
            if b[i] == leader + 'R':
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



