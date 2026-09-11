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
        self.depth = depth          # reported/current depth
        self.max_depth = max_depth  # iterative deepening ceiling
        self.time_used = 0.0        # seconds consumed this game
        self.test_budget = None     # optional fixed budget for local tests
        self.max_ply = 60           # bounds check-extension chains
        self._deadline = None
        self._search_nodes = 0
        self.b = None               # flat 48-square internal board
        self.wtm = True             # white to move on the internal board

    # ------------------------------------------------------------- setup
    def _load(self):
        b = []
        for row in self.engine.board:
            b.extend(row)
        self.b = b
        self.wtm = self.engine.white_to_move

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
        cap = self.max_depth + (10 if drive else 8 if few else 0)
        try:
            for d in range(1, cap + 1):
                if time.monotonic() >= deadline:
                    break
                alpha, beta = -MATE * 2, MATE * 2
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

        # Ordering: (mate-drive) checks first, then captures MVV-LVA.
        ordered = sorted(legal, key=lambda m: self._move_key(m, drive))
        for fr, to, captured, gives in ordered:
            piece = b[fr]
            b[to] = piece
            b[fr] = ES
            self.wtm = not self.wtm
            try:
                score = -self._negamax(max(depth - 1, 0), -beta, -alpha,
                                       ply + 1)
            except _AbortSearch:
                b[fr] = piece
                b[to] = captured
                self.wtm = not self.wtm
                raise
            b[fr] = piece
            b[to] = captured
            self.wtm = not self.wtm
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break
        return alpha

    def _move_key(self, move, drive):
        fr, to, captured, gives = move
        if drive and gives:
            return -300000
        if captured != ES:
            victim = PT_VAL.get(captured[1], 0)
            return -100000 - victim * 100
        return 0

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

    def _evaluate_white(self):
        """White's static score: material + PST (+ endgame king terms)."""
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
            else:
                b_pow += -PIECE_VALUES[p]
                b_nk += 1
                total -= PIECE_VALUES[p]
                if ptype in PST:
                    mr = BH - 1 - r
                    total -= PST[ptype][mr][c]

        # Endgame king placement terms.
        if w_nk + b_nk <= 6 and w_king >= 0 and b_king >= 0:
            wr, wc = divmod(w_king, BW)
            br, bc = divmod(b_king, BW)
            total += KING_PST[wr][wc]
            total -= KING_PST[BH - 1 - br][bc]

        # Mate-drive terms: a side that leads by >= a rook against a bare (or
        # nearly bare) king is pushed to *win*, not just hoard material.
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



