"""Spartans 6x8 chess agent, developed from the official ai_player.py template.

Search: iterative-deepening negamax, alpha-beta, capture quiescence, bounded
transpositions and move ordering. All hypothetical play uses a private engine.
Evaluation is White-relative; recursive scores are side-to-move-relative.
Submission: B23ME1074.py, class B23ME1074.
"""

import copy
import time


class _SearchTimeout(Exception):
    """Abort only the unfinished iteration; finally blocks restore private state."""


class B23ME1074:
    # Internal units are ten times the assignment's non-king capture values.
    VALUES = {"P": 200, "N": 700, "B": 700, "R": 1000, "K": 0}
    MATE = 1_000_000
    INF = 2_000_000
    GAME_SECONDS = 60.0
    RESERVE_SECONDS = 8.0
    MAX_MOVE_SECONDS = 1.2
    MAX_DEPTH = 12
    QUIESCENCE_DEPTH = 4
    MAX_QUIESCENCE_PLY = 12
    TABLE_LIMIT = 20_000
    USE_TABLE = True

    def __init__(self, engine):
        # The template's "Board" import and undefined "engine" assignment
        # were inconsistent with the supplied GameEngine interface.
        self.engine = engine
        self.nodes_expanded = 0
        self.depth = 0
        self.time_used = 0.0
        self.last_score = 0
        self.last_elapsed = 0.0
        self._deadline = float("inf")
        self._position = None
        self._table = {}
        self._killers = {}
        self._history = {}
        self._last_root_ply = -1
        self._game_marker = None
        self._root_ply = 0

    @staticmethod
    def _move_key(move):
        return (move.start_row, move.start_col, move.end_row, move.end_col)

    @staticmethod
    def _board_key(position):
        return (tuple(tuple(row) for row in position.board),
                position.white_to_move)

    def _check_deadline(self):
        if time.perf_counter() >= self._deadline:
            raise _SearchTimeout

    def _allocation(self, root_ply, white):
        completed = (root_ply + 1) // 2 if white else root_ply // 2
        moves_left = max(1, 75 - completed)
        available = max(0.0, self.GAME_SECONDS - self.RESERVE_SECONDS
                        - self.time_used)
        return min(self.MAX_MOVE_SECONDS, available / moves_left)

    def get_best_move(self):
        """Return a legal official Move without changing the supplied engine."""
        started = time.perf_counter()
        marker = (id(self.engine), id(self.engine.move_log))
        root_ply = len(self.engine.move_log)
        if marker != self._game_marker or root_ply < self._last_root_ply:
            self.time_used = 0.0
            self._game_marker = marker
        self._last_root_ply = root_ply
        self._root_ply = root_ply
        self.nodes_expanded = 0
        self.depth = 0
        self.last_score = 0
        self._table.clear()
        self._killers.clear()
        self._history.clear()
        # Color is read on our turn, not in __init__: both players are
        # constructed while the initial shared board has White to move.
        allocation = self._allocation(root_ply, self.engine.white_to_move)
        self._deadline = started + max(0.0, allocation - 0.008)
        try:
            self._position = copy.deepcopy(self.engine)
            legal = self._position.get_legal_moves()
            if not legal:
                return None
            # Ordering only breaks equal search scores in favor of less
            # repeated positions. No automatic repetition draw is invented.
            legal = self._root_order(legal)
            best_move = legal[0]
            if len(legal) == 1 or root_ply >= 150:
                return best_move
            for depth in range(1, self.MAX_DEPTH + 1):
                iteration_start = time.perf_counter()
                if iteration_start >= self._deadline:
                    break
                try:
                    candidate, score = self._search_root(legal, depth, best_move)
                    self._check_deadline()
                except _SearchTimeout:
                    break
                best_move, self.last_score = candidate, score
                self.depth = depth
                if abs(score) >= self.MATE - 200:
                    break
                elapsed = time.perf_counter() - iteration_start
                # Avoid starting a much deeper iteration with little time left.
                if time.perf_counter() + elapsed * 2.0 >= self._deadline:
                    break
            return best_move
        finally:
            # Release potentially large private history/cache objects while
            # still on our own clock, and account for setup and unwinding too.
            self._position = None
            self._table.clear()
            self.last_elapsed = time.perf_counter() - started
            self.time_used += self.last_elapsed

    def _root_order(self, legal):
        position = self._position
        visits = {}
        for move in legal:
            position.make_move(move)
            try:
                key = self._board_key(position)
                visits[self._move_key(move)] = self.engine.position_history.get(key, 0)
            finally:
                position.undo_move()
        ordered = self._ordered_moves(legal, None, 0)
        return sorted(ordered, key=lambda move: visits[self._move_key(move)])

    def _search_root(self, legal, depth, previous):
        position = self._position
        ordered = sorted(legal, key=lambda move: move != previous)
        alpha = -self.INF
        best = ordered[0]
        first = True
        for move in ordered:
            self._check_deadline()
            position.make_move(move)
            try:
                if first:
                    value = -self._negamax(depth - 1, -self.INF, -alpha, 1)
                    first = False
                else:
                    # Most root moves are expected to be worse than the
                    # current principal variation. Use a zero-width window
                    # first and re-search only moves that beat alpha.
                    value = -self._negamax(depth - 1, -alpha - 1, -alpha, 1)
                    if value > alpha and value < self.INF:
                        value = -self._negamax(depth - 1, -self.INF,
                                              -alpha, 1)
            finally:
                position.undo_move()
            if value > alpha:
                alpha, best = value, move
        return best, alpha

    def _terminal(self, legal, in_check, ply):
        if not legal:
            return -self.MATE + ply if in_check else 0
        # Mate on the last allowed ply takes precedence over the limit.
        if self._root_ply + ply >= 150:
            return 0
        return None

    def _negamax(self, depth, alpha, beta, ply):
        self._check_deadline()
        if depth <= 0:
            return self._quiescence(alpha, beta, ply, 0)
        self.nodes_expanded += 1
        position = self._position
        original_alpha, original_beta = alpha, beta
        # Remaining horizon is part of the key. Repetition/capture history
        # does not enter recursive evaluation, so it is not needed here.
        key = (self._board_key(position), self._root_ply + ply)
        entry = self._table.get(key) if self.USE_TABLE else None
        preferred = entry[3] if entry else None
        if entry and entry[0] >= depth:
            score = self._from_table(entry[1], ply)
            if entry[2] == "EXACT":
                return score
            if entry[2] == "LOWER":
                alpha = max(alpha, score)
            else:
                beta = min(beta, score)
            if alpha >= beta:
                return score
        legal = position.get_legal_moves()
        in_check = position.is_in_check()
        terminal = self._terminal(legal, in_check, ply)
        if terminal is not None:
            return terminal
        best_score, best_key = -self.INF, None
        for move in self._ordered_moves(legal, preferred, ply):
            self._check_deadline()
            position.make_move(move)
            try:
                score = -self._negamax(depth - 1, -beta, -alpha, ply + 1)
            finally:
                position.undo_move()
            if score > best_score:
                best_score, best_key = score, self._move_key(move)
            alpha = max(alpha, score)
            if alpha >= beta:
                if move.piece_captured == "--":
                    killers = self._killers.setdefault(ply, [])
                    move_key = self._move_key(move)
                    if move_key not in killers:
                        killers.insert(0, move_key)
                        del killers[2:]
                    # The cutoff move is the useful quiet-move ordering signal.
                    hkey = (move.piece_moved, move_key)
                    self._history[hkey] = min(
                        10_000, self._history.get(hkey, 0) + depth * depth)
                break
        if self.USE_TABLE and len(self._table) < self.TABLE_LIMIT:
            flag = ("UPPER" if best_score <= original_alpha else
                    "LOWER" if best_score >= original_beta else "EXACT")
            self._table[key] = (depth, self._to_table(best_score, ply),
                                flag, best_key)
        return best_score

    def _quiescence(self, alpha, beta, ply, qply):
        self._check_deadline()
        self.nodes_expanded += 1
        position = self._position
        legal = position.get_legal_moves()
        in_check = position.is_in_check()
        terminal = self._terminal(legal, in_check, ply)
        if terminal is not None:
            return terminal
        if in_check:
            if qply >= self.MAX_QUIESCENCE_PLY:
                # Never use stand-pat while checked: keep the previously
                # completed iteration if a checking chain exhausts this cap.
                raise _SearchTimeout
            candidates = legal
            best = -self.INF
        else:
            best = self._evaluate(position)
            if not position.white_to_move:
                best = -best
            if best >= beta:
                return best
            alpha = max(alpha, best)
            if qply >= self.QUIESCENCE_DEPTH:
                return best
            candidates = [m for m in legal if m.piece_captured != "--"]
        for move in self._ordered_moves(candidates, None, ply):
            self._check_deadline()
            position.make_move(move)
            try:
                score = -self._quiescence(-beta, -alpha, ply + 1, qply + 1)
            finally:
                position.undo_move()
            best = max(best, score)
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return best

    def _ordered_moves(self, moves, preferred, ply):
        killers = self._killers.get(ply, ())
        def priority(move):
            key = self._move_key(move)
            if key == preferred:
                return 1_000_000
            if move.piece_captured != "--":
                victim = self.VALUES[move.piece_captured[1]]
                attacker = self.VALUES[move.piece_moved[1]]
                return 100_000 + 16 * victim - attacker
            if key in killers:
                return 50_000 - killers.index(key)
            return self._history.get((move.piece_moved, key), 0)
        return sorted(moves, key=priority, reverse=True)

    def _to_table(self, score, ply):
        if score > self.MATE - 1000:
            return score + ply
        if score < -self.MATE + 1000:
            return score - ply
        return score

    def _from_table(self, score, ply):
        if score > self.MATE - 1000:
            return score - ply
        if score < -self.MATE + 1000:
            return score + ply
        return score

    def evaluate_board(self, game_state=None):
        """White-relative heuristic; optional state supports sample-style calls."""
        if game_state == "checkmate":
            return -self.MATE if self.engine.white_to_move else self.MATE
        if game_state == "stalemate":
            return 0
        return self._evaluate(self.engine)

    def _evaluate(self, position):
        board = position.board
        height, width = len(board), len(board[0])
        material = [0, 0]
        nonpawn = [0, 0]
        pawns = [[], []]
        pieces = []
        kings = [None, None]
        files = [[0] * width for _ in range(2)]
        for r, row in enumerate(board):
            for c, piece in enumerate(row):
                if piece == "--":
                    continue
                side = 0 if piece[0] == "w" else 1
                kind = piece[1]
                if kind == "K":
                    kings[side] = (r, c)
                else:
                    material[side] += self.VALUES[kind]
                    if kind == "P":
                        pawns[side].append((r, c))
                        files[side][c] += 1
                    else:
                        nonpawn[side] += self.VALUES[kind]
                pieces.append((r, c, side, kind))
        if kings[0] is None:
            return -self.MATE
        if kings[1] is None:
            return self.MATE
        phase = min(1.0, sum(nonpawn) / 8200.0)
        scores = [float(material[0]), float(material[1])]
        for r, c, side, kind in pieces:
            color = "w" if side == 0 else "b"
            direction = -1 if side == 0 else 1
            relative_row = r if side == 0 else height - 1 - r
            center = abs(c - (width - 1) / 2) + abs(r - (height - 1) / 2)
            value = 0.0
            if kind == "P":
                # Advancement creates space, never a promotion reward.
                value += (0, 4, 10, 16, 18, 12, 0, -15)[height - 1 - relative_row]
                value += 4 * ((width - 1) / 2 - abs(c - (width - 1) / 2))
                if files[side][c] > 1:
                    value -= 12
                if not any(files[side][f] for f in (c - 1, c + 1)
                           if 0 <= f < width):
                    value -= 8
                for dc in (-1, 1):
                    sr, sc = r - direction, c + dc
                    if 0 <= sr < height and 0 <= sc < width:
                        if board[sr][sc] == color + "P":
                            value += 9
                nr = r + direction
                if 0 <= nr < height and board[nr][c] != "--":
                    value -= 5
            elif kind == "N":
                value += 32 - 10 * center
                for dr, dc in ((2, 1), (2, -1), (-2, 1), (-2, -1),
                               (1, 2), (1, -2), (-1, 2), (-1, -2)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width:
                        if not board[nr][nc].startswith(color):
                            value += 3
            elif kind in ("B", "R"):
                directions = (((1, 1), (1, -1), (-1, 1), (-1, -1))
                              if kind == "B" else
                              ((1, 0), (-1, 0), (0, 1), (0, -1)))
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    while 0 <= nr < height and 0 <= nc < width:
                        target = board[nr][nc]
                        if not target.startswith(color):
                            value += 3 if kind == "B" else 2
                        if target != "--":
                            break
                        nr, nc = nr + dr, nc + dc
                if kind == "B":
                    value += 10 - 3 * center
                else:
                    if not files[side][c]:
                        value += 16 if files[1 - side][c] else 28
                    if relative_row <= 1:
                        value += 18
            else:
                # King safety matters early; safe centralization matters late.
                safety = -12 * max(0, height - 2 - relative_row)
                for dc in (-1, 0, 1):
                    nr, nc = r + direction, c + dc
                    if 0 <= nr < height and 0 <= nc < width:
                        if board[nr][nc] == color + "P":
                            safety += 14
                value += phase * safety + (1 - phase) * (35 - 10 * center)
            scores[side] += value
        # The official scoring awards a small bonus for delivering check.
        # It remains subordinate to material and mate, but breaks otherwise
        # equal leaf evaluations in favor of useful checking moves.
        if position.is_in_check():
            if position.white_to_move:
                scores[1] += 20
            else:
                scores[0] += 20
        # In a materially won sparse ending, bring our king closer and
        # drive the opposing king toward an edge to help finish checkmate.
        difference = material[0] - material[1]
        if abs(difference) >= 500:
            winner = 0 if difference > 0 else 1
            loser = 1 - winner
            if nonpawn[loser] <= 1000:
                kr, kc = kings[loser]
                wr, wc = kings[winner]
                edge_distance = min(kr, height - 1 - kr, kc, width - 1 - kc)
                distance = max(abs(kr - wr), abs(kc - wc))
                scores[winner] += (1 - phase) * (
                    25 * (min(height, width) / 2 - edge_distance)
                    + 12 * (max(height, width) - distance))
        return int(round(scores[0] - scores[1]))

