import sys
import copy
import time
from config import *
from board import GameEngine, Move

MY_KING_SAFETY_TABLE = [
    [30, 40, 10, 10, 40, 30],
    [20, 20, 0, 0, 20, 20],
    [-10, -20, -20, -20, -20, -10],
    [-30, -40, -40, -40, -40, -30],
    [-30, -40, -40, -40, -40, -30],
    [-10, -20, -20, -20, -20, -10],
    [20, 20, 0, 0, 20, 20],
    [30, 40, 10, 10, 40, 30],
]

MIDGAME_MATERIAL_CUTOFF = 400

class SearchTimeExpired(Exception):
    pass

class AIPlayer:
    def __init__(self, board):
        self.engine = board
        self.nodes_expanded = 0
        self.depth = 1
        self.total_time_budget = 55.0
        self.time_used = 0.0
        self.moves_made = 0
        self.max_turns = 75
        self.floor_seconds = 0.35
        self.ceiling_seconds = 3.0
        self.deepest_allowed = 8

    def get_best_move(self):
        clock_start = time.time()
        options = self.engine.get_legal_moves()
        if not options:
            return None
        if len(options) == 1:
            self.moves_made += 1
            return options[0]

        slice_seconds = self._time_slice_for_this_move()
        chosen = self._rank_by_capture_value(options)[0]
        am_maximizing = self.engine.white_to_move

        d = 1
        try:
            while d <= self.deepest_allowed:
                _, found = self._search(
                    d, float('-inf'), float('inf'),
                    am_maximizing, clock_start, slice_seconds
                )
                if found is not None:
                    chosen = found
                d += 1
        except SearchTimeExpired:
            pass

        self.time_used += time.time() - clock_start
        self.moves_made += 1
        return chosen

    def evaluate_board(self):
        board = self.engine.board
        total = 0

        leftover_material = 0
        for row in board:
            for piece in row:
                if piece != EMPTY_SQUARE and piece[1] != 'K':
                    leftover_material += abs(PIECE_VALUES[piece])
        in_endgame = leftover_material < MIDGAME_MATERIAL_CUTOFF
        king_table = KING_PST_LATE_GAME if in_endgame else MY_KING_SAFETY_TABLE

        for r in range(BOARD_HEIGHT):
            for c in range(BOARD_WIDTH):
                piece = board[r][c]
                if piece == EMPTY_SQUARE:
                    continue

                total += PIECE_VALUES[piece]
                kind = piece[1]
                side = 1 if piece[0] == 'w' else -1

                if kind == 'P':
                    total += side * PAWN_PST[r][c]
                elif kind == 'N':
                    total += side * KNIGHT_PST[r][c]
                elif kind == 'B':
                    total += side * BISHOP_PST[r][c]
                elif kind == 'R':
                    total += side * ROOK_PST[r][c]
                elif kind == 'K':
                    total += side * king_table[r][c]

        if self.engine.is_in_check():
            total += -2 if self.engine.white_to_move else 2

        repeat_count = self.engine.get_repetition_count()
        if repeat_count > 1:
            nudge = 5 * (repeat_count - 1)
            if total > 0:
                total -= nudge
            elif total < 0:
                total += nudge

        return total

    def _search(self, depth_left, alpha, beta, am_maximizing, clock_start, slice_seconds):
        if time.time() - clock_start > slice_seconds:
            raise SearchTimeExpired()

        self.nodes_expanded += 1

        candidates = self.engine.get_legal_moves()
        if not candidates:
            if self.engine.is_in_check():
                if self.engine.white_to_move:
                    return -100000 - depth_left, None
                return 100000 + depth_left, None
            return 0, None

        if depth_left == 0:
            return self._settle_captures(alpha, beta, am_maximizing, clock_start, slice_seconds), None

        candidates = self._rank_by_capture_value(candidates)
        best_so_far = candidates[0]

        if am_maximizing:
            best_value = float('-inf')
            for candidate in candidates:
                self.engine.make_move(candidate)
                try:
                    reply_value, _ = self._search(
                        depth_left - 1, alpha, beta, False, clock_start, slice_seconds
                    )
                finally:
                    self.engine.undo_move()

                if reply_value > best_value:
                    best_value = reply_value
                    best_so_far = candidate
                alpha = max(alpha, best_value)
                if alpha >= beta:
                    break
            return best_value, best_so_far
        else:
            best_value = float('inf')
            for candidate in candidates:
                self.engine.make_move(candidate)
                try:
                    reply_value, _ = self._search(
                        depth_left - 1, alpha, beta, True, clock_start, slice_seconds
                    )
                finally:
                    self.engine.undo_move()

                if reply_value < best_value:
                    best_value = reply_value
                    best_so_far = candidate
                beta = min(beta, best_value)
                if alpha >= beta:
                    break
            return best_value, best_so_far

    def _settle_captures(self, alpha, beta, am_maximizing, clock_start, slice_seconds):
        if time.time() - clock_start > slice_seconds:
            raise SearchTimeExpired()

        self.nodes_expanded += 1

        candidates = self.engine.get_legal_moves()
        if not candidates:
            if self.engine.is_in_check():
                return -100000 if self.engine.white_to_move else 100000
            return 0

        standing_score = self.evaluate_board()
        if am_maximizing:
            if standing_score >= beta:
                return beta
            alpha = max(alpha, standing_score)
        else:
            if standing_score <= alpha:
                return alpha
            beta = min(beta, standing_score)

        capture_moves = self._rank_by_capture_value(
            [m for m in candidates if m.piece_captured != EMPTY_SQUARE]
        )

        for candidate in capture_moves:
            self.engine.make_move(candidate)
            try:
                result = self._settle_captures(
                    alpha, beta, not am_maximizing, clock_start, slice_seconds
                )
            finally:
                self.engine.undo_move()

            if am_maximizing:
                if result > alpha:
                    alpha = result
                if alpha >= beta:
                    return beta
            else:
                if result < beta:
                    beta = result
                if beta <= alpha:
                    return alpha

        return alpha if am_maximizing else beta

    def _rank_by_capture_value(self, moves):
        def value_of(move):
            victim = move.piece_captured
            if victim != EMPTY_SQUARE:
                return -abs(PIECE_VALUES.get(victim, 0))
            return 0
        return sorted(moves, key=value_of)

    def _time_slice_for_this_move(self):
        remaining = max(self.floor_seconds, self.total_time_budget - self.time_used)
        moves_left = max(1, self.max_turns - self.moves_made)
        slice_seconds = remaining / moves_left
        slice_seconds = min(slice_seconds, remaining, self.ceiling_seconds)
        slice_seconds = max(slice_seconds, self.floor_seconds)
        return slice_seconds