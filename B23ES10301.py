"""
General-Purpose High-Performance Alpha-Beta Engine for Spartans Chess (6x8).
Roll Number: B23ES10301 (Bug-Free Tournament Edition)
"""
import time
from config import (
    PIECE_VALUES, PAWN_PST, KNIGHT_PST, BISHOP_PST, ROOK_PST, 
    KING_PST_LATE_GAME, EMPTY_SQUARE
)

MATE = 100000

PST_MAP = {
    'P': PAWN_PST,
    'N': KNIGHT_PST,
    'B': BISHOP_PST,
    'R': ROOK_PST,
    'K': KING_PST_LATE_GAME
}

VICTIM_SCORES = {'P': 100, 'N': 300, 'B': 300, 'R': 500, 'K': 9000}
ATTACKER_SCORES = {'P': 10, 'N': 20, 'B': 20, 'R': 30, 'K': 50}

class TimeoutException(Exception):
    pass

class B23ES10301:
    def __init__(self, engine, depth=12):
        self.engine = engine
        self.depth = depth
        self.max_depth = depth
        self.nodes_expanded = 0
        self.time_used = 0.0

    def evaluate_board(self, piece_count):
        board = self.engine.board
        board_height = len(board)
        board_width = len(board[0])
        
        white_score = 0
        w_pow = 0
        b_pow = 0
        w_king_pos = None
        b_king_pos = None

        for r in range(board_height):
            for c in range(board_width):
                piece = board[r][c]
                if piece == EMPTY_SQUARE or not piece:
                    continue
                
                ptype = piece[1]
                is_white = piece.startswith('w')
                val = PIECE_VALUES.get(piece, 0)
                abs_val = abs(val)

                if is_white:
                    w_pow += abs_val
                else:
                    b_pow += abs_val

                # Safe PST Lookup using dynamically bounded board height
                pst = PST_MAP.get(ptype)
                pst_val = 0
                if pst:
                    pst_row = r if is_white else (board_height - 1 - r)
                    if 0 <= pst_row < len(pst) and 0 <= c < len(pst[0]):
                        pst_val = pst[pst_row][c]
                        if not is_white:
                            pst_val = -pst_val

                if ptype == 'K':
                    if is_white:
                        w_king_pos = (r, c)
                    else:
                        b_king_pos = (r, c)

                # King safety penalty scaled dynamically to actual board size
                if ptype == 'K' and piece_count > 8:
                    safe_row = board_height - 2
                    if is_white and r < safe_row:
                        pst_val -= 40
                    elif not is_white and r > 1:
                        pst_val += 40
                
                white_score += (val + pst_val)

        if w_king_pos and b_king_pos and piece_count <= 6:
            if w_pow - b_pow >= 150:
                white_score += self._mate_drive_bonus(w_king_pos, b_king_pos, board_height, board_width)
            elif b_pow - w_pow >= 150:
                white_score -= self._mate_drive_bonus(b_king_pos, w_king_pos, board_height, board_width)

        return white_score if self.engine.white_to_move else -white_score

    def _mate_drive_bonus(self, winner_king, loser_king, h, w):
        lr, lc = loser_king
        wr, wc = winner_king
        score = 0
        
        edge_dist_r = min(lr, (h - 1) - lr)
        edge_dist_c = min(lc, (w - 1) - lc)
        score += ((h + w) - (edge_dist_r + edge_dist_c)) * 15
        
        dist = max(abs(wr - lr), abs(wc - lc))
        score += (8 - dist) * 10
        return score

    def _score_move(self, m):
        target_piece = self.engine.board[m.end_row][m.end_col]
        if target_piece != EMPTY_SQUARE and target_piece:
            vic = target_piece[1]
            att = m.piece_moved[1]
            return 10000 + VICTIM_SCORES.get(vic, 0) - ATTACKER_SCORES.get(att, 0)
        return 0

    def get_best_move(self):
        t0 = time.monotonic()
        self.nodes_expanded = 0
        
        legal_moves = self.engine.get_legal_moves()
        if not legal_moves:
            return None

        time_left = max(0.2, 60.0 - self.time_used)
        turn_budget = min(1.3, time_left / 12.0)
        deadline = t0 + turn_budget

        piece_count = sum(
            1 for row in self.engine.board for p in row if p != EMPTY_SQUARE
        )

        best_move_idx = 0

        for current_depth in range(1, self.max_depth + 1):
            if time.monotonic() >= deadline:
                break

            legal_moves = self.engine.get_legal_moves()
            if not legal_moves:
                break

            # Safe ordering using tuple indices instead of direct object comparisons
            ordered_moves = sorted(
                enumerate(legal_moves),
                key=lambda x: (1 if x[0] == best_move_idx else 0, self._score_move(x[1])),
                reverse=True
            )

            try:
                alpha = -MATE * 2
                beta = MATE * 2
                depth_best_idx = None
                depth_best_score = -MATE * 2

                for orig_idx, move in ordered_moves:
                    self.engine.make_move(move)
                    score = -self.alphabeta(
                        current_depth - 1, -beta, -alpha, 
                        piece_count, deadline
                    )
                    self.engine.undo_move()

                    if score > depth_best_score:
                        depth_best_score = score
                        depth_best_idx = orig_idx

                    alpha = max(alpha, score)

                if depth_best_idx is not None:
                    best_move_idx = depth_best_idx

            except TimeoutException:
                break

        self.time_used += (time.monotonic() - t0)
        
        # ALWAYS fetch fresh legal moves and return via direct index selection
        final_legal_moves = self.engine.get_legal_moves()
        if not final_legal_moves:
            return None
            
        if best_move_idx < len(final_legal_moves):
            return final_legal_moves[best_move_idx]
        return final_legal_moves[0]

    def alphabeta(self, depth, alpha, beta, piece_count, deadline):
        self.nodes_expanded += 1

        if (self.nodes_expanded & 1023) == 0:
            if time.monotonic() >= deadline:
                raise TimeoutException()

        legal_moves = self.engine.get_legal_moves()

        if not legal_moves:
            if self.engine.is_in_check():
                return -MATE - depth
            return 0

        if depth == 0:
            return self.quiescence(alpha, beta, piece_count, deadline, qdepth=0)

        legal_moves.sort(key=self._score_move, reverse=True)
        best_score = -MATE * 2

        for move in legal_moves:
            self.engine.make_move(move)
            try:
                score = -self.alphabeta(
                    depth - 1, -beta, -alpha, 
                    piece_count, deadline
                )
            finally:
                self.engine.undo_move()

            best_score = max(best_score, score)
            alpha = max(alpha, score)

            if alpha >= beta:
                break

        return best_score

    def quiescence(self, alpha, beta, piece_count, deadline, qdepth=0):
        self.nodes_expanded += 1
        if (self.nodes_expanded & 1023) == 0:
            if time.monotonic() >= deadline:
                raise TimeoutException()

        stand_pat = self.evaluate_board(piece_count)
        if qdepth >= 4:
            return stand_pat

        if stand_pat >= beta:
            return stand_pat
        if stand_pat > alpha:
            alpha = stand_pat

        legal_moves = self.engine.get_legal_moves()
        capture_moves = [
            m for m in legal_moves 
            if self.engine.board[m.end_row][m.end_col] != EMPTY_SQUARE 
            and self.engine.board[m.end_row][m.end_col]
        ]

        if not capture_moves:
            return alpha

        capture_moves.sort(key=self._score_move, reverse=True)

        for move in capture_moves:
            captured_piece = self.engine.board[move.end_row][move.end_col]
            cap_val = abs(PIECE_VALUES.get(captured_piece, 0))

            if stand_pat + cap_val + 200 < alpha:
                continue

            self.engine.make_move(move)
            try:
                score = -self.quiescence(
                    -beta, -alpha, piece_count, deadline, qdepth + 1
                )
            finally:
                self.engine.undo_move()

            if score > alpha:
                alpha = score
            if alpha >= beta:
                return alpha

        return alpha