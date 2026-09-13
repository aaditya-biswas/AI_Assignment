import time
import math
from config import *

class B23ES1030:
    """
    Optimized Spartans Chess AI Agent for B23ES1030.
    Features Depth-Weighted Mate Incentives, Endgame Boxing Heuristics, 
    Pawn Push Boosts, and Repetition Penalties.
    """
    def __init__(self, engine):
        self.engine = engine
        self.nodes_expanded = 0
        self.depth = 1
        self.transposition_table = {}
        self.board_history = []
        
        # Internal clock tracker
        self.my_clock = 57.0  
        self.last_move_time = None

        # Hardcoded Opening Book coordinates: (start_col, start_row, end_col, end_row)
        self.white_book = [
            (3, 6, 3, 5), # d2 -> d3
            (0, 7, 1, 5), # Na1 -> b3
            (2, 6, 2, 5)  # c2 -> c3
        ]
        
        self.black_book = [
            (3, 1, 3, 2), # d7 -> d6
            (5, 1, 5, 2), # f7 -> f6
            (2, 1, 2, 2)  # c7 -> c6
        ]

    def get_best_move(self):
        now = time.time()
        
        if self.last_move_time is not None:
            elapsed_since_last = now - self.last_move_time
            if elapsed_since_last < 10.0:  
                self.my_clock -= elapsed_since_last

        self.nodes_expanded = 0
        legal_moves = self.engine.get_legal_moves()
        if not legal_moves:
            self.last_move_time = time.time()
            return None

        is_white = self.engine.white_to_move
        
        board_str = str(self.engine.board)
        self.board_history.append(board_str)
        if len(self.board_history) > 20:
            self.board_history.pop(0)

        board_pieces_count = sum(
            1 for r in range(BOARD_HEIGHT) for c in range(BOARD_WIDTH) 
            if self.engine.board[r][c] != EMPTY_SQUARE
        )
        current_turn = max(1, (36 - board_pieces_count) + 1)

        # Opening Book
        book = self.white_book if is_white else self.black_book
        book_index = (current_turn - 1) // 2 if is_white else (current_turn - 2) // 2
        
        if 0 <= book_index < len(book):
            sc, sr, ec, er = book[book_index]
            for m in legal_moves:
                if m.start_col == sc and m.start_row == sr and m.end_col == ec and m.end_row == er:
                    self.nodes_expanded = 1
                    self.last_move_time = time.time()
                    return m

        # Dynamic Time Allocation
        remaining_turns = max(1, 75 - current_turn)
        base_time = max(0.04, self.my_clock / remaining_turns)
        
        if self.my_clock < 5.0:
            time_limit = min(base_time * 0.4, 0.08)
        elif current_turn <= 15:
            time_limit = min(base_time * 0.8, 0.25)
        elif 16 <= current_turn <= 50:
            time_limit = min(base_time * 1.4, 1.1)
        else:
            time_limit = min(base_time * 0.8, 0.3)

        start_search_time = time.time()
        best_move = legal_moves[0]

        current_depth = 1
        while True:
            elapsed_search = time.time() - start_search_time
            if elapsed_search >= time_limit:
                break
                
            move_at_depth = self._search_at_depth(current_depth, start_search_time, time_limit)
            if move_at_depth:
                best_move = move_at_depth
            
            if current_depth >= 8 or elapsed_search > (time_limit * 0.55):
                break
            current_depth += 1

        self.depth = current_depth
        self.last_move_time = time.time()
        self.my_clock -= (self.last_move_time - start_search_time)
        
        return best_move

    def _search_at_depth(self, depth, start_time, time_limit):
        legal_moves = self.engine.get_legal_moves()
        ordered_moves = self._order_moves(legal_moves)
        
        best_move = None
        is_maximizing = self.engine.white_to_move
        alpha = -float('inf')
        beta = float('inf')

        if is_maximizing:
            best_score = -float('inf')
            for move in ordered_moves:
                if time.time() - start_time >= time_limit:
                    break
                self.engine.make_move(move)
                score = self.alpha_beta(depth - 1, alpha, beta, False, start_time, time_limit, 1)
                self.engine.undo_move()
                
                if score > best_score:
                    best_score = score
                    best_move = move
                alpha = max(alpha, best_score)
        else:
            best_score = float('inf')
            for move in ordered_moves:
                if time.time() - start_time >= time_limit:
                    break
                self.engine.make_move(move)
                score = self.alpha_beta(depth - 1, alpha, beta, True, start_time, time_limit, 1)
                self.engine.undo_move()
                
                if score < best_score:
                    best_score = score
                    best_move = move
                beta = min(beta, score)

        return best_move

    def alpha_beta(self, depth, alpha, beta, is_maximizing, start_time, time_limit, ply=0):
        self.nodes_expanded += 1
        
        if time.time() - start_time >= time_limit:
            return self.evaluate_board()

        board_hash = hash(str(self.engine.board))
        if board_hash in self.transposition_table:
            cached_depth, cached_score = self.transposition_table[board_hash]
            if cached_depth >= depth:
                return cached_score

        game_state = self.engine.get_game_state()
        if game_state == "checkmate":
            # Prefer faster checkmates (subtract ply)
            return (-20000 + ply) if is_maximizing else (20000 - ply)
        if game_state == "stalemate":
            return 0

        if depth <= 0:
            score = self.quiescence_search(alpha, beta, is_maximizing, start_time, time_limit)
            self.transposition_table[board_hash] = (depth, score)
            return score

        legal_moves = self._order_moves(self.engine.get_legal_moves())
        if not legal_moves:
            return self.evaluate_board()

        if is_maximizing:
            max_eval = -float('inf')
            for move in legal_moves:
                self.engine.make_move(move)
                eval_score = self.alpha_beta(depth - 1, alpha, beta, False, start_time, time_limit, ply + 1)
                self.engine.undo_move()
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            self.transposition_table[board_hash] = (depth, max_eval)
            return max_eval
        else:
            min_eval = float('inf')
            for move in legal_moves:
                self.engine.make_move(move)
                eval_score = self.alpha_beta(depth - 1, alpha, beta, True, start_time, time_limit, ply + 1)
                self.engine.undo_move()
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            self.transposition_table[board_hash] = (depth, min_eval)
            return min_eval

    def quiescence_search(self, alpha, beta, is_maximizing, start_time, time_limit, depth=2):
        stand_pat = self.evaluate_board()
        if depth <= 0 or (time.time() - start_time >= time_limit):
            return stand_pat

        if is_maximizing:
            if stand_pat >= beta:
                return beta
            alpha = max(alpha, stand_pat)
        else:
            if stand_pat <= alpha:
                return alpha
            beta = min(beta, stand_pat)

        legal_moves = self.engine.get_legal_moves()
        capture_moves = [m for m in legal_moves if getattr(m, 'piece_captured', EMPTY_SQUARE) != EMPTY_SQUARE]
        ordered_captures = self._order_moves(capture_moves)

        for move in ordered_captures:
            self.engine.make_move(move)
            score = self.quiescence_search(alpha, beta, not is_maximizing, start_time, time_limit, depth - 1)
            self.engine.undo_move()

            if is_maximizing:
                if score >= beta:
                    return beta
                alpha = max(alpha, score)
            else:
                if score <= alpha:
                    return alpha
                beta = min(beta, score)

        return alpha if is_maximizing else beta

    def _order_moves(self, moves):
        def score_move(move):
            score = 0
            captured = getattr(move, 'piece_captured', EMPTY_SQUARE)
            if captured != EMPTY_SQUARE:
                victim_val = abs(PIECE_VALUES.get(captured, 20))
                moved = getattr(move, 'piece_moved', EMPTY_SQUARE)
                attacker_val = abs(PIECE_VALUES.get(moved, 20))
                score += 1000 + (victim_val - attacker_val)
            return score

        return sorted(moves, key=score_move, reverse=True)

    def evaluate_board(self, game_state=None):
        if game_state is None:
            game_state = self.engine.get_game_state()

        if game_state == "checkmate":
            return -20000 if self.engine.white_to_move else 20000
        if game_state == "stalemate":
            return 0

        material_map = {'P': 20, 'N': 70, 'B': 70, 'R': 100, 'K': 600}
        
        w_mat, b_mat = 0, 0
        w_king_pos, b_king_pos = (0, 0), (0, 0)
        total_pieces = 0
        pawn_advance_score = 0
        
        board = self.engine.board

        for r in range(BOARD_HEIGHT):
            for c in range(BOARD_WIDTH):
                piece = board[r][c]
                if piece != EMPTY_SQUARE:
                    total_pieces += 1
                    color = piece[0]
                    p_type = piece[1]
                    val = material_map.get(p_type, 0)
                    
                    if color == 'w':
                        w_mat += val
                        if p_type == 'K':
                            w_king_pos = (r, c)
                        elif p_type == 'P':
                            # White pawns moving towards row 0
                            pawn_advance_score += (7 - r) * 5
                    else:
                        b_mat += val
                        if p_type == 'K':
                            b_king_pos = (r, c)
                        elif p_type == 'P':
                            # Black pawns moving towards row 7
                            pawn_advance_score -= r * 5

        total_score = (w_mat - b_mat) + pawn_advance_score

        # Repetition penalty
        b_str = str(board)
        if self.board_history.count(b_str) > 0:
            rep_penalty = 120 * self.board_history.count(b_str)
            total_score += (-rep_penalty if self.engine.white_to_move else rep_penalty)

        # High-Priority Endgame Cornering (Triggered when piece count dropping below 12)
        if total_pieces <= 12 or abs(w_mat - b_mat) >= 140:
            if w_mat > b_mat:
                # White driving Black King to edges
                b_king_center_dst = abs(b_king_pos[0] - 3.5) + abs(b_king_pos[1] - 2.5)
                king_distance = abs(w_king_pos[0] - b_king_pos[0]) + abs(w_king_pos[1] - b_king_pos[1])
                total_score += int(b_king_center_dst * 25 - king_distance * 15)
            elif b_mat > w_mat:
                # Black driving White King to edges
                w_king_center_dst = abs(w_king_pos[0] - 3.5) + abs(w_king_pos[1] - 2.5)
                king_distance = abs(w_king_pos[0] - b_king_pos[0]) + abs(w_king_pos[1] - b_king_pos[1])
                total_score -= int(w_king_center_dst * 25 - king_distance * 15)

        return total_score