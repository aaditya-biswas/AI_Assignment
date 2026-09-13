"""Endgame conversion tests: set up winning endgames and check the agent
delivers checkmate instead of shuffling to a draw."""
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from board import GameEngine                # noqa: E402
from config import EMPTY_SQUARE             # noqa: E402
from B23CS1001 import B23CS1001             # noqa: E402


def place(engine, pieces, white_to_move=True):
    """Reset engine to a custom position. pieces = {(r,c): 'wR', ...}."""
    board = [[EMPTY_SQUARE] * 6 for _ in range(8)]
    for (r, c), p in pieces.items():
        board[r][c] = p
    engine.board = board
    engine.white_to_move = white_to_move
    engine.move_log = []
    engine.position_history = {}
    engine.update_position_history()


def random_legal(engine, rng):
    moves = engine.get_legal_moves()
    return rng.choice(moves) if moves else None


def run_case(name, pieces, ply_cap=90, budget=0.5):
    rng = random.Random(1234)
    engine = GameEngine()
    place(engine, pieces, white_to_move=True)
    white = B23CS1001(engine, max_depth=8)
    white.test_budget = budget
    result = "capped"
    t0 = time.monotonic()
    for ply in range(1, ply_cap + 1):
        state = engine.get_game_state()
        if state != "ongoing":
            result = state
            break
        if engine.white_to_move:               # our agent plays White
            mv = white.get_best_move()
        else:                                   # random Black defender
            mv = random_legal(engine, rng)
        if mv is None:
            result = "no-moves"
            break
        engine.make_move(mv)
    dt = time.monotonic() - t0
    winner = ""
    if result == "checkmate":
        winner = "Black" if engine.white_to_move else "White"
    print(f"{name:12s} -> {result:>9} (winner {winner:>5}) ply={ply:3d} "
          f"time={dt:5.1f}s", flush=True)
    return result == "checkmate" and winner == "White"


def main():
    cases = [
        ("KR vs K (a)",
         {(7, 1): 'wK', (7, 4): 'wR', (3, 3): 'bK'}),
        ("KR vs K (b)",
         {(7, 2): 'wK', (6, 5): 'wR', (0, 0): 'bK'}),
        ("KRR vs K",
         {(7, 3): 'wK', (6, 0): 'wR', (5, 5): 'wR', (0, 4): 'bK'}),
        ("KBB vs K",
         {(7, 3): 'wK', (4, 2): 'wB', (4, 3): 'wB', (0, 5): 'bK'}),
        ("KR vs KP",
         {(7, 1): 'wK', (7, 4): 'wR', (0, 5): 'bK', (2, 5): 'bP'}),
    ]
    wins = 0
    for name, pieces in cases:
        if run_case(name, pieces):
            wins += 1
    print(f"ENDGAME TEST: {wins}/{len(cases)} converted to checkmate")


if __name__ == "__main__":
    main()
