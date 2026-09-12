"""Gauntlet for the fast agent (ai_player.B23CS1001) vs sample players."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from board import GameEngine, Move          # noqa: E402
from config import EMPTY_SQUARE, PIECE_VALUES  # noqa: E402
from B23CS1001 import B23CS1001             # noqa: E402
from p22cs201 import P22CS201               # noqa: E402
from p25cs0004 import P25CS0004             # noqa: E402
from B23ME1074 import B23ME1074             # noqa: E402
BUDGET = 0.35
MAX_DEPTH = 8
CAP = 150


def make(cls, engine):
    if cls is B23CS1001:
        a = B23CS1001(engine, max_depth=MAX_DEPTH)
        a.test_budget = BUDGET
        return a
    return cls(engine)


def play(white_cls, black_cls):
    engine = GameEngine()
    white = make(white_cls, engine)
    black = make(black_cls, engine)
    wp = bp = 0
    for ply in range(1, CAP + 1):
        state = engine.get_game_state()
        if state != "ongoing":
            break
        player = white if engine.white_to_move else black
        mv = player.get_best_move()
        if mv is None:
            break
        legal = engine.get_legal_moves()
        assert any(mv == m for m in legal), \
            f"ILLEGAL MOVE {mv} by {type(player).__name__}"
        engine.make_move(mv)
        if mv.piece_captured != EMPTY_SQUARE:
            pts = abs(PIECE_VALUES.get(mv.piece_captured, 0))
            if engine.white_to_move:
                bp += pts
            else:
                wp += pts
    final = engine.get_game_state()
    winner = "-"
    if final == "checkmate":
        winner = "Black" if engine.white_to_move else "White"
    w = type(white).__name__
    b = type(black).__name__
    print(f"{w:>12} vs {b:<12} -> {final:>9} (winner {winner:>5}) "
          f"ply {ply:3d} | pts {wp:>4} vs {bp:<4} (diff {wp - bp:>4})",
          flush=True)


def main():
    for pair in [(B23CS1001, P25CS0004), (P25CS0004, B23CS1001),
                 (B23CS1001, B23ME1074), (B23ME1074, B23CS1001)]:
        play(*pair)
    print("FAST GAUNTLET DONE", flush=True)


if __name__ == "__main__":
    main()
