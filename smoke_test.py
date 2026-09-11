"""Smoke test: play the two sample (random) agents headlessly for up to N plies
to validate engine + agent API end to end.  Uses only the engine interface the
tournament runner uses."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from board import GameEngine, Move            # noqa: E402
from config import EMPTY_SQUARE, PIECE_VALUES # noqa: E402
from p22cs201 import P22CS201                  # noqa: E402
from p25cs0004 import P25CS0004                # noqa: E402

MAX_PLIES = 30  # keep the smoke test short (P22CS201 sleeps 0.2s/move)


def main():
    engine = GameEngine()
    white = P25CS0004(engine)   # random, instant
    black = P22CS201(engine)    # random, 0.2s delay
    white_points, black_points = 0, 0

    print(f"Engine board: {len(engine.board)}x{len(engine.board[0])}")
    print(f"State check methods: {[m for m in dir(engine) if not m.startswith('_')]}")

    for ply in range(1, MAX_PLIES + 1):
        state = engine.get_game_state()
        if state != "ongoing":
            print(f"Game over at ply {ply}: {state}")
            break

        player = white if engine.white_to_move else black
        move = player.get_best_move()
        if move is None:
            print(f"Ply {ply}: {player.__class__.__name__} returned None -> resign/stop")
            break

        assert isinstance(move, Move), f"Not a Move: {move!r}"
        engine.make_move(move)
        if move.piece_captured != EMPTY_SQUARE:
            pts = abs(PIECE_VALUES.get(move.piece_captured, 0))
            if engine.white_to_move:
                black_points += pts
            else:
                white_points += pts
        if ply <= 6 or ply % 10 == 0:
            color = "White" if not engine.white_to_move else "Black"
            print(f"Ply {ply:2d}: {color} {move.piece_moved} "
                  f"({move.start_row},{move.start_col})->({move.end_row},{move.end_col}) "
                  f"captured={move.piece_captured}")

    final = engine.get_game_state()
    print(f"\nFinal state after {MAX_PLIES} plies: {final}")
    print(f"White points (captures): {white_points}, Black points: {black_points}")
    print(f"move_log length: {len(engine.move_log)}, legal moves now: "
          f"{len(engine.get_legal_moves())}")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
