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
from B23ES1030 import B23ES1030             # noqa: E402
CAP = 150
PLAYER_LIMIT = 60.0       # strict 1-minute clock per player; running out = loss


def make(cls, engine):
    # B23CS1001 is built exactly the way the official runner builds it: its
    # clock, its time control and its depth cap all live inside B23CS1001.py
    # (the submission is a single self-contained file), so nothing here
    # overrides them.
    return cls(engine)


def play(white_cls, black_cls):
    engine = GameEngine()
    white = make(white_cls, engine)
    black = make(black_cls, engine)
    wp = bp = 0            # captured-piece points
    wchk = bchk = 0        # check points (+2 per check)
    # Strict 1-minute clock per player.  Each player's own thinking time is
    # accumulated; whoever runs out of time loses the game.
    clocks = {"White": 0.0, "Black": 0.0}
    timeout_loser = None
    for ply in range(1, CAP + 1):
        state = engine.get_game_state()
        if state != "ongoing":
            break
        side = "White" if engine.white_to_move else "Black"
        if clocks[side] >= PLAYER_LIMIT:
            timeout_loser = side                 # already out of time
            break
        player = white if engine.white_to_move else black
        t_move = time.perf_counter()
        mv = player.get_best_move()
        clocks[side] += time.perf_counter() - t_move
        if clocks[side] > PLAYER_LIMIT:
            timeout_loser = side                 # ran out while thinking -> loses
            break
        if mv is None:
            break
        legal = engine.get_legal_moves()
        assert any(mv == m for m in legal), \
            f"ILLEGAL MOVE {mv} by {type(player).__name__}"
        mover_white = engine.white_to_move
        engine.make_move(mv)
        if mv.piece_captured != EMPTY_SQUARE:
            pts = abs(PIECE_VALUES.get(mv.piece_captured, 0))
            if mover_white:
                wp += pts
            else:
                bp += pts
        if engine.is_in_check():        # the mover gave check (+2)
            if mover_white:
                wchk += 2
            else:
                bchk += 2
    final = engine.get_game_state()
    winner = "-"
    if timeout_loser is not None:
        winner = "Black" if timeout_loser == "White" else "White"
    elif final == "checkmate":
        winner = "Black" if engine.white_to_move else "White"
    # Official assignment scoring (AI_Assignment_I.pdf "Match Score Card"):
    #   checkmate      -> winner 600, loser 0 (overrides captured points)
    #   loss           -> 0   (a time-out counts as a loss)
    #   draw/stalemate -> Points table (captures + 2 per check)
    if timeout_loser is not None or final == "checkmate":
        ow, ob = (600, 0) if winner == "White" else (0, 600)
    else:
        ow, ob = wp + wchk, bp + bchk
    w = type(white).__name__
    b = type(black).__name__
    shown = "TIMEOUT" if timeout_loser is not None else final
    note = f"  <-- {timeout_loser} ran out of time" if timeout_loser else ""
    print(f"{w:>12} vs {b:<12} -> {shown:>9} (winner {winner:>5}) "
          f"ply {ply:3d} | official {ow:>4} vs {ob:<4} (diff {ow - ob:>4})"
          f" | caps {wp}-{bp} chk {wchk}-{bchk}"
          f" | clocks W {clocks['White']:4.1f}s B {clocks['Black']:4.1f}s{note}",
          flush=True)


def main():
    for pair in [
                 (B23CS1001, B23ES1030), (B23ES1030, B23ME1074), (B23CS1001, B23ME1074),
                 (B23ME1074, B23CS1001), ]:
        play(*pair)
    print("FAST GAUNTLET DONE", flush=True)


if __name__ == "__main__":
    main()
